# -*- coding: utf-8 -*-
"""
分段式3D检测训练器
支持teacher forcing -> scheduled sampling -> pure generation的渐进训练
多GPU、SwanLab日志、完整验证和checkpoint管理
"""

import os
import json
import time
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    warnings.warn("SwanLab not available. Install with: pip install swanlab")

from config_loader import ConfigLoader
from primitive_anything_3d import PrimitiveTransformer3D
from loss_3d import AdaptivePrimitiveTransformer3DLoss
from dataloader_3d import create_dataloader


@dataclass
class TrainingPhase:
    """训练阶段配置"""
    name: str
    epochs: int
    teacher_forcing_ratio: float    # 1.0=完全teacher forcing, 0.0=完全生成
    scheduled_sampling: bool        # 是否使用scheduled sampling
    sampling_strategy: str          # linear, exponential, inverse_sigmoid
    description: str


@dataclass
class TrainingStats:
    """训练统计信息"""
    epoch: int
    phase: str
    teacher_forcing_ratio: float
    train_loss: float
    train_classification_loss: float
    train_iou_loss: float
    train_delta_loss: float
    train_mean_iou: float
    val_loss: float
    val_generation_loss: float
    val_mean_iou: float
    val_generation_iou: float
    learning_rate: float
    adaptive_cls_weight: float
    adaptive_delta_weight: float
    epoch_time: float


class AdvancedTrainer:
    """高级3D检测训练器"""
    
    def __init__(
        self,
        config_loader: ConfigLoader,
        output_dir: str = "experiments",
        experiment_name: str = "primitive_3d_exp",
        resume_from: Optional[str] = None,
        local_rank: int = 0,
        world_size: int = 1,
        use_swanlab: bool = True,
        validation_samples: List[int] = None  # 指定用于验证可视化的样本索引
    ):
        """
        初始化训练器
        Args:
            config_loader: 配置加载器
            output_dir: 输出目录
            experiment_name: 实验名称
            resume_from: 恢复训练的checkpoint路径
            local_rank: 本地GPU排名
            world_size: 总GPU数量
            use_swanlab: 是否使用SwanLab
            validation_samples: 验证可视化样本索引列表
        """
        self.config_loader = config_loader
        self.local_rank = local_rank
        self.world_size = world_size
        
        self.is_main_process = (local_rank == 0)
        
        # 设置输出目录
        self.output_dir = Path(output_dir) / experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.validation_dir = self.output_dir / "validation_results"
        
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.validation_dir.mkdir(exist_ok=True)
        
        # 初始化设备
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(local_rank)
        
        # 加载配置
        self.model_config = config_loader.get_model_config()
        self.training_config = config_loader.get_training_config()
        
        # 初始化SwanLab - 只在主进程初始化
        self.use_swanlab = use_swanlab and SWANLAB_AVAILABLE and self.is_main_process
        if self.use_swanlab:
            if self.is_main_process:
                print(f"📊 初始化SwanLab日志...")
            swanlab.init(
                project="primitive-3d-detection",
                experiment_name=experiment_name,
                description="3D物体检测with分段式训练",
                config=dict(
                    model_config=self.model_config,
                    training_config=self.training_config
                )
            )
            if self.is_main_process:
                print(f"✅ SwanLab初始化完成")
        
        # 从配置文件加载训练阶段
        phases_config = config_loader.get_training_phases()
        self.training_phases = []
        
        for phase_name, phase_config in phases_config.items():
            self.training_phases.append(TrainingPhase(
                name=phase_name,
                epochs=phase_config.get('epochs', 15),
                teacher_forcing_ratio=phase_config.get('teacher_forcing_ratio', 1.0),
                scheduled_sampling=phase_config.get('scheduled_sampling', False),
                sampling_strategy=phase_config.get('sampling_strategy', "none"),
                description=phase_config.get('description', f"{phase_name}阶段")
            ))
        
        # 验证样本索引
        global_config = self.config_loader.get_global_config()
        self.validation_samples = validation_samples or global_config['logging']['validation_samples']
        
        # 训练状态
        self.current_epoch = 0
        self.current_phase_idx = 0
        self.best_val_loss = float('inf')
        self.best_generation_loss = float('inf')
        self.training_stats = []
        
        # 混合精度训练
        self.use_amp = self.training_config.get('mixed_precision', True)
        if self.use_amp:
            try:
                # 使用新的API（PyTorch 2.0+）
                self.scaler = torch.amp.GradScaler('cuda')
            except TypeError:
                # 回退到旧API
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # 训练优化配置
        opt_config = self.training_config.get('optimizations', {})
        
        # 提前停止配置
        early_stop_config = opt_config.get('early_stopping', {})
        self.enable_early_stopping = early_stop_config.get('enabled', True)
        self.eos_threshold = early_stop_config.get('eos_threshold', 0.5)
        self.adaptive_sequence_length = early_stop_config.get('adaptive_sequence_length', True)
        
        # 增量推理配置
        inference_config = opt_config.get('incremental_inference', {})
        self.use_incremental_inference = inference_config.get('enabled', True)
        self.incremental_temperature = inference_config.get('temperature', 1.0)
        
        # 梯度裁剪配置
        grad_clip_config = opt_config.get('gradient_clipping', {})
        self.use_grad_clipping = grad_clip_config.get('enabled', True)
        self.max_grad_norm = grad_clip_config.get('max_norm', 1.0)
        
        # PyTorch编译优化配置
        self.torch_compile_config = opt_config.get('torch_compile', {})
        
        # CuDNN优化配置
        cudnn_config = opt_config.get('cudnn_optimizations', {})
        if cudnn_config.get('benchmark', True):
            torch.backends.cudnn.benchmark = True
        if not cudnn_config.get('deterministic', True):
            torch.backends.cudnn.deterministic = False
        
        # 初始化模型、损失函数和数据加载器
        self._setup_model_and_data()
        
        # 恢复训练状态
        if resume_from:
            self._load_checkpoint(resume_from)
        
        if self.is_main_process:
            print(f"✅ 训练器初始化完成 (Rank {local_rank}/{world_size})")
            print(f"📁 输出目录: {self.output_dir}")
            print(f"🎯 训练阶段: {len(self.training_phases)}个阶段")
            print(f"📊 SwanLab日志: {'启用' if self.use_swanlab else '禁用'}")
            print(f"🖥️  GPU数量: {torch.cuda.device_count()}")
            print(f"🚀 分布式训练: {'启用' if (self.world_size > 1 and torch.cuda.device_count() > 1) else '禁用'}")
        else:
            print(f"🔄 工作进程启动 (Rank {local_rank}/{world_size})")
            print(f"   GPU设备: {self.device}")
            print(f"   等待主进程同步...")
        
        # 多GPU同步点
        if self.world_size > 1 and torch.cuda.device_count() > 1:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if self.is_main_process:
                    print(f"✅ 所有进程同步完成，开始训练...")
    
    def _setup_model_and_data(self):
        """设置模型、损失函数和数据加载器"""
        # 创建模型
        model_config = self.config_loader.get_model_config()
        global_config = self.config_loader.get_global_config()
        
        # 获取数据配置用于连续范围
        data_config = self.config_loader.get_data_config()
        continuous_ranges = data_config.get('continuous_ranges', {})
        
        # 从模型配置中获取各个部分
        discretization = model_config.get('discretization', {})
        embeddings = model_config.get('embeddings', {})
        transformer = model_config.get('transformer', {})
        image_encoder = model_config.get('image_encoder', {})
        conditioning = model_config.get('conditioning', {})
        advanced = model_config.get('advanced', {})
        
        # 验证必要的配置部分是否存在
        if not discretization:
            raise ValueError("模型配置中缺少 'discretization' 部分")
        if not embeddings:
            raise ValueError("模型配置中缺少 'embeddings' 部分")
        if not transformer:
            raise ValueError("模型配置中缺少 'transformer' 部分")
        if not image_encoder:
            raise ValueError("模型配置中缺少 'image_encoder' 部分")
        if not conditioning:
            raise ValueError("模型配置中缺少 'conditioning' 部分")
        if not advanced:
            raise ValueError("模型配置中缺少 'advanced' 部分")
        
        self.model = PrimitiveTransformer3D(
            # 离散化参数
            num_discrete_x=discretization['num_discrete_x'],
            num_discrete_y=discretization['num_discrete_y'],
            num_discrete_z=discretization['num_discrete_z'],
            num_discrete_w=discretization['num_discrete_w'],
            num_discrete_h=discretization['num_discrete_h'],
            num_discrete_l=discretization['num_discrete_l'],
            
            # 连续范围
            continuous_range_x=continuous_ranges['x'],
            continuous_range_y=continuous_ranges['y'],
            continuous_range_z=continuous_ranges['z'],
            continuous_range_w=continuous_ranges['w'],
            continuous_range_h=continuous_ranges['h'],
            continuous_range_l=continuous_ranges['l'],
            
            # 嵌入维度
            dim_x_embed=embeddings['dim_x_embed'],
            dim_y_embed=embeddings['dim_y_embed'],
            dim_z_embed=embeddings['dim_z_embed'],
            dim_w_embed=embeddings['dim_w_embed'],
            dim_h_embed=embeddings['dim_h_embed'],
            dim_l_embed=embeddings['dim_l_embed'],
            
            # 模型参数
            max_primitive_len=global_config['max_seq_len'],
            dim=transformer['dim'],
            attn_depth=transformer['depth'],
            attn_heads=transformer['heads'],
            attn_dim_head=transformer['dim_head'],
            attn_dropout=transformer['attn_dropout'],  # 注意力dropout
            ff_dropout=transformer['ff_dropout'],      # 前馈dropout
            
            # 图像编码器参数
            image_encoder_dim=image_encoder['output_dim'],
            use_fpn=image_encoder['use_fpn'],
            backbone=image_encoder['backbone'],
            pretrained=image_encoder['pretrained'],
            
            # 条件化配置
            condition_on_image=conditioning['condition_on_image'],
            gateloop_use_heinsen=advanced['gateloop_use_heinsen'],
            
            # 其他参数
            pad_id=global_config['pad_id']
        )
        self.model = self.model.to(self.device)
        
        # PyTorch 2.0 编译优化（如果启用）
        if hasattr(torch, 'compile') and self.torch_compile_config.get('enabled', False):
            compile_mode = self.torch_compile_config.get('mode', 'default')
            if self.is_main_process:
                print(f"🔥 启用PyTorch 2.0编译优化 (mode: {compile_mode})")
            self.model = torch.compile(self.model, mode=compile_mode)
        
        # 多GPU设置 - 只有在真正的多GPU环境下才启用DDP
        if self.world_size > 1 and torch.cuda.device_count() > 1:
            if self.is_main_process:
                print(f"🚀 启用分布式数据并行 (DDP)")
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            # 使用DDP包装模型
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            if self.is_main_process:
                print(f"✅ DDP模型包装完成")
        elif self.world_size > 1 and self.is_main_process:
            print(f"⚠️  检测到单GPU环境，禁用DDP")
        else:
            # 单GPU或world_size=1的情况
            if self.is_main_process:
                print(f"🖥️  单GPU训练模式")
        
        # 创建数据加载器 - 移除硬编码
        data_config = self.config_loader.get_data_config()
        global_config = self.config_loader.get_global_config()
        dataset_config = data_config['dataset']
        dataloader_config = data_config['dataloader']
        
        # 调试信息
        if self.is_main_process:
            print(f"🔍 数据加载器配置:")
            print(f"   原始batch_size: {dataloader_config.get('batch_size', 'N/A')}")
            print(f"   world_size: {self.world_size}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   数据集路径: {dataset_config.get('data_root', 'N/A')}")
        
        # 先创建数据集
        from dataloader_3d import Box3DDataset
        
        # 🔧 修复：直接传递配置参数，不依赖外部配置文件
        train_dataset = Box3DDataset(
            data_root=dataset_config['data_root'],
            stage="train",
            max_boxes=global_config['max_seq_len'],
            image_size=global_config['image_size'],
            continuous_ranges=data_config['continuous_ranges'],
            augmentation_config=data_config['augmentation']
        )
        
        val_dataset = Box3DDataset(
            data_root=dataset_config['data_root'],
            stage="val",
            max_boxes=global_config['max_seq_len'],
            image_size=global_config['image_size'],
            continuous_ranges=data_config['continuous_ranges'],
            augmentation_config={}  # 验证时不使用数据增强
        )
        
        # 分布式采样器 - 只有在真正的多GPU环境下才创建
        if self.world_size > 1 and torch.cuda.device_count() > 1:
            if self.is_main_process:
                print(f"🚀 启用分布式采样器")
            self.train_sampler = DistributedSampler(train_dataset)
            self.val_sampler = DistributedSampler(val_dataset, shuffle=False)
            if self.is_main_process:
                print(f"✅ 分布式采样器创建完成")
        else:
            if self.world_size > 1 and self.is_main_process:
                print(f"⚠️  检测到单GPU环境，禁用分布式采样器")
            self.train_sampler = None
            self.val_sampler = None
            if self.is_main_process:
                print(f"🖥️  使用普通数据加载器")
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        
        # 多GPU时调整batch_size
        original_batch_size = dataloader_config['batch_size']
        effective_batch_size = original_batch_size
        
        if self.world_size > 1 and torch.cuda.device_count() > 1:
            # 计算每GPU的batch_size
            effective_batch_size = max(1, original_batch_size // self.world_size)
            
            # 如果每GPU的batch_size太小，自动调整总batch_size
            if effective_batch_size < 1:
                effective_batch_size = 1
                adjusted_total_batch_size = self.world_size
                if self.is_main_process:
                    print(f"⚠️  原始batch_size={original_batch_size}过小，自动调整为总batch_size={adjusted_total_batch_size}")
            else:
                adjusted_total_batch_size = original_batch_size
            
            if self.is_main_process:
                print(f"📊 多GPU训练: 总batch_size={adjusted_total_batch_size}, 每GPU={effective_batch_size}")
        else:
            if self.is_main_process:
                print(f"📊 单GPU训练: batch_size={effective_batch_size}")
        
        # 最终验证
        if effective_batch_size < 1:
            print(f"❌ 错误: batch_size={effective_batch_size}，强制设置为1")
            effective_batch_size = 1
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),  # 如果没有采样器则shuffle
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=True,
            prefetch_factor=dataloader_config['prefetch_factor'] if dataloader_config['num_workers'] > 0 else None,
            persistent_workers=dataloader_config['persistent_workers'] if dataloader_config['num_workers'] > 0 else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=effective_batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            pin_memory=dataloader_config['pin_memory'],
            drop_last=False,
            prefetch_factor=dataloader_config['prefetch_factor'] if dataloader_config['num_workers'] > 0 else None,
            persistent_workers=dataloader_config['persistent_workers'] if dataloader_config['num_workers'] > 0 else False
        )
        
        # 优化器 - 移除硬编码
        optimizer_config = self.training_config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=optimizer_config['betas']
        )
        
        # 学习率调度器 - 支持多种调度器类型
        scheduler_config = self.training_config['scheduler']
        if scheduler_config['type'] == 'CosineAnnealingLR':
            total_epochs = sum(phase.epochs for phase in self.training_phases)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config['T_max'],
                eta_min=scheduler_config['eta_min']
            )
        elif scheduler_config['type'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config.get('T_mult', 2),
                eta_min=scheduler_config['eta_min']
            )
        else:
            self.scheduler = None
    
    def _compute_teacher_forcing_ratio(self, phase: TrainingPhase, epoch_in_phase: int) -> float:
        """计算当前的teacher forcing比例"""
        if not phase.scheduled_sampling:
            return phase.teacher_forcing_ratio
        
        # Scheduled sampling策略
        progress = epoch_in_phase / phase.epochs
        
        if phase.sampling_strategy == "linear":
            # 线性衰减
            return phase.teacher_forcing_ratio * (1 - progress)
        elif phase.sampling_strategy == "exponential":
            # 指数衰减
            return phase.teacher_forcing_ratio * (0.5 ** (progress * 5))
        elif phase.sampling_strategy == "inverse_sigmoid":
            # 反sigmoid衰减：从1.0平滑衰减到接近0
            k = 5  # 控制衰减速度
            # 调整公式确保从1.0开始衰减
            sigmoid_factor = 1 / (1 + math.exp(-k * (progress - 0.5)))
            return phase.teacher_forcing_ratio * (1 - sigmoid_factor)
        else:
            return phase.teacher_forcing_ratio
    
    def _create_loss_function(self, phase_name: str):
        """为不同训练阶段创建损失函数"""
        # 根据训练阶段映射到配置中的stage
        stage_mapping = {
            "teacher_forcing": "warmup",
            "scheduled_sampling": "main", 
            "pure_generation": "finetune"
        }
        
        # 创建损失函数
        loss_config = self.config_loader.get_loss_config()
        global_config = self.config_loader.get_global_config()
        data_config = self.config_loader.get_data_config()
        model_config = self.config_loader.get_model_config()
        
        # 移除硬编码，强制从配置读取
        base_weights = loss_config['base_weights']
        adaptive_weights = loss_config['adaptive_weights']
        continuous_ranges = data_config['continuous_ranges']
        discretization = model_config['discretization']
        algorithm_config = loss_config['algorithm']
        data_processing = loss_config['data_processing']
        
        return AdaptivePrimitiveTransformer3DLoss(
            # 离散化参数
            num_discrete_x=discretization['num_discrete_x'],
            num_discrete_y=discretization['num_discrete_y'],
            num_discrete_z=discretization['num_discrete_z'],
            num_discrete_w=discretization['num_discrete_w'],
            num_discrete_h=discretization['num_discrete_h'],
            num_discrete_l=discretization['num_discrete_l'],
            
            # 连续范围参数
            continuous_range_x=tuple(continuous_ranges['x']),
            continuous_range_y=tuple(continuous_ranges['y']),
            continuous_range_z=tuple(continuous_ranges['z']),
            continuous_range_w=tuple(continuous_ranges['w']),
            continuous_range_h=tuple(continuous_ranges['h']),
            continuous_range_l=tuple(continuous_ranges['l']),
            
            # 基础损失权重
            base_classification_weight=base_weights['classification'],
            iou_weight=base_weights['iou'],
            delta_weight=base_weights['delta'],
            eos_weight=base_weights['eos'],
            
            # 自适应权重参数
            adaptive_classification=adaptive_weights['adaptive_classification'],
            adaptive_delta=adaptive_weights['adaptive_delta'],
            min_classification_weight=adaptive_weights['classification_range']['min'],
            max_classification_weight=adaptive_weights['classification_range']['max'],
            min_delta_weight=adaptive_weights['delta_range']['min'],
            max_delta_weight=adaptive_weights['delta_range']['max'],
            iou_threshold_high=adaptive_weights['thresholds']['high'],
            iou_threshold_low=adaptive_weights['thresholds']['low'],
            
            # 数据处理参数
            pad_id=data_processing['pad_id'],
            label_smoothing=data_processing['label_smoothing'],
            
            # 算法参数
            distance_aware_cls=algorithm_config['distance_aware']['enabled'],
            distance_alpha=algorithm_config['distance_aware']['alpha'],
            focal_gamma=algorithm_config['focal']['gamma']
        )
    
    def _forward_with_sampling_strategy(self, batch: Dict, teacher_forcing_ratio: float) -> Dict:
        """根据采样策略进行前向传播
        
        Args:
            batch: 输入batch数据
            teacher_forcing_ratio: 采样比例
                - 1.0: 纯Teacher Forcing (100% GT)
                - 0.0~1.0: Scheduled Sampling (部分GT + 部分预测)
                - 0.0: 纯Generation (100% 预测)
        """
        rgbxyz = batch['image'].to(self.device)  # [B, 6, H, W]
        targets = {
            'x': batch['x'].to(self.device),
            'y': batch['y'].to(self.device),
            'z': batch['z'].to(self.device),
            'w': batch['w'].to(self.device),
            'h': batch['h'].to(self.device),
            'l': batch['l'].to(self.device),
        }
        
        # 获取模型
        if hasattr(self.model, 'module'):
            model = self.model.module
        else:
            model = self.model
        
        # 根据teacher_forcing_ratio选择不同的策略
        if teacher_forcing_ratio >= 1.0:
            # 完全teacher forcing：直接用GT
            inputs = targets
            # 统一使用forward_with_predictions
            outputs = model.forward_with_predictions(
                x=inputs['x'],
                y=inputs['y'],
                z=inputs['z'],
                w=inputs['w'],
                h=inputs['h'],
                l=inputs['l'],
                image=rgbxyz
            )
            return outputs
        elif teacher_forcing_ratio == 0.0:
            return self._forward_with_pure_generation(rgbxyz, targets, model)
        else:
            return self._forward_with_scheduled_sampling(rgbxyz, targets, teacher_forcing_ratio, model)
        
        
    
    def _forward_with_scheduled_sampling(self, rgbxyz: torch.Tensor, targets: Dict, teacher_forcing_ratio: float, model) -> Dict:
        """Scheduled Sampling实现 - 支持梯度传播"""
        batch_size = rgbxyz.size(0)
        seq_len = targets['x'].size(1)
        device = rgbxyz.device
        
        # ===== 使用支持梯度的增量生成获取预测序列 =====
        # 生成完整的预测序列（保持梯度）
        predicted_output = self._forward_with_gradient_preserving_generation(
            model, rgbxyz, targets, seq_len, device
        )
        
        # 从预测输出中提取连续值
        continuous_predictions = predicted_output['continuous_dict']
        
        # 计算序列长度和创建mask
        sequence_lengths = self._compute_sequence_lengths(targets)
        
        # 构建混合输入序列（保持梯度）
        mixed_inputs = {}
        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
            continuous_pred = continuous_predictions[f'{attr}_continuous']  # [B, seq_len]
            # 确保维度匹配GT
            target_seq_len = targets[attr].shape[1]
            if continuous_pred.shape[1] > target_seq_len:
                continuous_pred = continuous_pred[:, :target_seq_len]
            elif continuous_pred.shape[1] < target_seq_len:
                # 填充到目标长度
                padding = torch.zeros(
                    continuous_pred.shape[0], 
                    target_seq_len - continuous_pred.shape[1], 
                    device=device,
                    requires_grad=True
                )
                continuous_pred = torch.cat([continuous_pred, padding], dim=1)
            
            # 初始化混合输入
            mixed_inputs[attr] = targets[attr].clone().float()  # 确保数据类型一致
            
            # 只在有效位置进行teacher forcing vs 预测的选择
            for b in range(batch_size):
                seq_len_b = sequence_lengths[b].item()
                if seq_len_b > 0:
                    # 在有效位置随机选择使用GT还是预测
                    use_gt = torch.rand(seq_len_b, device=device) < teacher_forcing_ratio
                    
                    # 创建新的序列而不是原地修改
                    new_sequence = mixed_inputs[attr][b].clone()
                    for pos in range(seq_len_b):
                        if use_gt[pos]:
                            new_sequence[pos] = targets[attr][b, pos]
                        else:
                            new_sequence[pos] = continuous_pred[b, pos]
                    # 使用cat操作来避免原地修改
                    mixed_inputs[attr] = torch.cat([
                        mixed_inputs[attr][:b],
                        new_sequence.unsqueeze(0),
                        mixed_inputs[attr][b+1:]
                    ], dim=0)
        
        # ===== 使用混合序列进行前向传播（保持梯度） =====
        return model.forward_with_predictions(
            x=mixed_inputs['x'],
            y=mixed_inputs['y'],
            z=mixed_inputs['z'],
            w=mixed_inputs['w'],
            h=mixed_inputs['h'],
            l=mixed_inputs['l'],
            image=rgbxyz
        )
    

    def _forward_with_pure_generation(self, rgbxyz: torch.Tensor, targets: Dict, model) -> Dict:
        """Pure Generation训练 - 支持梯度传播的增量生成"""
        batch_size = rgbxyz.size(0)
        seq_len = targets['x'].size(1)
        device = rgbxyz.device
        
        # ===== 使用支持梯度的增量生成 =====
        return self._forward_with_gradient_preserving_generation(
            model, rgbxyz, targets, seq_len, device
        )
    

    def _forward_with_gradient_preserving_generation(self, model, rgbxyz: torch.Tensor, targets: Dict, seq_len: int, device: torch.device) -> Dict:
        """
        支持梯度的增量生成 - 使用真正的增量解码
        
        这个版本使用类似 generate_next_box_incremental 的逻辑，但保持梯度流动
        """
        batch_size = rgbxyz.size(0)
        
        # 1. 编码图像（只计算一次）
        image_embed = model.image_encoder(rgbxyz)
        
        # 🔧 修复Bug：添加2D位置编码（与推理代码保持一致）
        H = W = int(np.sqrt(image_embed.shape[1]))
        if H * W == image_embed.shape[1]:
            from primitive_anything_3d import build_2d_sine_positional_encoding
            pos_embed_2d = build_2d_sine_positional_encoding(H, W, image_embed.shape[-1])
            pos_embed_2d = pos_embed_2d.flatten(0, 1).unsqueeze(0).to(image_embed.device)
            image_embed = image_embed + pos_embed_2d
        
        image_cond = None
        if model.condition_on_image and model.image_film_cond is not None:
            pooled_image_embed = image_embed.mean(dim=1)
            image_cond = model.image_cond_proj_film(pooled_image_embed)
        
        # 2. 初始化序列状态
        from einops import repeat
        current_sequence = repeat(model.sos_token, 'n d -> b n d', b=batch_size)
        
        # 3. 初始化缓存（用于真正的增量解码）
        decoder_cache = None
        gateloop_cache = []
        
        # 存储每一步的输出
        all_logits = {f'{attr}_logits': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        all_deltas = {f'{attr}_delta': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        all_continuous = {f'{attr}_continuous': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        all_eos_logits = []
        
        # 4. 逐步生成，使用真正的增量解码
        for step in range(seq_len):
            current_len = current_sequence.shape[1]
            if step == 0:
                # 第一步：完整前向传播，初始化缓存
                primitive_codes = current_sequence
                
                # 添加位置编码
                pos_embed = model.pos_embed[:, :current_len, :]
                primitive_codes = primitive_codes + pos_embed
                
                # 图像条件化
                if image_cond is not None:
                    primitive_codes = model.image_film_cond(primitive_codes, image_cond)
                
                # 门控循环块（初始化缓存）
                if model.gateloop_block is not None:
                    primitive_codes, gateloop_cache = model.gateloop_block(primitive_codes, cache=None)
                
                # Transformer解码（初始化decoder缓存）
                attended_codes, decoder_cache = model.decoder(
                    primitive_codes,
                    context=image_embed,
                    cache=None,
                    return_hiddens=True
                )
            else:
                # 后续步骤：只处理新token（真正的增量！）
                new_token = current_sequence[:, -1:, :]  # 只有最新的token
                
                # 添加位置编码（只对新token）
                pos_embed = model.pos_embed[:, current_len-1:current_len, :]
                primitive_codes = new_token + pos_embed
                
                # 图像条件化（只对新token）
                if image_cond is not None:
                    primitive_codes = model.image_film_cond(primitive_codes, image_cond)
                
                # 门控循环块增量计算
                if model.gateloop_block is not None:
                    primitive_codes, gateloop_cache = model.gateloop_block(
                        primitive_codes, 
                        cache=gateloop_cache
                    )
                
                # 真正的增量Transformer解码！（保持梯度）
                attended_codes, decoder_cache = model.decoder(
                    primitive_codes,
                    context=image_embed,
                    cache=decoder_cache,
                    return_hiddens=True
                )
            
            # 预测下一个token（只需要最后一个位置）
            step_embed = attended_codes[:, -1, :]
            
            # 预测各个属性（保持梯度，使用Gumbel Softmax）
            gumbel_temp = self.incremental_temperature
            x_logits, x_delta, x_continuous, x_embed = model.predict_attribute_with_continuous_embed(step_embed, 'x', prev_embeds=None, use_gumbel=True, temperature=gumbel_temp)
            y_logits, y_delta, y_continuous, y_embed = model.predict_attribute_with_continuous_embed(step_embed, 'y', prev_embeds=[x_embed], use_gumbel=True, temperature=gumbel_temp)
            z_logits, z_delta, z_continuous, z_embed = model.predict_attribute_with_continuous_embed(step_embed, 'z', prev_embeds=[x_embed, y_embed], use_gumbel=True, temperature=gumbel_temp)
            w_logits, w_delta, w_continuous, w_embed = model.predict_attribute_with_continuous_embed(step_embed, 'w', prev_embeds=[x_embed, y_embed, z_embed], use_gumbel=True, temperature=gumbel_temp)
            h_logits, h_delta, h_continuous, h_embed = model.predict_attribute_with_continuous_embed(step_embed, 'h', prev_embeds=[x_embed, y_embed, z_embed, w_embed], use_gumbel=True, temperature=gumbel_temp)
            l_logits, l_delta, l_continuous, l_embed = model.predict_attribute_with_continuous_embed(step_embed, 'l', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed], use_gumbel=True, temperature=gumbel_temp)
            
            # EOS预测
            eos_logits = model.to_eos_logits(step_embed).squeeze(-1)
            
            # 保存这一步的输出
            all_logits['x_logits'].append(x_logits)
            all_logits['y_logits'].append(y_logits)
            all_logits['z_logits'].append(z_logits)
            all_logits['w_logits'].append(w_logits)
            all_logits['h_logits'].append(h_logits)
            all_logits['l_logits'].append(l_logits)
            
            all_deltas['x_delta'].append(x_delta)
            all_deltas['y_delta'].append(y_delta)
            all_deltas['z_delta'].append(z_delta)
            all_deltas['w_delta'].append(w_delta)
            all_deltas['h_delta'].append(h_delta)
            all_deltas['l_delta'].append(l_delta)
            
            all_continuous['x_continuous'].append(x_continuous)
            all_continuous['y_continuous'].append(y_continuous)
            all_continuous['z_continuous'].append(z_continuous)
            all_continuous['w_continuous'].append(w_continuous)
            all_continuous['h_continuous'].append(h_continuous)
            all_continuous['l_continuous'].append(l_continuous)
            
            all_eos_logits.append(eos_logits)
            
            # 构建下一步的输入：当前token + 预测的连续值
            # 这里需要创建新的embedding来加入到序列中
            next_embeds = []
            for attr, continuous_val in [('x', x_continuous), ('y', y_continuous), ('z', z_continuous), 
                                       ('w', w_continuous), ('h', h_continuous), ('l', l_continuous)]:
                # 获取对应的离散化参数
                num_discrete = getattr(model, f'num_discrete_{attr}')
                continuous_range = getattr(model, f'continuous_range_{attr}')
                
                # 离散化连续值
                attr_discrete = model.discretize(continuous_val, num_discrete, continuous_range)
                attr_embed = getattr(model, f'{attr}_embed')(attr_discrete)
                next_embeds.append(attr_embed)
            
            # 组合所有属性的embedding
            combined_embed = torch.cat(next_embeds, dim=-1)  # [B, total_embed_dim]
            projected_embed = model.project_in(combined_embed).unsqueeze(1)  # [B, 1, model_dim]
            
            # 更新当前序列（保持梯度）
            current_sequence = torch.cat([current_sequence, projected_embed], dim=1)
        
        # 5. 组合所有输出
        result = {
            'logits_dict': {},
            'delta_dict': {},
            'continuous_dict': {},
            'eos_logits': torch.stack(all_eos_logits, dim=1)  # [B, seq_len]
        }
        
        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
            result['logits_dict'][f'{attr}_logits'] = torch.stack(all_logits[f'{attr}_logits'], dim=1)
            result['delta_dict'][f'{attr}_delta'] = torch.stack(all_deltas[f'{attr}_delta'], dim=1)  
            result['continuous_dict'][f'{attr}_continuous'] = torch.stack(all_continuous[f'{attr}_continuous'], dim=1)
        
        return result
    
 
    
    def _compute_sequence_lengths(self, targets_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算每个样本的真实序列长度"""
        # 使用x坐标来计算序列长度（非padding的位置数量）
        x_targets = targets_dict['x']  # [B, max_boxes]
        batch_size = x_targets.shape[0]
        
        sequence_lengths = []
        for b in range(batch_size):
            # 计算非padding值的数量（假设padding值为-1）
            valid_mask = x_targets[b] != -1
            seq_len = valid_mask.sum().item()
            sequence_lengths.append(seq_len)
        
        return torch.tensor(sequence_lengths, device=x_targets.device)
    
    def _train_epoch(self, phase: TrainingPhase, epoch_in_phase: int, loss_fn) -> TrainingStats:
        """训练一个epoch"""
        self.model.train()
        
        if self.train_sampler:
            self.train_sampler.set_epoch(self.current_epoch)
        
        # 计算当前teacher forcing比例
        teacher_forcing_ratio = self._compute_teacher_forcing_ratio(phase, epoch_in_phase)
        
        total_loss = 0.0
        total_cls_loss = 0.0
        total_iou_loss = 0.0
        total_delta_loss = 0.0
        total_mean_iou = 0.0
        total_adaptive_cls_weight = 0.0
        total_adaptive_delta_weight = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # 前向传播
                outputs = self._forward_with_sampling_strategy(batch, teacher_forcing_ratio)
                
                # 准备目标数据
                targets = {
                    'x': batch['x'].to(self.device),
                    'y': batch['y'].to(self.device), 
                    'z': batch['z'].to(self.device),
                    'w': batch['w'].to(self.device),
                    'h': batch['h'].to(self.device),
                    'l': batch['l'].to(self.device),
                    'rotations': batch['rotations'].to(self.device),
                }
                
                # 计算损失
                sequence_lengths = self._compute_sequence_lengths(targets)
                loss_dict = loss_fn(
                    logits_dict=outputs['logits_dict'],
                    delta_dict=outputs['delta_dict'],
                    eos_logits=outputs['eos_logits'],
                    targets_dict=targets,
                    sequence_lengths=sequence_lengths
                )
                
                loss = loss_dict['total_loss']
            
            # 反向传播
            if self.use_amp:
                self.scaler.scale(loss).backward()
                # 梯度裁剪（混合精度）
                if self.use_grad_clipping:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # 梯度裁剪（普通精度）
                if self.use_grad_clipping:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            total_cls_loss += loss_dict['total_classification'].item()
            total_iou_loss += loss_dict['iou_loss'].item()
            total_delta_loss += loss_dict['total_delta'].item()
            total_mean_iou += loss_dict['mean_iou'].item()
            total_adaptive_cls_weight += loss_dict.get('adaptive_classification_weight', torch.tensor(0.0)).item()
            total_adaptive_delta_weight += loss_dict.get('adaptive_delta_weight', torch.tensor(0.0)).item()
            num_batches += 1
            
                    # 日志记录 - 只在主进程打印详细日志
        log_interval = self.config_loader.get_global_config()['logging']['log_interval']
        if log_interval > 0 and batch_idx % log_interval == 0 and self.is_main_process:
            eos_loss_val = loss_dict.get('eos_loss', torch.tensor(0.0)).item()
            print(f"Epoch {self.current_epoch} [{batch_idx}/{len(self.train_loader)}] "
                  f"Total: {loss.item():.4f} | "
                  f"Cls: {loss_dict['total_classification'].item():.4f} | "
                  f"IoU: {loss_dict['iou_loss'].item():.4f} | "
                  f"Delta: {loss_dict['total_delta'].item():.4f} | "
                  f"EOS: {eos_loss_val:.4f} | "
                  f"TF: {teacher_forcing_ratio:.3f}")
        
        epoch_time = time.time() - start_time
        
        # 学习率调度
        if self.scheduler:
            self.scheduler.step()
        
        # 返回平均统计
        return TrainingStats(
            epoch=self.current_epoch,
            phase=phase.name,
            teacher_forcing_ratio=teacher_forcing_ratio,
            train_loss=total_loss / num_batches,
            train_classification_loss=total_cls_loss / num_batches,
            train_iou_loss=total_iou_loss / num_batches,
            train_delta_loss=total_delta_loss / num_batches,
            train_mean_iou=total_mean_iou / num_batches,
            val_loss=0.0,  # 稍后填充
            val_generation_loss=0.0,
            val_mean_iou=0.0,
            val_generation_iou=0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            adaptive_cls_weight=total_adaptive_cls_weight / num_batches,
            adaptive_delta_weight=total_adaptive_delta_weight / num_batches,
            epoch_time=epoch_time
        )
    
    def _validate_epoch(self, loss_fn, phase: TrainingPhase) -> Dict[str, float]:
        """验证一个epoch，返回详细的loss组件"""
        self.model.eval()
        
        # Teacher forcing验证统计
        total_tf_loss = 0.0
        total_tf_cls_loss = 0.0
        total_tf_iou_loss = 0.0
        total_tf_delta_loss = 0.0
        total_tf_eos_loss = 0.0
        total_tf_iou = 0.0
        
        # 生成验证统计
        total_gen_loss = 0.0
        total_gen_iou = 0.0
        total_gen_cls_loss = 0.0
        total_gen_iou_loss = 0.0
        total_gen_delta_loss = 0.0
        total_gen_eos_loss = 0.0
        total_generated_boxes = 0.0
        total_gt_boxes = 0.0
        num_batches = 0

        total_x_error = 0.0
        total_y_error = 0.0
        total_z_error = 0.0
        total_w_error = 0.0
        total_h_error = 0.0
        total_l_error = 0.0
        total_overall_error = 0.0
        
        # 保存验证样本的推理结果
        validation_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # 1. Teacher forcing验证 (与训练一致)
                tf_outputs = self._forward_with_sampling_strategy(batch, 1.0)  # 完全teacher forcing
                
                targets = {
                    'x': batch['x'].to(self.device),
                    'y': batch['y'].to(self.device),
                    'z': batch['z'].to(self.device),
                    'w': batch['w'].to(self.device),
                    'h': batch['h'].to(self.device),
                    'l': batch['l'].to(self.device),
                    'rotations': batch['rotations'].to(self.device),
                }
                
                sequence_lengths = self._compute_sequence_lengths(targets)
                tf_loss_dict = loss_fn(
                    logits_dict=tf_outputs['logits_dict'],
                    delta_dict=tf_outputs['delta_dict'],
                    eos_logits=tf_outputs['eos_logits'],
                    targets_dict=targets,
                    sequence_lengths=sequence_lengths
                )
                
                # 累积teacher forcing loss组件
                total_tf_loss += tf_loss_dict['total_loss'].item()
                total_tf_cls_loss += tf_loss_dict['total_classification'].item()
                total_tf_iou_loss += tf_loss_dict['iou_loss'].item()
                total_tf_delta_loss += tf_loss_dict['total_delta'].item()
                total_tf_eos_loss += tf_loss_dict.get('eos_loss', torch.tensor(0.0)).item()
                
                # 计算真正的评估IoU (使用与生成验证相同的方法)
                eval_iou = self._compute_tf_evaluation_iou(tf_outputs, targets)
                
                # 2. 纯生成验证
                rgbxyz = batch['image'].to(self.device)
                
                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                
                # 根据配置选择推理方法
                if self.use_incremental_inference:
                    # 使用增量推理进行验证生成（更高效）
                    # 🔧 修复：使用配置中的正确max_seq_len，而不是不存在的max_primitive_len
                    max_len = self.config_loader.get_global_config()['max_seq_len']
                    gen_results = model.generate_incremental(
                        image=rgbxyz,
                        max_seq_len=max_len,
                        temperature=self.incremental_temperature
                    )
                else:
                    # 使用传统推理方法
                    # 🔧 修复：使用配置中的正确max_seq_len
                    max_len = self.config_loader.get_global_config()['max_seq_len']
                    gen_results = model.generate(
                        image=rgbxyz,
                        max_seq_len=max_len,
                        temperature=1.0
                    )
                # print(f"gen_results: {gen_results}")
                
                # 计算生成结果的详细损失和统计信息
                gen_metrics = self._compute_generation_metrics(gen_results, targets, loss_fn, verbose=False)
                
                # 删除这行重复的累积
                # total_tf_loss += tf_loss_dict['total_loss'].item()
                
                total_tf_iou += eval_iou  # 使用评估IoU而不是损失IoU
                total_gen_iou += gen_metrics['iou']
                total_generated_boxes += gen_metrics['num_generated_boxes']
                total_gt_boxes += gen_metrics['num_gt_boxes']
                
                # 累积维度误差
                total_x_error += gen_metrics['x_error']
                total_y_error += gen_metrics['y_error']
                total_z_error += gen_metrics['z_error']
                total_w_error += gen_metrics['w_error']
                total_h_error += gen_metrics['h_error']
                total_l_error += gen_metrics['l_error']
                total_overall_error += gen_metrics['overall_mean_error']
                
                num_batches += 1
                
                # 保存指定样本的验证结果
                if batch_idx in self.validation_samples and self.is_main_process:
                    validation_results.append({
                        'batch_idx': batch_idx,
                        'epoch': self.current_epoch,
                        'phase': phase.name,
                        'generation_results': gen_results,
                        'ground_truth': targets,
                        'tf_loss': tf_loss_dict['total_loss'].item(),
                        'gen_iou': gen_metrics['iou'],
                        'image_shape': rgbxyz.shape
                    })
        
        # 保存验证结果 - 只在主进程保存
        if validation_results and self.is_main_process:
            results_file = self.validation_dir / f"validation_epoch_{self.current_epoch:04d}.json"
            with open(results_file, 'w') as f:
                # 将tensor转换为可序列化的格式
                serializable_results = []
                for result in validation_results:
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            serializable_result[key] = value.detach().cpu().tolist()
                        elif isinstance(value, dict):
                            serializable_result[key] = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v 
                                                      for k, v in value.items()}
                        else:
                            serializable_result[key] = value
                    serializable_results.append(serializable_result)
                
                json.dump(serializable_results, f, indent=2)
        
        # 计算平均值并返回详细loss组件
        avg_tf_loss = total_tf_loss / num_batches if num_batches > 0 else 0.0
        avg_tf_cls_loss = total_tf_cls_loss / num_batches if num_batches > 0 else 0.0
        avg_tf_iou_loss = total_tf_iou_loss / num_batches if num_batches > 0 else 0.0
        avg_tf_delta_loss = total_tf_delta_loss / num_batches if num_batches > 0 else 0.0
        avg_tf_eos_loss = total_tf_eos_loss / num_batches if num_batches > 0 else 0.0
        avg_tf_iou = total_tf_iou / num_batches if num_batches > 0 else 0.0
        
        # 移除虚假的生成损失计算
        # avg_gen_loss = total_gen_loss / num_batches if num_batches > 0 else 0.0
        # avg_gen_cls_loss = total_gen_cls_loss / num_batches if num_batches > 0 else 0.0
        # avg_gen_iou_loss = total_gen_iou_loss / num_batches if num_batches > 0 else 0.0
        # avg_gen_delta_loss = total_gen_delta_loss / num_batches if num_batches > 0 else 0.0
        # avg_gen_eos_loss = total_gen_eos_loss / num_batches if num_batches > 0 else 0.0
        
        avg_gen_iou = total_gen_iou / num_batches if num_batches > 0 else 0.0
        avg_generated_boxes = total_generated_boxes / num_batches if num_batches > 0 else 0.0
        avg_gt_boxes = total_gt_boxes / num_batches if num_batches > 0 else 0.0
        
        # 计算平均误差
        avg_x_error = total_x_error / num_batches if num_batches > 0 else 0.0
        avg_y_error = total_y_error / num_batches if num_batches > 0 else 0.0
        avg_z_error = total_z_error / num_batches if num_batches > 0 else 0.0
        avg_w_error = total_w_error / num_batches if num_batches > 0 else 0.0
        avg_h_error = total_h_error / num_batches if num_batches > 0 else 0.0
        avg_l_error = total_l_error / num_batches if num_batches > 0 else 0.0
        avg_overall_error = total_overall_error / num_batches if num_batches > 0 else 0.0
        
        # 返回结果 - 移除虚假的损失值
        return {
            'tf_total_loss': avg_tf_loss,
            'tf_classification_loss': avg_tf_cls_loss,
            'tf_iou_loss': avg_tf_iou_loss,
            'tf_delta_loss': avg_tf_delta_loss,
            'tf_eos_loss': avg_tf_eos_loss,
            'tf_mean_iou': avg_tf_iou,
            'generation_iou': avg_gen_iou,
            'avg_generated_boxes': avg_generated_boxes,
            'avg_gt_boxes': avg_gt_boxes,
            'avg_x_error': avg_x_error,
            'avg_y_error': avg_y_error,
            'avg_z_error': avg_z_error,
            'avg_w_error': avg_w_error,
            'avg_h_error': avg_h_error,
            'avg_l_error': avg_l_error,
            'avg_overall_error': avg_overall_error
        }
    
    def _compute_tf_evaluation_iou(self, outputs: Dict, targets: Dict, verbose: bool = False) -> float:
        """计算TF评估IoU，使用一对一匹配策略"""
        try:
            batch_size = targets['x'].size(0)
            total_iou = 0.0
            valid_samples = 0
            
            for b in range(batch_size):
                # 构建预测的3D boxes
                pred_boxes = []
                gt_boxes = []
                gt_rotations = []
                
                # 获取每个位置的预测值
                seq_len = targets['x'].size(1)
                for s in range(seq_len):
                    # 检查是否为有效位置（非padding）
                    if targets['x'][b, s].item() != -1.0:
                        # 将logits和delta转换为连续预测值
                        pred_box = []
                        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
                            if attr + '_logits' in outputs['logits_dict'] and attr + '_delta' in outputs['delta_dict']:
                                logits = outputs['logits_dict'][attr + '_logits'][b, s]  # [num_bins]
                                delta = outputs['delta_dict'][attr + '_delta'][b, s]     # scalar
                                continuous_val = self._get_continuous_prediction(logits, delta, attr)
                                pred_box.append(continuous_val)
                        
                        if len(pred_box) == 6:
                            pred_boxes.append(pred_box)
                            
                            # 对应的GT box
                            gt_box = [
                                targets['x'][b, s].cpu().item(),
                                targets['y'][b, s].cpu().item(),
                                targets['z'][b, s].cpu().item(),
                                targets['w'][b, s].cpu().item(),
                                targets['h'][b, s].cpu().item(),
                                targets['l'][b, s].cpu().item(),
                            ]
                            gt_boxes.append(gt_box)
                            
                            # GT旋转（如果可用）
                            if 'rotations' in targets:
                                gt_rot = targets['rotations'][b, s].cpu().numpy()  # [4] quaternion
                                gt_rotations.append(gt_rot)
                            else:
                                # 使用单位四元数（无旋转）
                                gt_rotations.append([0.0, 0.0, 0.0, 1.0])
                
                # 计算该样本的IoU
                if pred_boxes and gt_boxes:
                    sample_ious = []
                    
                    # 计算每个预测box与对应GT box的IoU
                    for i, (pred_box, gt_box, gt_rot) in enumerate(zip(pred_boxes, gt_boxes, gt_rotations)):
                        try:
                            # 使用AABB IoU计算
                            iou = self._compute_box_iou(pred_box, gt_box, gt_rot)
                            sample_ious.append(iou)
                            
                        except Exception as e:
                            print(f"⚠️  计算box IoU时出错: {e}")
                            sample_ious.append(0.0)
                    
                    if sample_ious:
                        sample_mean_iou = sum(sample_ious) / len(sample_ious)
                        total_iou += sample_mean_iou
                        valid_samples += 1
                        if verbose:  # 只在verbose=True时打印
                            print(f"📊 TF Sample {b}: Mean IoU = {sample_mean_iou:.4f} ({len(sample_ious)} boxes)")
            
            if valid_samples == 0:
                return 0.0
            
            mean_iou = total_iou / valid_samples
            if verbose:  # 只在verbose=True时打印
                print(f"\n🎯 TF Overall Mean IoU: {mean_iou:.4f} (from {valid_samples} samples)")
            return float(mean_iou)
            
        except Exception as e:
            if verbose:
                print(f"⚠️  计算TF评估IoU时出错: {e}")
            return 0.0
    
    def _get_continuous_prediction(self, logits: torch.Tensor, delta: torch.Tensor, attr: str) -> float:
        """将分类logits和delta组合成连续预测值"""
        # 获取属性的配置（从ConfigLoader返回的平铺结构中获取）
        attr_configs = {
            'x': (self.model_config.get('num_discrete_x', 128), 
                  self.model_config.get('continuous_range_x', [0.5, 2.5])),
            'y': (self.model_config.get('num_discrete_y', 128), 
                  self.model_config.get('continuous_range_y', [-2.0, 2.0])),
            'z': (self.model_config.get('num_discrete_z', 128), 
                  self.model_config.get('continuous_range_z', [-1.5, 1.5])),
            'w': (self.model_config.get('num_discrete_w', 64), 
                  self.model_config.get('continuous_range_w', [0.1, 1.0])),
            'h': (self.model_config.get('num_discrete_h', 64), 
                  self.model_config.get('continuous_range_h', [0.1, 1.0])),
            'l': (self.model_config.get('num_discrete_l', 64), 
                  self.model_config.get('continuous_range_l', [0.1, 1.0]))
        }
        
        num_bins, value_range = attr_configs[attr]
        min_val, max_val = value_range
        
        # 🔧 修复：使用正确的bin内offset逻辑
        # 获取分类预测（确定性采样）
        discrete_pred = torch.argmax(logits).item()
        
        # 计算bin_width和连续值
        bin_width = (max_val - min_val) / (num_bins - 1)
        continuous_base = min_val + discrete_pred * bin_width
        
        # 确保delta在CPU上，并正确缩放为bin内offset
        if delta.is_cuda:
            delta_val = float(delta.cpu().detach()) * bin_width
        else:
            delta_val = float(delta.detach()) * bin_width
        
        return continuous_base + delta_val

    def _compute_generation_metrics(self, gen_results: Dict, targets: Dict, loss_fn, verbose: bool = False) -> Dict[str, float]:
        """
        计算生成结果的详细指标
        Args:
            gen_results: 生成结果字典（已经是连续值）
            targets: 目标值字典
            loss_fn: 损失函数（不使用，因为gen_results不是模型输出格式）
            verbose: 是否打印详细信息
        Returns:
            metrics: 包含各项统计信息的字典
        """
        try:
            # 生成结果已经是连续值，直接使用
            processed_gen_results = {}
            target_seq_len = targets['x'].shape[1]  # GT的序列长度
            
            for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
                if attr in gen_results:
                    # 获取生成结果（已经是连续值）
                    gen_values = gen_results[attr]  # [B, seq_len]
                    
                    # 对齐序列长度
                    if gen_values.shape[1] > target_seq_len:
                        gen_values = gen_values[:, :target_seq_len]
                    elif gen_values.shape[1] < target_seq_len:
                        # 用零填充
                        padding_length = target_seq_len - gen_values.shape[1]
                        padding = torch.zeros(gen_values.shape[0], padding_length, device=gen_values.device)
                        gen_values = torch.cat([gen_values, padding], dim=1)
                    
                    processed_gen_results[attr] = gen_values
                else:
                    # 如果某个属性缺失，使用零填充
                    processed_gen_results[attr] = torch.zeros(targets[attr].shape, device=targets[attr].device)
            
            # 计算IoU
            gen_iou = self._compute_generation_iou(processed_gen_results, targets, verbose)
            
            # 计算6个维度的平均误差
            dimension_errors = {}
            total_valid_predictions = 0
            
            for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
                if attr in processed_gen_results and attr in targets:
                    # 获取生成结果和目标值
                    gen_values = processed_gen_results[attr]  # [B, seq_len]
                    gt_values = targets[attr]                  # [B, seq_len]
                    
                    # 创建有效mask（排除padding值）
                    valid_mask = (gt_values != -1.0) & (gt_values != 0.0)  # GT非padding且非零
                    
                    if valid_mask.sum() > 0:
                        # 计算有效位置的绝对误差
                        abs_errors = torch.abs(gen_values - gt_values)
                        valid_errors = abs_errors[valid_mask]
                        
                        # 计算平均误差
                        mean_error = valid_errors.mean().item()
                        dimension_errors[f'{attr}_error'] = mean_error
                        total_valid_predictions += valid_mask.sum().item()
                    else:
                        dimension_errors[f'{attr}_error'] = 0.0
                else:
                    dimension_errors[f'{attr}_error'] = 0.0
            
            # 计算总体平均误差
            if total_valid_predictions > 0:
                overall_mean_error = sum(dimension_errors.values()) / 6.0
            else:
                overall_mean_error = 0.0
            
            # 统计生成的箱子数量 vs GT箱子数量
            num_generated_boxes = 0
            num_gt_boxes = 0
            
            batch_size = targets['x'].shape[0]
            for b in range(batch_size):
                # 统计GT箱子数量 (非padding的位置)
                gt_valid = (targets['x'][b] != -1.0).sum().item()
                num_gt_boxes += gt_valid
                
                # 统计生成的箱子数量 (简化：检查是否生成了有效坐标)
                if 'x' in processed_gen_results:
                    gen_valid = (processed_gen_results['x'][b] != 0.0).sum().item()  # 假设0为填充值
                    num_generated_boxes += gen_valid
            
            # 合并所有指标 - 移除虚假的损失值
            metrics = {
                'iou': gen_iou,
                'num_generated_boxes': num_generated_boxes,
                'num_gt_boxes': num_gt_boxes,
                'overall_mean_error': overall_mean_error,
                'x_error': dimension_errors['x_error'],
                'y_error': dimension_errors['y_error'],
                'z_error': dimension_errors['z_error'],
                'w_error': dimension_errors['w_error'],
                'h_error': dimension_errors['h_error'],
                'l_error': dimension_errors['l_error']
            }
            
            return metrics
            
        except Exception as e:
            print(f"⚠️  计算生成指标时出错: {e}")
            # 返回默认值
            return {
                'iou': 0.0,
                'num_generated_boxes': 0,
                'num_gt_boxes': 0,
                'overall_mean_error': 1.0,
                'x_error': 1.0,
                'y_error': 1.0,
                'z_error': 1.0,
                'w_error': 1.0,
                'h_error': 1.0,
                'l_error': 1.0
            }
    
    def _compute_generation_iou(self, gen_results: Dict, targets: Dict, verbose: bool = False) -> float:
        """计算生成结果与目标的IoU，使用一对一匹配策略"""
        try:
            # 检查生成结果
            if not gen_results or gen_results['x'].shape[1] == 0:
                return 0.0
            
            batch_size = targets['x'].size(0)
            total_iou = 0.0
            valid_samples = 0
            
            for b in range(batch_size):
                # 构建预测boxes和GT boxes的一对一匹配
                pred_boxes = []
                gt_boxes = []
                gt_rotations = []
                
                # 获取序列长度
                seq_len = targets['x'].size(1)
                gen_len = gen_results['x'].shape[1]
                
                # 一对一匹配：只计算同时存在的位置
                for s in range(seq_len):
                    # 检查GT是否为有效位置（非padding）且生成结果中也存在该位置
                    if targets['x'][b, s].item() != -1.0 and s < gen_len:
                        # GT box
                        gt_box = [
                            targets['x'][b, s].cpu().item(),
                            targets['y'][b, s].cpu().item(),
                            targets['z'][b, s].cpu().item(),
                            targets['w'][b, s].cpu().item(),
                            targets['h'][b, s].cpu().item(),
                            targets['l'][b, s].cpu().item(),
                        ]
                        gt_boxes.append(gt_box)
                        
                        # GT旋转信息
                        if 'rotations' in targets:
                            gt_rot = [
                                targets['rotations'][b, s, 0].cpu().item(),
                                targets['rotations'][b, s, 1].cpu().item(),
                                targets['rotations'][b, s, 2].cpu().item(),
                                targets['rotations'][b, s, 3].cpu().item(),
                            ]
                            gt_rotations.append(gt_rot)
                        else:
                            gt_rotations.append([0.0, 0.0, 0.0, 1.0])  # 单位四元数
                        
                        # 对应的预测box - 修复访问格式
                        pred_box = [
                            float(gen_results['x'][b, s]),
                            float(gen_results['y'][b, s]),
                            float(gen_results['z'][b, s]),
                            float(gen_results['w'][b, s]),
                            float(gen_results['h'][b, s]),
                            float(gen_results['l'][b, s]),
                        ]
                        pred_boxes.append(pred_box)
                
                # 计算该样本的IoU（一对一匹配）
                if pred_boxes and gt_boxes:
                    sample_ious = []
                    
                    # 计算每个预测box与对应GT box的IoU
                    for i, (pred_box, gt_box, gt_rot) in enumerate(zip(pred_boxes, gt_boxes, gt_rotations)):
                        try:
                            # 使用AABB IoU计算
                            iou = self._compute_box_iou(pred_box, gt_box, gt_rot)
                            sample_ious.append(iou)
                            
                        except Exception as e:
                            print(f"⚠️  计算box IoU时出错: {e}")
                            sample_ious.append(0.0)
                    
                    if sample_ious:
                        sample_mean_iou = sum(sample_ious) / len(sample_ious)
                        total_iou += sample_mean_iou
                        valid_samples += 1
                        if verbose:  # 只在verbose=True时打印每个sample的IoU
                            print(f"📊 Generation Sample {b}: Mean IoU = {sample_mean_iou:.4f} ({len(sample_ious)} boxes)")
            
            if valid_samples == 0:
                return 0.0
            
            mean_iou = total_iou / valid_samples
            if verbose:  # 只在verbose=True时打印总体IoU
                print(f"\n🎯 Generation Overall Mean IoU: {mean_iou:.4f} (from {valid_samples} samples)")
            return float(mean_iou)
            
        except Exception as e:
            if verbose:
                print(f"计算生成IoU时出错: {e}")
            return 0.0
    
    def _compute_box_iou(self, box1: List[float], box2: List[float], rot1: List[float] = None, rot2: List[float] = None) -> float:
        """
        计算两个3D box的IoU，统一使用AABB计算
        Args:
            box1: [x, y, z, w, h, l] - 第一个box的中心坐标和尺寸
            box2: [x, y, z, w, h, l] - 第二个box的中心坐标和尺寸 
            rot1: [qx, qy, qz, qw] - 第一个box的四元数旋转（忽略）
            rot2: [qx, qy, qz, qw] - 第二个box的四元数旋转（忽略）
        Returns:
            IoU值 (0.0-1.0)
        """
        # 直接使用AABB IoU计算，忽略旋转信息
        return self._compute_simple_aabb_iou(box1, box2)
    
    def _compute_simple_aabb_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个轴对齐box的IoU"""
        try:
            # 确保box格式正确
            if len(box1) != 6 or len(box2) != 6:
                print(f"⚠️  Box格式错误: box1长度={len(box1)}, box2长度={len(box2)}")
                return 0.0
                
            x1, y1, z1, w1, h1, l1 = box1
            x2, y2, z2, w2, h2, l2 = box2
            
            # 计算边界
            x1_min, x1_max = x1 - w1/2, x1 + w1/2
            y1_min, y1_max = y1 - h1/2, y1 + h1/2
            z1_min, z1_max = z1 - l1/2, z1 + l1/2
            
            x2_min, x2_max = x2 - w2/2, x2 + w2/2
            y2_min, y2_max = y2 - h2/2, y2 + h2/2
            z2_min, z2_max = z2 - l2/2, z2 + l2/2
            
            # 计算交集（使用正确的逻辑）
            inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
            
            # 计算体积
            inter_volume = inter_x * inter_y * inter_z
            volume1 = w1 * h1 * l1
            volume2 = w2 * h2 * l2
            union_volume = volume1 + volume2 - inter_volume
            
            if union_volume <= 0:
                return 0.0
                
            iou = inter_volume / union_volume
            return max(0.0, min(1.0, iou))  # 确保在[0,1]范围内
            
        except Exception as e:
            print(f"轴对齐IoU计算出错: {e}")
            return 0.0
    
    def _quaternion_to_rotation_matrix(self, quat: List[float]) -> np.ndarray:
        """将四元数转换为旋转矩阵"""
        import numpy as np
        
        if len(quat) != 4:
            return np.eye(3)
            
        qx, qy, qz, qw = quat
        
        # 标准化四元数
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm == 0:
            return np.eye(3)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
        
        # 四元数到旋转矩阵的转换
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        return R
    
    def _compute_axis_aligned_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算axis-aligned box的IoU（原始实现）"""
        x1, y1, z1, w1, h1, l1 = box1
        x2, y2, z2, w2, h2, l2 = box2
        
        # 计算边界
        x1_min, x1_max = x1 - w1/2, x1 + w1/2
        y1_min, y1_max = y1 - h1/2, y1 + h1/2
        z1_min, z1_max = z1 - l1/2, z1 + l1/2
        
        x2_min, x2_max = x2 - w2/2, x2 + w2/2
        y2_min, y2_max = y2 - h2/2, y2 + h2/2
        z2_min, z2_max = z2 - l2/2, z2 + l2/2
        
        # 计算交集
        inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
        
        inter_volume = inter_x * inter_y * inter_z
        
        # 计算并集
        vol1 = w1 * h1 * l1
        vol2 = w2 * h2 * l2
        union_volume = vol1 + vol2 - inter_volume
        
        if union_volume <= 0:
            return 0.0
        
        return inter_volume / union_volume
    
    def _compute_oriented_box_iou_sampling(self, center1, size1, R1, center2, size2, R2, samples_per_dim=20):
        """
        使用采样方法计算旋转box的IoU
        通过在3D空间中密集采样点来近似计算交集体积
        """
        import numpy as np
        
        # 计算两个box的体积
        vol1 = size1[0] * size1[1] * size1[2]
        vol2 = size2[0] * size2[1] * size2[2]
        
        if vol1 <= 0 or vol2 <= 0:
            return 0.0
        
        # 确定采样区域（两个box的包围盒）
        corners1 = self._get_box_corners(center1, size1, R1)
        corners2 = self._get_box_corners(center2, size2, R2)
        
        all_corners = np.vstack([corners1, corners2])
        min_coords = np.min(all_corners, axis=0)
        max_coords = np.max(all_corners, axis=0)
        
        # 扩展采样区域一点点，确保包含边界
        margin = 0.01
        min_coords -= margin
        max_coords += margin
        
        # 生成采样点
        x = np.linspace(min_coords[0], max_coords[0], samples_per_dim)
        y = np.linspace(min_coords[1], max_coords[1], samples_per_dim)
        z = np.linspace(min_coords[2], max_coords[2], samples_per_dim)
        
        # 计算采样体积元素
        dx = (max_coords[0] - min_coords[0]) / samples_per_dim
        dy = (max_coords[1] - min_coords[1]) / samples_per_dim
        dz = (max_coords[2] - min_coords[2]) / samples_per_dim
        dV = dx * dy * dz
        
        # 检查每个采样点是否在两个box内
        intersection_count = 0
        union_count = 0
        
        for xi in x:
            for yi in y:
                for zi in z:
                    point = np.array([xi, yi, zi])
                    
                    in_box1 = self._point_in_oriented_box(point, center1, size1, R1)
                    in_box2 = self._point_in_oriented_box(point, center2, size2, R2)
                    
                    if in_box1 and in_box2:
                        intersection_count += 1
                    
                    if in_box1 or in_box2:
                        union_count += 1
        
        if union_count == 0:
            return 0.0
        
        # 计算IoU
        intersection_volume = intersection_count * dV
        union_volume = vol1 + vol2 - intersection_volume
        
        if union_volume <= 0:
            return 0.0
        
        iou = intersection_volume / union_volume
        
        # 调试信息：当IoU异常时打印详细信息
        if iou < 0 or iou > 1:
            print(f"⚠️  异常IoU值: {iou:.6f}")
            print(f"  intersection_count: {intersection_count}, dV: {dV:.6f}")
            print(f"  vol1: {vol1:.6f}, vol2: {vol2:.6f}")
            print(f"  intersection_volume: {intersection_volume:.6f}")
            print(f"  union_volume: {union_volume:.6f}")
            # 强制限制在[0,1]范围内
            iou = max(0.0, min(1.0, iou))
        
        return iou
    
    def _get_box_corners(self, center, size, R):
        """获取旋转box的8个顶点坐标"""
        import numpy as np
        
        # box在局部坐标系中的8个顶点
        w, h, l = size
        local_corners = np.array([
            [-w/2, -h/2, -l/2],
            [+w/2, -h/2, -l/2],
            [+w/2, +h/2, -l/2],
            [-w/2, +h/2, -l/2],
            [-w/2, -h/2, +l/2],
            [+w/2, -h/2, +l/2],
            [+w/2, +h/2, +l/2],
            [-w/2, +h/2, +l/2],
        ])
        
        # 转换到世界坐标系
        world_corners = (R @ local_corners.T).T + center
        return world_corners
    
    def _point_in_oriented_box(self, point, center, size, R):
        """检查点是否在旋转的box内"""
        import numpy as np
        
        # 将点转换到box的局部坐标系
        local_point = R.T @ (point - center)
        
        # 检查是否在局部坐标系的边界内
        w, h, l = size
        return (abs(local_point[0]) <= w/2 and 
                abs(local_point[1]) <= h/2 and 
                abs(local_point[2]) <= l/2)
    
    def _test_inference(self) -> Dict[str, float]:
        """在测试集上进行推理，计算指标并保存结果"""
        if self.is_main_process:
            print("🧪 开始测试集推理...")
        
        # 创建测试数据加载器
        test_loader = create_dataloader(
            data_root=self.config_loader.get('data.dataset.data_root'),
            stage="test",
            batch_size=1,  # 测试时使用batch_size=1
            max_boxes=self.config_loader.get('global.max_seq_len'),
            image_size=self.config_loader.get('global.image_size'),
            continuous_ranges=self.config_loader.get('data.dataset.continuous_ranges'),
            augmentation_config={},  # 测试时不使用数据增强
            num_workers=0,  # 测试时使用单进程
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=False
        )
        
        self.model.eval()
        
        # 测试统计
        total_gen_iou = 0.0
        total_generated_boxes = 0.0
        total_gt_boxes = 0.0
        total_x_error = 0.0
        total_y_error = 0.0
        total_z_error = 0.0
        total_w_error = 0.0
        total_h_error = 0.0
        total_l_error = 0.0
        total_overall_error = 0.0
        num_batches = 0
        
        # 保存测试结果
        test_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if self.is_main_process:
                    print(f"  📊 处理测试样本 {batch_idx + 1}/{len(test_loader)}")
                
                # 准备输入数据
                rgbxyz = batch['image'].to(self.device)
                targets = {
                    'x': batch['x'].to(self.device),
                    'y': batch['y'].to(self.device),
                    'z': batch['z'].to(self.device),
                    'w': batch['w'].to(self.device),
                    'h': batch['h'].to(self.device),
                    'l': batch['l'].to(self.device),
                    'rotations': batch['rotations'].to(self.device),
                }
                
                # 使用训练好的模型进行生成
                if hasattr(self.model, 'module'):
                    model = self.model.module
                else:
                    model = self.model
                
                try:
                    # 检查输入数据的有效性
                    if torch.isnan(rgbxyz).any() or torch.isinf(rgbxyz).any():
                        if self.is_main_process:
                            print(f"  ⚠️  输入数据包含NaN或Inf值，跳过此样本")
                        continue
                    
                    # 根据配置选择推理方法
                    if self.use_incremental_inference:
                        # 使用增量推理进行测试生成（更高效）
                        # 🔧 修复：使用正确的max_seq_len参数
                        max_len = self.config_loader.get_global_config()['max_seq_len']
                        gen_results = model.generate_incremental(
                            image=rgbxyz,
                            max_seq_len=max_len,
                            temperature=self.incremental_temperature
                        )
                    else:
                        # 使用传统推理方法
                        # 🔧 修复：使用正确的max_seq_len参数
                        max_len = self.config_loader.get_global_config()['max_seq_len']
                        gen_results = model.generate(
                            image=rgbxyz,
                            max_seq_len=max_len,
                            temperature=1.0
                        )
                    
                    # 检查生成结果的有效性
                    if not gen_results or not isinstance(gen_results, dict):
                        if self.is_main_process:
                            print(f"  ⚠️  生成结果无效，跳过此样本")
                        continue
                        
                except Exception as e:
                    if self.is_main_process:
                        print(f"  ❌ 生成过程中出错: {e}")
                        print(f"  📊 输入图像形状: {rgbxyz.shape}")
                        print(f"  📊 输入图像范围: [{rgbxyz.min().item():.4f}, {rgbxyz.max().item():.4f}]")
                    continue
                
                # 计算生成指标
                gen_metrics = self._compute_generation_metrics(gen_results, targets, None, verbose=False)
                
                # 在test推理时打印每个sample的IoU
                if self.is_main_process:
                    print(f"   Test Sample {batch_idx + 1}: Mean IoU = {gen_metrics['iou']:.4f} ({gen_metrics['num_generated_boxes']:.0f} boxes)")
                
                # 累积统计
                total_gen_iou += gen_metrics['iou']
                total_generated_boxes += gen_metrics['num_generated_boxes']
                total_gt_boxes += gen_metrics['num_gt_boxes']
                total_x_error += gen_metrics['x_error']
                total_y_error += gen_metrics['y_error']
                total_z_error += gen_metrics['z_error']
                total_w_error += gen_metrics['w_error']
                total_h_error += gen_metrics['h_error']
                total_l_error += gen_metrics['l_error']
                total_overall_error += gen_metrics['overall_mean_error']
                
                # 保存详细结果
                test_results.append({
                    'batch_idx': batch_idx,
                    'folder_name': batch['folder_name'][0] if 'folder_name' in batch else f'test_{batch_idx}',
                    'generation_results': {
                        'x': gen_results['x'].cpu().tolist() if 'x' in gen_results else [],
                        'y': gen_results['y'].cpu().tolist() if 'y' in gen_results else [],
                        'z': gen_results['z'].cpu().tolist() if 'z' in gen_results else [],
                        'w': gen_results['w'].cpu().tolist() if 'w' in gen_results else [],
                        'h': gen_results['h'].cpu().tolist() if 'h' in gen_results else [],
                        'l': gen_results['l'].cpu().tolist() if 'l' in gen_results else [],
                    },
                    'ground_truth': {
                        'x': targets['x'].cpu().tolist(),
                        'y': targets['y'].cpu().tolist(),
                        'z': targets['z'].cpu().tolist(),
                        'w': targets['w'].cpu().tolist(),
                        'h': targets['h'].cpu().tolist(),
                        'l': targets['l'].cpu().tolist(),
                        'rotations': targets['rotations'].cpu().tolist(),
                    },
                    'metrics': {
                        'iou': gen_metrics['iou'],
                        'num_generated_boxes': gen_metrics['num_generated_boxes'],
                        'num_gt_boxes': gen_metrics['num_gt_boxes'],
                        'x_error': gen_metrics['x_error'],
                        'y_error': gen_metrics['y_error'],
                        'z_error': gen_metrics['z_error'],
                        'w_error': gen_metrics['w_error'],
                        'h_error': gen_metrics['h_error'],
                        'l_error': gen_metrics['l_error'],
                        'overall_mean_error': gen_metrics['overall_mean_error']
                    }
                })
                
                num_batches += 1
        
        # 计算平均值
        avg_gen_iou = total_gen_iou / num_batches if num_batches > 0 else 0.0
        avg_generated_boxes = total_generated_boxes / num_batches if num_batches > 0 else 0.0
        avg_gt_boxes = total_gt_boxes / num_batches if num_batches > 0 else 0.0
        avg_x_error = total_x_error / num_batches if num_batches > 0 else 0.0
        avg_y_error = total_y_error / num_batches if num_batches > 0 else 0.0
        avg_z_error = total_z_error / num_batches if num_batches > 0 else 0.0
        avg_w_error = total_w_error / num_batches if num_batches > 0 else 0.0
        avg_h_error = total_h_error / num_batches if num_batches > 0 else 0.0
        avg_l_error = total_l_error / num_batches if num_batches > 0 else 0.0
        avg_overall_error = total_overall_error / num_batches if num_batches > 0 else 0.0
        
        # 保存测试结果到文件 - 只在主进程保存
        if self.is_main_process:
            test_results_file = self.output_dir / "test_results.json"
            with open(test_results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'test_summary': {
                        'num_samples': num_batches,
                        'avg_generation_iou': avg_gen_iou,
                        'avg_generated_boxes': avg_generated_boxes,
                        'avg_gt_boxes': avg_gt_boxes,
                        'avg_overall_error': avg_overall_error,
                        'avg_x_error': avg_x_error,
                        'avg_y_error': avg_y_error,
                        'avg_z_error': avg_z_error,
                        'avg_w_error': avg_w_error,
                        'avg_h_error': avg_h_error,
                        'avg_l_error': avg_l_error,
                        'generation_rate': avg_generated_boxes / max(avg_gt_boxes, 1)
                    },
                    'detailed_results': test_results
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 测试结果已保存到: {test_results_file}")
        
        # 打印测试结果摘要 - 只在主进程打印
        if self.is_main_process:
            print(f"\n🧪 测试集推理结果摘要:")
            print(f"   样本数量: {num_batches}")
            print(f"   平均生成IoU: {avg_gen_iou:.4f}")
            print(f"   平均生成箱子数: {avg_generated_boxes:.1f}")
            print(f"   平均GT箱子数: {avg_gt_boxes:.1f}")
            print(f"   生成率: {avg_generated_boxes/max(avg_gt_boxes, 1):.2f}")
            print(f"   总体平均误差: {avg_overall_error:.4f}")
            print(f"   X误差: {avg_x_error:.4f} | Y误差: {avg_y_error:.4f} | Z误差: {avg_z_error:.4f}")
            print(f"   W误差: {avg_w_error:.4f} | H误差: {avg_h_error:.4f} | L误差: {avg_l_error:.4f}")
        
        return {
            'generation_iou': avg_gen_iou,
            'avg_generated_boxes': avg_generated_boxes,
            'avg_gt_boxes': avg_gt_boxes,
            'avg_overall_error': avg_overall_error,
            'avg_x_error': avg_x_error,
            'avg_y_error': avg_y_error,
            'avg_z_error': avg_z_error,
            'avg_w_error': avg_w_error,
            'avg_h_error': avg_h_error,
            'avg_l_error': avg_l_error
        }
    
    def _save_checkpoint(self, is_best: bool = False, is_best_generation: bool = False):
        """保存checkpoint"""
        if not self.is_main_process:
            return
        checkpoint = {
            'model': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'epoch': self.current_epoch,
            'current_phase_idx': self.current_phase_idx,
            'best_val_loss': self.best_val_loss,
            'best_generation_loss': self.best_generation_loss,
            'training_stats': self.training_stats,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'metadata': {
                'total_epochs': sum(phase.epochs for phase in self.training_phases),
                'training_phases': [asdict(phase) for phase in self.training_phases],
                'device': str(self.device),
                'world_size': self.world_size,
                'local_rank': self.local_rank
            }
        }
        
        # 只保存最新checkpoint (用于resume)
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳验证损失模型
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best_val.pt"
            torch.save(checkpoint, best_path)
            if self.is_main_process:
                print(f"✅ 最佳验证模型已保存: {best_path}")
        
        # 保存最佳生成模型
        if is_best_generation:
            best_gen_path = self.checkpoint_dir / "checkpoint_best_generation.pt"
            torch.save(checkpoint, best_gen_path)
            if self.is_main_process:
                print(f"✅ 最佳生成模型已保存: {best_gen_path}")
        
        if not (is_best or is_best_generation):
            if self.is_main_process:
                print(f"✅ 最新checkpoint已保存: {latest_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载checkpoint"""
        if self.is_main_process:
            print(f"📂 加载checkpoint: {checkpoint_path}")
        
        # 🔧 修复PyTorch 2.6的weights_only问题
        # 由于这是我们自己的checkpoint文件，设置weights_only=False是安全的
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 加载模型状态
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])
        
        # 加载优化器状态
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # 加载调度器状态
        if self.scheduler and checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # 加载scaler状态
        if self.scaler and checkpoint['scaler']:
            self.scaler.load_state_dict(checkpoint['scaler'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch'] + 1  # 从下一个epoch开始
        self.current_phase_idx = checkpoint.get('current_phase_idx', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_generation_loss = checkpoint.get('best_generation_loss', float('inf'))
        self.training_stats = checkpoint.get('training_stats', [])
        
        # 🔧 修复：重新计算正确的训练阶段（以防checkpoint中的phase_idx不准确）
        epoch_count = 0
        old_phase_idx = self.current_phase_idx
        for phase_idx, phase in enumerate(self.training_phases):
            if self.is_main_process:
                print(f"   检查阶段{phase_idx} ({phase.name}): epochs {epoch_count}-{epoch_count + phase.epochs - 1}")
                print(f"     条件: {self.current_epoch} < {epoch_count + phase.epochs} = {self.current_epoch < epoch_count + phase.epochs}")
            
            if self.current_epoch < epoch_count + phase.epochs:
                self.current_phase_idx = phase_idx
                if self.is_main_process:
                    print(f"     -> 选择阶段{phase_idx} ({phase.name})")
                break
            epoch_count += phase.epochs
        
        if self.is_main_process:
            print(f"🔄 阶段索引: {old_phase_idx} -> {self.current_phase_idx}")
            print(f"✅ 已恢复到epoch {self.current_epoch}")
            print(f"🎯 当前训练阶段: {self.training_phases[self.current_phase_idx].name} (阶段{self.current_phase_idx})")
            print(f"📊 最佳验证损失: {self.best_val_loss:.4f}")
            print(f"🎯 最佳生成损失: {self.best_generation_loss:.4f}")
    
    def _evaluate_best_model_on_test(self, is_best_val: bool = False, is_best_generation: bool = False):
        """在获得最佳模型时在test集上进行推理并记录结果"""
        if not self.is_main_process:
            return
        
        # 检查是否启用最佳模型test推理
        best_model_testing_config = self.training_config.get('optimizations', {}).get('best_model_testing', {})
        if not best_model_testing_config.get('enabled', True):
            return
        
        # 只有在获得最佳模型时才进行test推理
        if not (is_best_val or is_best_generation):
            return
            
        print(f"\n🎯 检测到{'最佳验证模型' if is_best_val else ''}{'和' if is_best_val and is_best_generation else ''}{'最佳生成模型' if is_best_generation else ''}，开始在test集上进行推理...")
        
        # 保存当前模型状态
        current_model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        try:
            # 在test集上进行推理
            test_results = self._test_inference()
            
            # 记录test结果到SwanLab
            if self.use_swanlab and best_model_testing_config.get('save_results', True):
                # 为最佳验证模型记录test结果
                if is_best_val:
                    swanlab.log({
                        'test_best_val/generation_iou': test_results['generation_iou'],
                        'test_best_val/overall_mean_error': test_results.get('avg_overall_error', 0.0),
                        'test_best_val/x_error': test_results.get('avg_x_error', 0.0),
                        'test_best_val/y_error': test_results.get('avg_y_error', 0.0),
                        'test_best_val/z_error': test_results.get('avg_z_error', 0.0),
                        'test_best_val/w_error': test_results.get('avg_w_error', 0.0),
                        'test_best_val/h_error': test_results.get('avg_h_error', 0.0),
                        'test_best_val/l_error': test_results.get('avg_l_error', 0.0),
                        'test_best_val/avg_generated_boxes': test_results.get('avg_generated_boxes', 0.0),
                        'test_best_val/avg_gt_boxes': test_results.get('avg_gt_boxes', 0.0),
                        'test_best_val/generation_rate': test_results.get('avg_generated_boxes', 0.0) / max(test_results.get('avg_gt_boxes', 1), 1),
                        'test_best_val/epoch': self.current_epoch
                    }, step=self.current_epoch)
                
                # 为最佳生成模型记录test结果
                if is_best_generation:
                    swanlab.log({
                        'test_best_gen/generation_iou': test_results['generation_iou'],
                        'test_best_gen/overall_mean_error': test_results.get('avg_overall_error', 0.0),
                        'test_best_gen/x_error': test_results.get('avg_x_error', 0.0),
                        'test_best_gen/y_error': test_results.get('avg_y_error', 0.0),
                        'test_best_gen/z_error': test_results.get('avg_z_error', 0.0),
                        'test_best_gen/w_error': test_results.get('avg_w_error', 0.0),
                        'test_best_gen/h_error': test_results.get('avg_h_error', 0.0),
                        'test_best_gen/l_error': test_results.get('avg_l_error', 0.0),
                        'test_best_gen/avg_generated_boxes': test_results.get('avg_generated_boxes', 0.0),
                        'test_best_gen/avg_gt_boxes': test_results.get('avg_gt_boxes', 0.0),
                        'test_best_gen/generation_rate': test_results.get('avg_generated_boxes', 0.0) / max(test_results.get('avg_gt_boxes', 1), 1),
                        'test_best_gen/epoch': self.current_epoch
                    }, step=self.current_epoch)
            
            # 打印test结果
            print(f"🧪 Test集推理结果 (Epoch {self.current_epoch}):")
            print(f"   生成IoU: {test_results['generation_iou']:.4f}")
            print(f"   总体平均误差: {test_results.get('avg_overall_error', 0.0):.4f}")
            print(f"   X误差: {test_results.get('avg_x_error', 0.0):.4f} | Y误差: {test_results.get('avg_y_error', 0.0):.4f} | Z误差: {test_results.get('avg_z_error', 0.0):.4f}")
            print(f"   W误差: {test_results.get('avg_w_error', 0.0):.4f} | H误差: {test_results.get('avg_h_error', 0.0):.4f} | L误差: {test_results.get('avg_l_error', 0.0):.4f}")
            print(f"   平均生成数量: {test_results.get('avg_generated_boxes', 0.0):.1f} | 平均GT数量: {test_results.get('avg_gt_boxes', 0.0):.1f}")
            print(f"   生成率: {test_results.get('avg_generated_boxes', 0.0) / max(test_results.get('avg_gt_boxes', 1), 1):.2f}")
            
        except Exception as e:
            print(f"❌ Test集推理失败: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 确保模型状态没有被改变
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(current_model_state)
            else:
                self.model.load_state_dict(current_model_state)

    def train(self):
        """开始训练"""
        # 多GPU同步点 - 确保所有进程都准备好
        if self.world_size > 1 and torch.cuda.device_count() > 1:
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                if self.is_main_process:
                    print("✅ 所有进程同步完成，开始训练...")
        
        if self.is_main_process:
            print("🚀 开始分段式训练...")
        
        total_epochs = sum(phase.epochs for phase in self.training_phases)
        
        # 🔧 修复：不再重新计算阶段索引，使用_load_checkpoint中已正确计算的值
        for phase_idx in range(self.current_phase_idx, len(self.training_phases)):
            phase = self.training_phases[phase_idx]
            
            if self.is_main_process:
                print(f"\n{'='*60}")
                print(f"🎯 训练阶段: {phase.name}")
                print(f"📝 {phase.description}")
                print(f"⏱️  持续{phase.epochs}个epochs")
                print(f"🎲 Teacher Forcing: {phase.teacher_forcing_ratio}")
                print(f"📊 Scheduled Sampling: {phase.scheduled_sampling}")
                print(f"{'='*60}")
            
            # 为当前阶段创建损失函数
            loss_fn = self._create_loss_function(phase.name)
            loss_fn = loss_fn.to(self.device)
            
            # 计算阶段内的起始epoch
            phase_start_epoch = sum(p.epochs for p in self.training_phases[:phase_idx])
            
            for epoch_in_phase in range(phase.epochs):
                if self.current_epoch < phase_start_epoch + epoch_in_phase:
                    continue  # 跳过已训练的epoch
                
                # 🔧 修复：计算正确的阶段内epoch位置
                actual_epoch_in_phase = self.current_epoch - phase_start_epoch + 1
                
                if self.is_main_process:
                    print(f"\n--- Epoch {self.current_epoch + 1}/{total_epochs} "
                          f"(Phase: {phase.name}, {actual_epoch_in_phase}/{phase.epochs}) ---")
                
                # 训练
                train_stats = self._train_epoch(phase, epoch_in_phase, loss_fn)
                
                # 验证
                val_results = self._validate_epoch(loss_fn, phase)
                val_tf_loss = val_results['tf_total_loss']
                val_gen_iou = val_results['generation_iou']
                val_tf_iou = val_results['tf_mean_iou']
                
                # 完善统计信息
                train_stats.val_loss = val_tf_loss
                train_stats.val_generation_loss = 1.0 - val_gen_iou  # 将IoU转换为损失形式（用于兼容现有代码）
                train_stats.val_mean_iou = val_tf_iou
                
                # 判断是否为最佳模型
                is_best_val = val_tf_loss < self.best_val_loss
                # 使用IoU判断最佳生成模型：更高的IoU = 更好的模型
                is_best_generation = val_gen_iou > (1.0 - self.best_generation_loss)
                
                if is_best_val:
                    self.best_val_loss = val_tf_loss
                if is_best_generation:
                    # 更新最佳生成损失（存储为1-IoU的形式，保持与现有代码兼容）
                    self.best_generation_loss = 1.0 - val_gen_iou
                
                # SwanLab日志记录
                if self.use_swanlab:
                    # 训练loss组件
                    swanlab.log({
                        'train/total_loss': train_stats.train_loss,
                        'train/classification_loss': train_stats.train_classification_loss,
                        'train/iou_loss': train_stats.train_iou_loss,
                        'train/delta_loss': train_stats.train_delta_loss,
                        'train/mean_iou': train_stats.train_mean_iou,
                        
                        # Teacher Forcing验证loss组件  
                        'val/tf_total_loss': val_results['tf_total_loss'],
                        'val/tf_classification_loss': val_results['tf_classification_loss'],
                        'val/tf_iou_loss': val_results['tf_iou_loss'],
                        'val/tf_delta_loss': val_results['tf_delta_loss'],
                        'val/tf_eos_loss': val_results['tf_eos_loss'],
                        'val/tf_mean_iou': val_results['tf_mean_iou'],
                        
                        # Autoregressive验证loss组件
                        'val/ar_mean_iou': val_results['generation_iou'],
                        # 生成统计
                        'val/generation_rate': val_results['avg_generated_boxes'] / max(val_results['avg_gt_boxes'], 1),
                        # 维度误差指标
                        'val/overall_mean_error': val_results.get('avg_overall_error', 0.0),
                        'val/x_error': val_results.get('avg_x_error', 0.0),
                        'val/y_error': val_results.get('avg_y_error', 0.0),
                        'val/z_error': val_results.get('avg_z_error', 0.0),
                        'val/w_error': val_results.get('avg_w_error', 0.0),
                        'val/h_error': val_results.get('avg_h_error', 0.0),
                        'val/l_error': val_results.get('avg_l_error', 0.0),
                        
                        # 训练参数
                        'train/teacher_forcing_ratio': train_stats.teacher_forcing_ratio,
                        'train/learning_rate': train_stats.learning_rate,
                        'train/adaptive_cls_weight': train_stats.adaptive_cls_weight,
                        'train/adaptive_delta_weight': train_stats.adaptive_delta_weight,
                        
                        # epoch信息
                        # 'epoch': self.current_epoch,
                        # 'phase_idx': phase_idx,
                        'epoch_time': train_stats.epoch_time
                    }, step=self.current_epoch)  # 🔧 修复：添加step参数确保resume时正确续接

                # 控制台输出详细信息
                # 控制台输出详细信息 - 只在主进程打印
                if self.is_main_process:
                    print(f"📊 训练损失详情:")
                    print(f"   总损失: {train_stats.train_loss:.4f}")
                    print(f"   分类损失: {train_stats.train_classification_loss:.4f}")
                    print(f"   IoU损失: {train_stats.train_iou_loss:.4f}")
                    print(f"   Delta损失: {train_stats.train_delta_loss:.4f}")
                    print(f"🎯 验证损失详情:")
                    print(f"   TF总损失: {val_results['tf_total_loss']:.4f}")
                    print(f"   TF分类: {val_results['tf_classification_loss']:.4f} | TF IoU: {val_results['tf_iou_loss']:.4f} | TF Delta: {val_results['tf_delta_loss']:.4f} | TF EOS: {val_results['tf_eos_loss']:.4f}")
                    print(f"   TF mean IoU: {val_results['tf_mean_iou']:.4f}")
                    print(f"📊 生成模式详情:")
                    print(f"   生成IoU评估: {val_results['generation_iou']:.4f}")
                    print(f"📏 维度误差详情:")
                    print(f"   总体平均误差: {val_results.get('avg_overall_error', 0.0):.4f}")
                    print(f"   X误差: {val_results.get('avg_x_error', 0.0):.4f} | Y误差: {val_results.get('avg_y_error', 0.0):.4f} | Z误差: {val_results.get('avg_z_error', 0.0):.4f}")
                    print(f"   W误差: {val_results.get('avg_w_error', 0.0):.4f} | H误差: {val_results.get('avg_h_error', 0.0):.4f} | L误差: {val_results.get('avg_l_error', 0.0):.4f}")
                    print(f"📦 箱子统计:")
                    print(f"   平均生成数量: {val_results['avg_generated_boxes']:.1f} | 平均GT数量: {val_results['avg_gt_boxes']:.1f} | 生成率: {val_results['avg_generated_boxes']/max(val_results['avg_gt_boxes'], 1):.2f}")
                    print(f"⚖️  权重详情:")
                    print(f"   自适应分类权重: {train_stats.adaptive_cls_weight:.3f}")
                    print(f"   自适应Delta权重: {train_stats.adaptive_delta_weight:.3f}")
                    print(f"   TF比例: {train_stats.teacher_forcing_ratio:.3f} | 学习率: {train_stats.learning_rate:.2e}")
                
                # 保存checkpoint
                if (self.current_epoch + 1) % self.training_config.get('save_interval', 10) == 0:
                    self._save_checkpoint(is_best_val, is_best_generation)
                elif is_best_val or is_best_generation:
                    self._save_checkpoint(is_best_val, is_best_generation)
                
                # 在获得最佳模型时在test集上进行推理
                self._evaluate_best_model_on_test(is_best_val, is_best_generation)
                
                self.current_epoch += 1
        
        if self.is_main_process:
            print("\n🎉 训练完成!")
            print(f"📊 最佳验证损失: {self.best_val_loss:.4f}")
            print(f"🎯 最佳生成损失: {self.best_generation_loss:.4f}")
        
        # 保存最终checkpoint
        if self.is_main_process:
            self._save_checkpoint()
        
        # 在测试集上进行推理（训练结束时的最终评估）
        if self.is_main_process:
            print("\n🧪 训练结束，开始在测试集上进行最终推理...")
        test_results = self._test_inference()
        
        # 记录测试结果到SwanLab
        if self.use_swanlab:
            swanlab.log({
                'test/generation_iou': test_results['generation_iou'],
                'test/overall_mean_error': test_results.get('avg_overall_error', 0.0),
                'test/x_error': test_results.get('avg_x_error', 0.0),
                'test/y_error': test_results.get('avg_y_error', 0.0),
                'test/z_error': test_results.get('avg_z_error', 0.0),
                'test/w_error': test_results.get('avg_w_error', 0.0),
                'test/h_error': test_results.get('avg_h_error', 0.0),
                'test/l_error': test_results.get('avg_l_error', 0.0),
                'test/avg_generated_boxes': test_results.get('avg_generated_boxes', 0.0),
                'test/avg_gt_boxes': test_results.get('avg_gt_boxes', 0.0),
                'test/generation_rate': test_results.get('avg_generated_boxes', 0.0) / max(test_results.get('avg_gt_boxes', 1), 1)
            }, step=self.current_epoch)  # 🔧 修复：添加step参数
            swanlab.finish()


def setup_distributed(local_rank: int, world_size: int):
    """设置分布式训练"""
    # 检查是否已经有环境变量设置
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print(f"⚠️  CUDA不可用，无法进行分布式训练")
        return False
    
    # 检查GPU数量
    if torch.cuda.device_count() < world_size:
        print(f"⚠️  GPU数量({torch.cuda.device_count()})少于world_size({world_size})，无法进行分布式训练")
        return False
    
    try:
        dist.init_process_group(
            backend='nccl',
            rank=local_rank,
            world_size=world_size
        )
        print(f"✅ 分布式训练初始化成功 (Rank {local_rank}/{world_size})")
        return True
    except Exception as e:
        print(f"❌ 分布式训练初始化失败: {e}")
        return False


def cleanup_distributed():
    """清理分布式训练"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(
    config_dir: str = ".",
    experiment_name: str = "primitive_3d_exp",
    resume_from: Optional[str] = None,
    local_rank: int = 0,
    world_size: int = 1,
    validation_samples: List[int] = None
):
    """主训练函数"""
    
    # 设置分布式训练
    distributed_success = False
    if world_size > 1:
        distributed_success = setup_distributed(local_rank, world_size)
        if not distributed_success:
            print(f"⚠️  分布式训练设置失败，回退到单GPU模式")
            world_size = 1
            local_rank = 0
    
    try:
        # 设置随机种子
        torch.manual_seed(42 + local_rank)
        np.random.seed(42 + local_rank)
        random.seed(42 + local_rank)
        
        # 加载统一配置 - 只在主进程显示详细信息
        config_loader = ConfigLoader()
        config_loader.load_unified_config(os.path.join(config_dir, "training_config.yaml"))
        
        if local_rank == 0:
            print(f"✅ 成功加载统一配置文件: {os.path.join(config_dir, 'training_config.yaml')}")
            # 显示配置信息
            global_config = config_loader.get_global_config()
            print(f"📊 配置版本: {global_config.get('version', 'unknown')}")
            
            # 显示训练阶段信息
            phases_config = config_loader.get_training_phases()
            total_epochs = sum(phase.get('epochs', 0) for phase in phases_config.values())
            print(f"🎯 训练阶段: {len(phases_config)}个阶段，总计{total_epochs}个epoch")
            for phase_name, phase_config in phases_config.items():
                print(f"  - {phase_name}: {phase_config.get('epochs', 0)} epochs")
        
        # 创建训练器
        trainer = AdvancedTrainer(
            config_loader=config_loader,
            experiment_name=experiment_name,
            resume_from=resume_from,
            local_rank=local_rank,
            world_size=world_size,
            use_swanlab=True,
            validation_samples=validation_samples or [0, 1, 2, 3, 4]
        )
        
        # 开始训练
        trainer.train()
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        raise
    finally:
        # 清理分布式训练
        if distributed_success:
            cleanup_distributed()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Primitive Detection Training")
    parser.add_argument("--config-dir", default=".", help="配置文件目录")
    parser.add_argument("--experiment-name", default="primitive_3d_exp", help="实验名称")
    parser.add_argument("--resume-from", default=None, help="恢复训练的checkpoint路径")
    parser.add_argument("--local-rank", type=int, default=0, help="本地GPU排名")
    parser.add_argument("--world-size", type=int, default=1, help="总GPU数量")
    parser.add_argument("--validation-samples", nargs='+', type=int, default=[0, 1, 2, 3, 4], 
                       help="验证可视化样本索引")
    
    args = parser.parse_args()
    
    main(
        config_dir=args.config_dir,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from,
        local_rank=args.local_rank,
        world_size=args.world_size,
        validation_samples=args.validation_samples
    ) 