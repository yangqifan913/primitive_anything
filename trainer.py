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
from dataloader_3d import create_dataloader, generate_equivalent_box_representations, normalize_angle

def select_best_equivalent_representation(pred_box, equivalent_boxes):
    """
    选择与预测box旋转角度误差最小的等价表示（过滤掉角度大于60度的等效box）
    
    Args:
        pred_box: 预测box (x, y, z, l, w, h, roll, pitch, yaw)
        equivalent_boxes: 等价表示列表 [(x, y, z, l, w, h, roll, pitch, yaw), ...]
    
    Returns:
        best_box: 最优的等价表示
        min_loss: 最小旋转loss
    """
    import math
    
    # 🔧 过滤掉任何角度大于180度的等效box
    valid_equivalent_boxes = []
    for equiv_box in equivalent_boxes:
        roll, pitch, yaw = equiv_box[6], equiv_box[7], equiv_box[8]

        if abs(roll) <= math.pi/3 and abs(pitch) <= math.pi/3 and abs(yaw) <= math.pi/3:
            valid_equivalent_boxes.append(equiv_box)
    
    # 如果过滤后没有有效的等效box，报错
    if len(valid_equivalent_boxes) == 0:
        raise ValueError(f"❌ 错误：所有等效box都包含大于60度的角度！原始等效box数量: {len(equivalent_boxes)}")
    
    # 🔍 打印筛选统计信息
    # print(f"     📊 等效box筛选统计: 原始={len(equivalent_boxes)}, 筛选后={len(valid_equivalent_boxes)}")
    
    min_loss = float('inf')
    best_box = valid_equivalent_boxes[0]  # 默认选择第一个有效box
    
    pred_roll, pred_pitch, pred_yaw = pred_box[6], pred_box[7], pred_box[8]
    
    # 确保角度值是标量，不是列表
    if isinstance(pred_roll, list):
        pred_roll = pred_roll[0]
    if isinstance(pred_pitch, list):
        pred_pitch = pred_pitch[0]
    if isinstance(pred_yaw, list):
        pred_yaw = pred_yaw[0]
    
    def angular_error(pred_angle, gt_angle):
        """计算角度误差，简单L1 loss"""
        return abs(pred_angle - gt_angle)
    
    # 只在有效的等效box中选择最优的
    for equiv_box in valid_equivalent_boxes:
        gt_roll, gt_pitch, gt_yaw = equiv_box[6], equiv_box[7], equiv_box[8]
        
        # 计算简单L1 loss的旋转角度误差
        roll_loss = angular_error(pred_roll, gt_roll)
        pitch_loss = angular_error(pred_pitch, gt_pitch)
        yaw_loss = angular_error(pred_yaw, gt_yaw)
        
        total_loss = roll_loss + pitch_loss + yaw_loss
        
        if total_loss < min_loss:
            min_loss = total_loss
            best_box = equiv_box
    
    return best_box, min_loss


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
    train_eos_loss: float
    train_mean_iou: float
    val_loss: float
    val_generation_loss: float
    val_mean_iou: float
    val_generation_iou: float
    learning_rate: float
    adaptive_cls_weight: float
    adaptive_delta_weight: float
    epoch_time: float
    
    # 🔍 添加旋转角度损失统计
    train_roll_cls_loss: float = 0.0
    train_pitch_cls_loss: float = 0.0
    train_yaw_cls_loss: float = 0.0
    train_roll_delta_loss: float = 0.0
    train_pitch_delta_loss: float = 0.0
    train_yaw_delta_loss: float = 0.0


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
        
        # 设置pad_id属性
        self.pad_id = global_config['pad_id']
        
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
            # 离散化参数 - 3个属性
            num_discrete_position=discretization['num_discrete_position'],
            num_discrete_rotation=discretization['num_discrete_rotation'],
            num_discrete_size=discretization['num_discrete_size'],
            
            # 连续范围 - 3个属性
            continuous_range_position=continuous_ranges['position'],
            # 🔧 修复：将角度制转换为弧度制
            continuous_range_rotation=[[math.radians(continuous_ranges['rotation'][0][0]), math.radians(continuous_ranges['rotation'][0][1])],
                                      [math.radians(continuous_ranges['rotation'][1][0]), math.radians(continuous_ranges['rotation'][1][1])],
                                      [math.radians(continuous_ranges['rotation'][2][0]), math.radians(continuous_ranges['rotation'][2][1])]],
            continuous_range_size=continuous_ranges['size'],
            
            # 嵌入维度 - 3个属性
            dim_position_embed=embeddings['dim_position_embed'],
            dim_rotation_embed=embeddings['dim_rotation_embed'],
            dim_size_embed=embeddings['dim_size_embed'],
            
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
        
        # 使用修复的create_dataloader函数
        from dataloader_3d import create_dataloader
        
        # 准备数据集参数
        dataset_kwargs = {
            'data_root': dataset_config['data_root'],
            'stage': "train",
            'max_boxes': global_config['max_seq_len'],
            'image_size': global_config['image_size'],
            'continuous_ranges': data_config['continuous_ranges'],
            'augmentation_config': data_config['augmentation']
        }
        
        # 准备DataLoader参数
        dataloader_kwargs = {
            'batch_size': dataloader_config['batch_size'],
            'shuffle': True,
            'num_workers': dataloader_config['num_workers'],
            'pin_memory': dataloader_config['pin_memory'],
            'drop_last': True,
            'prefetch_factor': dataloader_config['prefetch_factor'],
            'persistent_workers': dataloader_config['persistent_workers']
        }
        
        # 先创建数据集（用于分布式采样器）
        from dataloader_3d import Box3DDataset
        
        train_dataset = Box3DDataset(**dataset_kwargs)
        
        val_dataset_kwargs = dataset_kwargs.copy()
        val_dataset_kwargs['stage'] = "val"
        val_dataset_kwargs['augmentation_config'] = {}  # 验证时不使用数据增强
        val_dataset = Box3DDataset(**val_dataset_kwargs)
        
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
        
        # 更新batch_size
        dataloader_kwargs['batch_size'] = effective_batch_size
        
        # 创建训练DataLoader（使用修复的create_dataloader）
        self.train_loader = create_dataloader(**dataset_kwargs, **dataloader_kwargs)
        
        # 创建验证DataLoader
        val_dataloader_kwargs = dataloader_kwargs.copy()
        val_dataloader_kwargs['shuffle'] = False
        val_dataloader_kwargs['drop_last'] = False
        
        self.val_loader = create_dataloader(**val_dataset_kwargs, **val_dataloader_kwargs)
        
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
            # 离散化参数 - 9个单独属性
            num_discrete_x=discretization['num_discrete_position'],
            num_discrete_y=discretization['num_discrete_position'],
            num_discrete_z=discretization['num_discrete_position'],
            num_discrete_w=discretization['num_discrete_size'],
            num_discrete_h=discretization['num_discrete_size'],
            num_discrete_l=discretization['num_discrete_size'],
            num_discrete_roll=discretization['num_discrete_rotation'],
            num_discrete_pitch=discretization['num_discrete_rotation'],
            num_discrete_yaw=discretization['num_discrete_rotation'],
            
            # 连续范围参数 - 9个单独属性
            continuous_range_x=continuous_ranges['position'][0],
            continuous_range_y=continuous_ranges['position'][1],
            continuous_range_z=continuous_ranges['position'][2],
            continuous_range_w=continuous_ranges['size'][0],
            continuous_range_h=continuous_ranges['size'][1],
            continuous_range_l=continuous_ranges['size'][2],
            # 🔧 修复：将角度制转换为弧度制 [-180°, 180°] -> [-π, π]
            continuous_range_roll=[math.radians(continuous_ranges['rotation'][0][0]), math.radians(continuous_ranges['rotation'][0][1])],
            continuous_range_pitch=[math.radians(continuous_ranges['rotation'][1][0]), math.radians(continuous_ranges['rotation'][1][1])],
            continuous_range_yaw=[math.radians(continuous_ranges['rotation'][2][0]), math.radians(continuous_ranges['rotation'][2][1])],
            
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
    
    def _prepare_targets_with_equivalent_boxes(self, batch, outputs):
        """
        使用等价表示优化目标数据，选择旋转L1 loss最小的表示
        
        Args:
            batch: 输入batch数据
            outputs: 模型输出
            
        Returns:
            targets: 优化后的目标数据
        """
        batch_size = batch['x'].shape[0]
        device = batch['x'].device
        
        # 初始化目标数据（batch已经在正确的设备上）
        targets = {
            'x': batch['x'].clone(),
            'y': batch['y'].clone(),
            'z': batch['z'].clone(),
            'w': batch['w'].clone(),
            'h': batch['h'].clone(),
            'l': batch['l'].clone(),
            'roll': batch['roll'].clone(),
            'pitch': batch['pitch'].clone(),
            'yaw': batch['yaw'].clone(),
        }
        
        # 如果有等价表示，进行优化
        if 'equivalent_boxes' in batch:
            equivalent_boxes = batch['equivalent_boxes']
            
            # 对每个样本的每个box进行优化
            for b in range(batch_size):
                for s in range(batch['x'].shape[1]):
                    # 跳过padding
                    if batch['x'][b, s] == self.pad_id:
                        continue
                    
                    # 构建预测box（保持张量格式，避免设备不匹配）
                    pred_box_tensor = torch.stack([
                        outputs['continuous_dict']['x_continuous'][b, s],
                        outputs['continuous_dict']['y_continuous'][b, s],
                        outputs['continuous_dict']['z_continuous'][b, s],
                        outputs['continuous_dict']['l_continuous'][b, s],
                        outputs['continuous_dict']['w_continuous'][b, s],
                        outputs['continuous_dict']['h_continuous'][b, s],
                        outputs['continuous_dict']['roll_continuous'][b, s],
                        outputs['continuous_dict']['pitch_continuous'][b, s],
                        outputs['continuous_dict']['yaw_continuous'][b, s],
                    ], dim=0)  # [9]
                    
                    # 转换为Python列表用于等价box选择
                    pred_box = pred_box_tensor.detach().cpu().numpy().tolist()
                    
                    # 确保pred_box是扁平的列表，不是嵌套列表
                    if isinstance(pred_box[6], list):
                        pred_box = [item[0] if isinstance(item, list) else item for item in pred_box]
                    
                    # 获取该box的等价表示
                    if s < len(equivalent_boxes[b]):
                        equiv_boxes = equivalent_boxes[b][s]
                        
                        # 选择最优的等价表示
                        best_box, min_loss = select_best_equivalent_representation(pred_box, equiv_boxes)
                        
                        # 更新目标数据（确保在正确的设备上）
                        device = targets['x'].device
                        targets['x'][b, s] = torch.tensor(best_box[0], device=device, dtype=targets['x'].dtype)
                        targets['y'][b, s] = torch.tensor(best_box[1], device=device, dtype=targets['y'].dtype)
                        targets['z'][b, s] = torch.tensor(best_box[2], device=device, dtype=targets['z'].dtype)
                        targets['l'][b, s] = torch.tensor(best_box[3], device=device, dtype=targets['l'].dtype)
                        targets['w'][b, s] = torch.tensor(best_box[4], device=device, dtype=targets['w'].dtype)
                        targets['h'][b, s] = torch.tensor(best_box[5], device=device, dtype=targets['h'].dtype)
                        targets['roll'][b, s] = torch.tensor(best_box[6], device=device, dtype=targets['roll'].dtype)
                        targets['pitch'][b, s] = torch.tensor(best_box[7], device=device, dtype=targets['pitch'].dtype)
                        targets['yaw'][b, s] = torch.tensor(best_box[8], device=device, dtype=targets['yaw'].dtype)
        
        return targets
    
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
            'roll': batch['roll'].to(self.device),
            'pitch': batch['pitch'].to(self.device),
            'yaw': batch['yaw'].to(self.device),
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
            # 统一使用forward_with_predictions - 3属性格式
            # 构建3属性张量
            position = torch.stack([inputs['x'], inputs['y'], inputs['z']], dim=-1)  # [B, seq_len, 3]
            rotation = torch.stack([inputs['roll'], inputs['pitch'], inputs['yaw']], dim=-1)  # [B, seq_len, 3]
            size = torch.stack([inputs['l'], inputs['w'], inputs['h']], dim=-1)  # [B, seq_len, 3]
            
            outputs = model.forward_with_predictions(
                position=position,
                rotation=rotation,
                size=size,
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
        
        # ===== 第一次推理：Teacher Forcing模式（不需要梯度） =====
        # 使用Ground Truth作为输入，得到模型的预测结果
        with torch.no_grad():
            # 构建3属性张量
            position = torch.stack([targets['x'], targets['y'], targets['z']], dim=-1)  # [B, seq_len, 3]
            rotation = torch.stack([targets['roll'], targets['pitch'], targets['yaw']], dim=-1)  # [B, seq_len, 3]
            size = torch.stack([targets['l'], targets['w'], targets['h']], dim=-1)  # [B, seq_len, 3]
            
            predicted_output = model.forward_with_predictions(
                position=position,
                rotation=rotation,
                size=size,
                image=rgbxyz
            )
        
        # 从预测输出中提取连续值
        continuous_predictions = predicted_output['continuous_dict']
        
        # 计算序列长度和创建mask
        sequence_lengths = self._compute_sequence_lengths(targets)
        
        # 构建混合输入序列（保持梯度）
        mixed_inputs = {}
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
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
        # 构建3属性张量
        position = torch.stack([mixed_inputs['x'], mixed_inputs['y'], mixed_inputs['z']], dim=-1)  # [B, seq_len, 3]
        rotation = torch.stack([targets['roll'], targets['pitch'], targets['yaw']], dim=-1)  # [B, seq_len, 3]
        size = torch.stack([mixed_inputs['l'], mixed_inputs['w'], mixed_inputs['h']], dim=-1)  # [B, seq_len, 3]
        
        return model.forward_with_predictions(
            position=position,
            rotation=rotation,
            size=size,
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
        all_logits = {f'{attr}_logits': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
        all_deltas = {f'{attr}_delta': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
        all_continuous = {f'{attr}_continuous': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
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
            
            # 预测3个属性（保持梯度，使用Gumbel Softmax）
            gumbel_temp = self.incremental_temperature
            
            # 预测位置属性 (x, y, z)
            pos_result = model.predict_3d_vector_with_continuous_embed(
                step_embed, 'position', prev_embeds=None, use_gumbel=True, temperature=gumbel_temp
            )
            pos_logits = pos_result['logits']
            pos_deltas = pos_result['deltas'] 
            pos_continuous = pos_result['continuous']
            pos_embeds = [pos_result['embed']]
            
            # 预测旋转属性 (roll, pitch, yaw)
            rot_result = model.predict_3d_vector_with_continuous_embed(
                step_embed, 'rotation', prev_embeds=pos_embeds, use_gumbel=True, temperature=gumbel_temp
            )
            rot_logits = rot_result['logits']
            rot_deltas = rot_result['deltas']
            rot_continuous = rot_result['continuous']
            rot_embeds = [rot_result['embed']]
            
            # 预测尺寸属性 (w, h, l)
            size_result = model.predict_3d_vector_with_continuous_embed(
                step_embed, 'size', prev_embeds=pos_embeds + rot_embeds, use_gumbel=True, temperature=gumbel_temp
            )
            size_logits = size_result['logits']
            size_deltas = size_result['deltas']
            size_continuous = size_result['continuous'] 
            size_embeds = [size_result['embed']]
            
            # 将3D向量分解为单独属性
            x_continuous, y_continuous, z_continuous = pos_continuous[:, 0], pos_continuous[:, 1], pos_continuous[:, 2]
            w_continuous, h_continuous, l_continuous = size_continuous[:, 0], size_continuous[:, 1], size_continuous[:, 2]
            roll_continuous, pitch_continuous, yaw_continuous = rot_continuous[:, 0], rot_continuous[:, 1], rot_continuous[:, 2]
            
            # EOS预测 - 需要所有属性的embedding
            eos_input = torch.cat([step_embed] + pos_embeds + rot_embeds + size_embeds, dim=-1)
            eos_logits = model.to_eos_logits(eos_input).squeeze(-1)
            
            # 保存这一步的输出
            # 位置属性 (x, y, z) - pos_logits是[B, sum(num_bins)]形状，需要按维度分割
            num_bins = model.num_discrete_position
            x_logits = pos_logits[:, 0:num_bins]  # [B, num_bins]
            y_logits = pos_logits[:, num_bins:2*num_bins]  # [B, num_bins]
            z_logits = pos_logits[:, 2*num_bins:3*num_bins]  # [B, num_bins]
            
            all_logits['x_logits'].append(x_logits)
            all_logits['y_logits'].append(y_logits)
            all_logits['z_logits'].append(z_logits)
            
            all_deltas['x_delta'].append(pos_deltas[:, 0])
            all_deltas['y_delta'].append(pos_deltas[:, 1])
            all_deltas['z_delta'].append(pos_deltas[:, 2])
            
            all_continuous['x_continuous'].append(x_continuous)
            all_continuous['y_continuous'].append(y_continuous)
            all_continuous['z_continuous'].append(z_continuous)
            
            # 尺寸属性 (w, h, l) - size_logits是[B, sum(num_bins)]形状，需要按维度分割
            size_num_bins = model.num_discrete_size
            w_logits = size_logits[:, 0:size_num_bins]  # [B, num_bins]
            h_logits = size_logits[:, size_num_bins:2*size_num_bins]  # [B, num_bins]
            l_logits = size_logits[:, 2*size_num_bins:3*size_num_bins]  # [B, num_bins]
            
            all_logits['w_logits'].append(w_logits)
            all_logits['h_logits'].append(h_logits)
            all_logits['l_logits'].append(l_logits)
            
            all_deltas['w_delta'].append(size_deltas[:, 0])
            all_deltas['h_delta'].append(size_deltas[:, 1])
            all_deltas['l_delta'].append(size_deltas[:, 2])
            
            all_continuous['w_continuous'].append(w_continuous)
            all_continuous['h_continuous'].append(h_continuous)
            all_continuous['l_continuous'].append(l_continuous)
            
            # 旋转属性 (roll, pitch, yaw) - rot_logits是[B, sum(num_bins)]形状，需要按维度分割
            rot_num_bins = model.num_discrete_rotation
            roll_logits = rot_logits[:, 0:rot_num_bins]  # [B, num_bins]
            pitch_logits = rot_logits[:, rot_num_bins:2*rot_num_bins]  # [B, num_bins]
            yaw_logits = rot_logits[:, 2*rot_num_bins:3*rot_num_bins]  # [B, num_bins]
            
            all_logits['roll_logits'].append(roll_logits)
            all_logits['pitch_logits'].append(pitch_logits)
            all_logits['yaw_logits'].append(yaw_logits)
            
            all_deltas['roll_delta'].append(rot_deltas[:, 0])
            all_deltas['pitch_delta'].append(rot_deltas[:, 1])
            all_deltas['yaw_delta'].append(rot_deltas[:, 2])
            
            all_continuous['roll_continuous'].append(roll_continuous)
            all_continuous['pitch_continuous'].append(pitch_continuous)
            all_continuous['yaw_continuous'].append(yaw_continuous)
            
            all_eos_logits.append(eos_logits)
            
            # 构建下一步的输入：当前token + 预测的连续值
            # 现在我们有3D向量的embeds，直接组合（顺序：位置+旋转+尺寸）
            combined_embeds = pos_embeds + rot_embeds + size_embeds
            combined_embed = torch.cat(combined_embeds, dim=-1)  # [B, total_embed_dim]
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
        
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
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
        total_eos_loss = 0.0
        total_mean_iou = 0.0
        total_adaptive_cls_weight = 0.0
        total_adaptive_delta_weight = 0.0
        
        # 🔍 初始化旋转角度损失统计
        total_roll_cls_loss = 0.0
        total_pitch_cls_loss = 0.0
        total_yaw_cls_loss = 0.0
        total_roll_delta_loss = 0.0
        total_pitch_delta_loss = 0.0
        total_yaw_delta_loss = 0.0
        
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # 将batch移动到正确的设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast(enabled=self.use_amp):
                # 前向传播
                outputs = self._forward_with_sampling_strategy(batch, teacher_forcing_ratio)
                
                # 准备目标数据 - 使用等价表示优化
                targets = self._prepare_targets_with_equivalent_boxes(batch, outputs)
                
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
            total_eos_loss += loss_dict.get('eos_loss', torch.tensor(0.0)).item()
            total_mean_iou += loss_dict['mean_iou'].item()
            total_adaptive_cls_weight += loss_dict.get('adaptive_classification_weight', torch.tensor(0.0)).item()
            total_adaptive_delta_weight += loss_dict.get('adaptive_delta_weight', torch.tensor(0.0)).item()
            
            # 🔍 添加旋转角度损失的单独统计
            total_roll_cls_loss += loss_dict.get('roll_cls', torch.tensor(0.0)).item()
            total_pitch_cls_loss += loss_dict.get('pitch_cls', torch.tensor(0.0)).item()
            total_yaw_cls_loss += loss_dict.get('yaw_cls', torch.tensor(0.0)).item()
            total_roll_delta_loss += loss_dict.get('roll_delta', torch.tensor(0.0)).item()
            total_pitch_delta_loss += loss_dict.get('pitch_delta', torch.tensor(0.0)).item()
            total_yaw_delta_loss += loss_dict.get('yaw_delta', torch.tensor(0.0)).item()
            
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
            train_eos_loss=total_eos_loss / num_batches,
            train_mean_iou=total_mean_iou / num_batches,
            val_loss=0.0,  # 稍后填充
            # 🔍 添加旋转角度损失统计
            train_roll_cls_loss=total_roll_cls_loss / num_batches,
            train_pitch_cls_loss=total_pitch_cls_loss / num_batches,
            train_yaw_cls_loss=total_yaw_cls_loss / num_batches,
            train_roll_delta_loss=total_roll_delta_loss / num_batches,
            train_pitch_delta_loss=total_pitch_delta_loss / num_batches,
            train_yaw_delta_loss=total_yaw_delta_loss / num_batches,
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
        total_roll_error = 0.0
        total_pitch_error = 0.0
        total_yaw_error = 0.0
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
                    'roll': batch['roll'].to(self.device),
                    'pitch': batch['pitch'].to(self.device),
                    'yaw': batch['yaw'].to(self.device),
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
                        temperature=self.incremental_temperature,
                        eos_threshold=self.eos_threshold
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
                gen_metrics = self._compute_generation_metrics(gen_results, targets, loss_fn, verbose=False, equivalent_boxes=batch.get('equivalent_boxes'))
                
                # 删除这行重复的累积
                # total_tf_loss += tf_loss_dict['total_loss'].item()
                
                total_tf_iou += eval_iou  # 使用评估IoU而不是损失IoU
                total_gen_iou += gen_metrics['iou']
                total_generated_boxes += gen_metrics['num_generated_boxes']
                total_gt_boxes += gen_metrics['num_gt_boxes']
                
                # 累积维度误差
                total_x_error += gen_metrics.get('x_error', 0.0)
                total_y_error += gen_metrics.get('y_error', 0.0)
                total_z_error += gen_metrics.get('z_error', 0.0)
                total_w_error += gen_metrics.get('w_error', 0.0)
                total_h_error += gen_metrics.get('h_error', 0.0)
                total_l_error += gen_metrics.get('l_error', 0.0)
                total_roll_error += gen_metrics.get('roll_error', 0.0)
                total_pitch_error += gen_metrics.get('pitch_error', 0.0)
                total_yaw_error += gen_metrics.get('yaw_error', 0.0)
                total_overall_error += gen_metrics.get('overall_mean_error', 0.0)
                
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
        avg_roll_error = total_roll_error / num_batches if num_batches > 0 else 0.0
        avg_pitch_error = total_pitch_error / num_batches if num_batches > 0 else 0.0
        avg_yaw_error = total_yaw_error / num_batches if num_batches > 0 else 0.0
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
            'avg_roll_error': avg_roll_error,
            'avg_pitch_error': avg_pitch_error,
            'avg_yaw_error': avg_yaw_error,
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
                pred_rotations = []
                gt_boxes = []
                gt_rotations = []
                
                # 获取每个位置的预测值
                seq_len = targets['x'].size(1)
                for s in range(seq_len):
                    # 检查是否为有效位置（非padding）
                    if targets['x'][b, s].item() != -1.0:
                        # 将logits和delta转换为连续预测值
                        pred_box = []
                        pred_rot = []
                        
                        # 位置和尺寸预测
                        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
                            if attr + '_logits' in outputs['logits_dict'] and attr + '_delta' in outputs['delta_dict']:
                                logits = outputs['logits_dict'][attr + '_logits'][b, s]  # [num_bins]
                                delta = outputs['delta_dict'][attr + '_delta'][b, s]     # scalar
                                continuous_val = self._get_continuous_prediction(logits, delta, attr)
                                pred_box.append(continuous_val)
                                
                        
                        # 旋转预测
                        for attr in ['roll', 'pitch', 'yaw']:
                            if attr + '_logits' in outputs['logits_dict'] and attr + '_delta' in outputs['delta_dict']:
                                logits = outputs['logits_dict'][attr + '_logits'][b, s]  # [num_bins]
                                delta = outputs['delta_dict'][attr + '_delta'][b, s]     # scalar
                                continuous_val = self._get_continuous_prediction(logits, delta, attr)
                                pred_rot.append(continuous_val)
                        
                        if len(pred_box) == 6 and len(pred_rot) == 3:
                            pred_boxes.append(pred_box)
                            pred_rotations.append(pred_rot)
                            
                            # 对应的GT box
                            gt_box = [
                                targets['x'][b, s].cpu().item(),
                                targets['y'][b, s].cpu().item(),
                                targets['z'][b, s].cpu().item(),
                                targets['l'][b, s].cpu().item(),
                                targets['w'][b, s].cpu().item(),
                                targets['h'][b, s].cpu().item(),
                            ]
                            gt_boxes.append(gt_box)
                            
                            # GT旋转（如果可用）
                            if 'roll' in targets and 'pitch' in targets and 'yaw' in targets:
                                gt_rot = [
                                    targets['roll'][b, s].cpu().item(),
                                    targets['pitch'][b, s].cpu().item(),
                                    targets['yaw'][b, s].cpu().item(),
                                ]  # [3] euler angles
                                gt_rotations.append(gt_rot)
                            else:
                                # 使用零旋转（无旋转）
                                gt_rotations.append([0.0, 0.0, 0.0])
                
                # 计算该样本的IoU
                if pred_boxes and gt_boxes:
                    sample_ious = []
                    
                    # 计算每个预测box与对应GT box的IoU
                    for i, (pred_box, pred_rot, gt_box, gt_rot) in enumerate(zip(pred_boxes, pred_rotations, gt_boxes, gt_rotations)):
                        try:
                            # 检查box尺寸
                            pred_size = np.array(pred_box[3:])  # [l, w, h]
                            gt_size = np.array(gt_box[3:])     # [l, w, h]
                            
                            if np.any(pred_size <= 0) or np.any(gt_size <= 0):
                                continue
                            
                            # 使用OBB IoU计算
                            iou = self._compute_box_iou(pred_box, gt_box, pred_rot, gt_rot)
                            sample_ious.append(iou)
                            
                        except Exception as e:
                            sample_ious.append(0.0)
                    
                    if sample_ious:
                        sample_mean_iou = sum(sample_ious) / len(sample_ious)
                        total_iou += sample_mean_iou
                        valid_samples += 1
            
            if valid_samples == 0:
                return 0.0
            
            mean_iou = total_iou / valid_samples
            return float(mean_iou)
            
        except Exception as e:
            if verbose:
                print(f"⚠️  计算TF评估IoU时出错: {e}")
            return 0.0
    
    def _get_continuous_prediction(self, logits: torch.Tensor, delta: torch.Tensor, attr: str) -> float:
        """将分类logits和delta组合成连续预测值"""
        # 获取属性的配置（从ConfigLoader返回的平铺结构中获取）
        attr_configs = {
            # 位置属性
            'x': (self.model_config.get('num_discrete_position', 64), 
                  self.model_config.get('continuous_range_position', [[0.5, 2.5], [-2.0, 2.0], [-1.5, 1.5]])[0]),
            'y': (self.model_config.get('num_discrete_position', 64), 
                  self.model_config.get('continuous_range_position', [[0.5, 2.5], [-2.0, 2.0], [-1.5, 1.5]])[1]),
            'z': (self.model_config.get('num_discrete_position', 64), 
                  self.model_config.get('continuous_range_position', [[0.5, 2.5], [-2.0, 2.0], [-1.5, 1.5]])[2]),
            # 旋转属性
            'roll': (self.model_config.get('num_discrete_rotation', 64), 
                     self.model_config.get('continuous_range_rotation', [[-1.5708, 1.5708], [-1.5708, 1.5708], [-1.5708, 1.5708]])[0]),
            'pitch': (self.model_config.get('num_discrete_rotation', 64), 
                      self.model_config.get('continuous_range_rotation', [[-1.5708, 1.5708], [-1.5708, 1.5708], [-1.5708, 1.5708]])[1]),
            'yaw': (self.model_config.get('num_discrete_rotation', 64), 
                    self.model_config.get('continuous_range_rotation', [[-1.5708, 1.5708], [-1.5708, 1.5708], [-1.5708, 1.5708]])[2]),
            # 尺寸属性
            'w': (self.model_config.get('num_discrete_size', 64), 
                  self.model_config.get('continuous_range_size', [[0.1, 1.0], [0.1, 1.0], [0.1, 1.0]])[0]),
            'h': (self.model_config.get('num_discrete_size', 64), 
                  self.model_config.get('continuous_range_size', [[0.1, 1.0], [0.1, 1.0], [0.1, 1.0]])[1]),
            'l': (self.model_config.get('num_discrete_size', 64), 
                  self.model_config.get('continuous_range_size', [[0.1, 1.0], [0.1, 1.0], [0.1, 1.0]])[2])
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

    def _compute_generation_metrics(self, gen_results: Dict, targets: Dict, loss_fn, verbose: bool = False, equivalent_boxes: List = None) -> Dict[str, float]:
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
            
            for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
                if attr in gen_results:
                    # 获取生成结果（已经是连续值）
                    gen_values = gen_results[attr]  # [B, seq_len]
                    
                    # 对齐序列长度
                    if gen_values.shape[1] > target_seq_len:
                        gen_values = gen_values[:, :target_seq_len]
                    elif gen_values.shape[1] < target_seq_len:
                        # 不填充，保持原始长度，让IoU计算自己处理长度不匹配
                        pass
                    
                    processed_gen_results[attr] = gen_values
                else:
                    # 如果某个属性缺失，跳过该属性，不进行IoU计算
                    # print(f"⚠️  生成结果中缺少属性 {attr}，跳过该属性的IoU计算")
                    continue
            
            # 计算IoU
            gen_iou = self._compute_generation_iou(processed_gen_results, targets, verbose, equivalent_boxes)
            
            # 计算9个维度的平均误差（包括旋转角度）
            dimension_errors = {}
            total_valid_predictions = 0
            
            # 🔧 新增：如果有等效box信息，先选择最优等效box
            if equivalent_boxes is not None:
                # 为每个batch和sequence位置选择最优等效box
                batch_size = targets['x'].shape[0]
                seq_len = targets['x'].shape[1]
                
                # 创建最优等效box的目标值
                optimal_targets = {}
                for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
                    optimal_targets[attr] = targets[attr].clone()
                
                for b in range(batch_size):
                    for s in range(seq_len):
                        # 检查是否为有效位置
                        if targets['x'][b, s].item() != -1.0:
                            # 🔧 检查生成结果序列长度是否足够（增量生成可能提前停止）
                            gen_seq_len = processed_gen_results['x'].shape[1]
                            if s >= gen_seq_len:
                                # 增量生成提前停止，跳过该位置（这是正常行为）
                                continue
                            
                            # 🔧 检查等效box是否存在且索引不越界
                            if b < len(equivalent_boxes) and s < len(equivalent_boxes[b]) and len(equivalent_boxes[b][s]) > 0:
                                # 构建预测box
                                pred_box = [
                                    processed_gen_results['x'][b, s].item(),
                                    processed_gen_results['y'][b, s].item(),
                                    processed_gen_results['z'][b, s].item(),
                                    processed_gen_results['l'][b, s].item(),
                                    processed_gen_results['w'][b, s].item(),
                                    processed_gen_results['h'][b, s].item(),
                                    processed_gen_results['roll'][b, s].item(),
                                    processed_gen_results['pitch'][b, s].item(),
                                    processed_gen_results['yaw'][b, s].item(),
                                ]
                                
                                # 选择最优等效box
                                equiv_boxes = equivalent_boxes[b][s]
                                best_box, min_loss = select_best_equivalent_representation(pred_box, equiv_boxes)
                            else:
                                # 如果等效box不存在或索引越界，跳过该位置
                                if b >= len(equivalent_boxes):
                                    # print(f"⚠️  等效box batch索引越界: batch={b}, equiv_boxes_len={len(equivalent_boxes)}")
                                    pass
                                elif s >= len(equivalent_boxes[b]):
                                    # print(f"⚠️  等效box序列索引越界: batch={b}, seq={s}, equiv_boxes_len={len(equivalent_boxes[b])}")
                                    pass
                                continue
                            
                            # 🔍 添加详细log：打印旋转误差计算过程（已注释）
                            # print(f"🔍 旋转误差计算 - Batch {b}, Box {s}:")
                            # print(f"   预测box: pos=({pred_box[0]:.3f}, {pred_box[1]:.3f}, {pred_box[2]:.3f}), "
                            #       f"size=({pred_box[3]:.3f}, {pred_box[4]:.3f}, {pred_box[5]:.3f}), "
                            #       f"rot=({pred_box[6]:.3f}, {pred_box[7]:.3f}, {pred_box[8]:.3f})")
                            # print(f"   预测角度(度): roll={math.degrees(pred_box[6]):.1f}°, "
                            #       f"pitch={math.degrees(pred_box[7]):.1f}°, "
                            #       f"yaw={math.degrees(pred_box[8]):.1f}°")
                            
                            # print(f"   GT等效box数量: {len(equiv_boxes)}")
                            # for i, equiv_box in enumerate(equiv_boxes):
                            #     print(f"     等效box {i+1}: pos=({equiv_box[0]:.3f}, {equiv_box[1]:.3f}, {equiv_box[2]:.3f}), "
                            #           f"size=({equiv_box[3]:.3f}, {equiv_box[4]:.3f}, {equiv_box[5]:.3f}), "
                            #           f"rot=({equiv_box[6]:.3f}, {equiv_box[7]:.3f}, {equiv_box[8]:.3f})")
                            #     print(f"     等效box {i+1}角度(度): roll={math.degrees(equiv_box[6]):.1f}°, "
                            #           f"pitch={math.degrees(equiv_box[7]):.1f}°, "
                            #           f"yaw={math.degrees(equiv_box[8]):.1f}°")
                            
                            # print(f"   选择的最优等效box: pos=({best_box[0]:.3f}, {best_box[1]:.3f}, {best_box[2]:.3f}), "
                            #       f"size=({best_box[3]:.3f}, {best_box[4]:.3f}, {best_box[5]:.3f}), "
                            #       f"rot=({best_box[6]:.3f}, {best_box[7]:.3f}, {best_box[8]:.3f})")
                            # print(f"   最优等效box角度(度): roll={math.degrees(best_box[6]):.1f}°, "
                            #       f"pitch={math.degrees(best_box[7]):.1f}°, "
                            #       f"yaw={math.degrees(best_box[8]):.1f}°")
                            # print(f"   最小旋转loss: {min_loss:.6f}")
                            
                            # 更新目标值
                            optimal_targets['x'][b, s] = best_box[0]
                            optimal_targets['y'][b, s] = best_box[1]
                            optimal_targets['z'][b, s] = best_box[2]
                            optimal_targets['l'][b, s] = best_box[3]
                            optimal_targets['w'][b, s] = best_box[4]
                            optimal_targets['h'][b, s] = best_box[5]
                            optimal_targets['roll'][b, s] = best_box[6]
                            optimal_targets['pitch'][b, s] = best_box[7]
                            optimal_targets['yaw'][b, s] = best_box[8]
                
                # 使用最优等效box作为目标值
                targets_to_use = optimal_targets
            else:
                # 没有等效box信息，使用原始目标值
                targets_to_use = targets
            
            for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
                if attr in processed_gen_results and attr in targets_to_use:
                    # 获取生成结果和目标值
                    gen_values = processed_gen_results[attr]  # [B, seq_len]
                    gt_values = targets_to_use[attr]         # [B, seq_len]
                    
                    # 创建有效mask（排除padding值）
                    valid_mask = (gt_values != -1.0) & (gt_values != 0.0)  # GT非padding且非零
                    
                    if valid_mask.sum() > 0:
                        # 计算有效位置的绝对误差
                        # 确保两个张量形状匹配
                        min_len = min(gen_values.shape[1], gt_values.shape[1])
                        gen_values_aligned = gen_values[:, :min_len]
                        gt_values_aligned = gt_values[:, :min_len]
                        valid_mask_aligned = valid_mask[:, :min_len]
                        
                        # 🔧 修复：对于旋转角度，使用周期性误差计算
                        if attr in ['roll', 'pitch', 'yaw']:
                            # 计算角度差值并归一化到[-π, π]
                            angle_diff = gen_values_aligned - gt_values_aligned
                            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
                            abs_errors = torch.abs(angle_diff)
                        else:
                            # 对于位置和尺寸，使用普通绝对误差
                            abs_errors = torch.abs(gen_values_aligned - gt_values_aligned)
                        
                        valid_errors = abs_errors[valid_mask_aligned]
                        
                        # 计算平均误差
                        mean_error = valid_errors.mean().item() if len(valid_errors) > 0 else 0.0
                        
                        # 🔧 修复：对于旋转角度，将弧度转换为角度制记录到SwanLab
                        if attr in ['roll', 'pitch', 'yaw']:
                            dimension_errors[f'{attr}_error'] = mean_error * 180.0 / math.pi  # 弧度转角度
                        else:
                            dimension_errors[f'{attr}_error'] = mean_error
                        total_valid_predictions += valid_mask_aligned.sum().item()
                        
                        # 🔍 添加调试日志：打印旋转角度的误差信息
                        # if attr in ['roll', 'pitch', 'yaw'] and verbose:
                        #     print(f"🔍 {attr}角度误差计算:")
                        #     print(f"   有效预测数量: {valid_mask_aligned.sum().item()}")
                        #     print(f"   平均误差: {mean_error:.6f}")
                        #     print(f"   生成值范围: [{gen_values_aligned.min().item():.6f}, {gen_values_aligned.max().item():.6f}]")
                        #     print(f"   GT值范围: [{gt_values_aligned.min().item():.6f}, {gt_values_aligned.max().item():.6f}]")
                    else:
                        dimension_errors[f'{attr}_error'] = 0.0
                        # if attr in ['roll', 'pitch', 'yaw'] and verbose:
                        #     print(f"⚠️  {attr}角度没有有效预测（全为padding）")
                else:
                    dimension_errors[f'{attr}_error'] = 0.0
                    # if attr in ['roll', 'pitch', 'yaw'] and verbose:
                    #     print(f"⚠️  {attr}角度缺失在生成结果或目标中")
            
            # 计算总体平均误差
            if total_valid_predictions > 0:
                overall_mean_error = sum(dimension_errors.values()) / 9.0  # 9个维度
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
                'l_error': dimension_errors['l_error'],
                'roll_error': dimension_errors['roll_error'],
                'pitch_error': dimension_errors['pitch_error'],
                'yaw_error': dimension_errors['yaw_error']
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
                'l_error': 1.0,
                'roll_error': 1.0,
                'pitch_error': 1.0,
                'yaw_error': 1.0
            }
    
    def _compute_generation_iou(self, gen_results: Dict, targets: Dict, verbose: bool = False, equivalent_boxes: List = None) -> float:
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
                pred_rotations = []
                gt_boxes = []
                gt_rotations = []
                
                # 获取序列长度
                seq_len = targets['x'].size(1)
                gen_len = gen_results['x'].shape[1]
                
                # 一对一匹配：只计算同时存在的位置
                for s in range(seq_len):
                    # 检查GT是否为有效位置（非padding）且生成结果中也存在该位置
                    if targets['x'][b, s].item() != -1.0 and s < gen_len:
                        # 对应的预测box - 修复访问格式
                        pred_box = [
                            float(gen_results['x'][b, s]),
                            float(gen_results['y'][b, s]),
                            float(gen_results['z'][b, s]),
                            float(gen_results['l'][b, s]),
                            float(gen_results['w'][b, s]),
                            float(gen_results['h'][b, s]),
                        ]
                        
                        # 🔧 修复：检查预测box是否为padding
                        pred_pos = np.array(pred_box[:3])  # [x, y, z]
                        pred_size = np.array(pred_box[3:])  # [l, w, h]
                        
                        # 检查是否为padding值（-1.0）或全零
                        is_padding_value = np.allclose(pred_pos, -1.0, atol=1e-6) or np.allclose(pred_pos, 0.0, atol=1e-6)
                        is_zero_size = np.allclose(pred_size, 0.0, atol=1e-6) or np.allclose(pred_size, -1.0, atol=1e-6)
                        
                        if is_padding_value or is_zero_size:
                            continue
                        
                        # GT box
                        gt_box = [
                            targets['x'][b, s].cpu().item(),
                            targets['y'][b, s].cpu().item(),
                            targets['z'][b, s].cpu().item(),
                            targets['l'][b, s].cpu().item(),
                            targets['w'][b, s].cpu().item(),
                            targets['h'][b, s].cpu().item(),
                        ]
                        gt_boxes.append(gt_box)
                        pred_boxes.append(pred_box)
                        
                        # GT旋转信息
                        if 'roll' in targets and 'pitch' in targets and 'yaw' in targets:
                            gt_rot = [
                                targets['roll'][b, s].cpu().item(),
                                targets['pitch'][b, s].cpu().item(),
                                targets['yaw'][b, s].cpu().item(),
                            ]  # [3] euler angles
                            gt_rotations.append(gt_rot)
                        else:
                            gt_rotations.append([0.0, 0.0, 0.0])  # 零旋转
                        
                        # 预测旋转信息
                        if 'roll' in gen_results and 'pitch' in gen_results and 'yaw' in gen_results:
                            pred_rot = [
                                float(gen_results['roll'][b, s]),
                                float(gen_results['pitch'][b, s]),
                                float(gen_results['yaw'][b, s]),
                            ]  # [3] euler angles
                            pred_rotations.append(pred_rot)
                        else:
                            pred_rotations.append([0.0, 0.0, 0.0])  # 零旋转
                
                # 🔍 添加调试信息：检查生成结果长度
                # print(f"🔍 生成IoU调试 - Batch {b}:")
                # print(f"   目标序列长度: {seq_len}")
                # print(f"   生成序列长度: {gen_len}")
                # print(f"   预测box数量: {len(pred_boxes)}")
                # print(f"   GT box数量: {len(gt_boxes)}")
                
                # 计算该样本的IoU（一对一匹配）
                if pred_boxes and gt_boxes:
                    sample_ious = []
                    
                    # 计算每个预测box与对应GT box的IoU
                    for i, (pred_box, pred_rot, gt_box, gt_rot) in enumerate(zip(pred_boxes, pred_rotations, gt_boxes, gt_rotations)):
                        try:
                            # 🔍 添加详细日志：检查box尺寸
                            pred_size = np.array(pred_box[3:])  # [l, w, h]
                            gt_size = np.array(gt_box[3:])     # [l, w, h]
                            
                            if np.any(pred_size <= 0) or np.any(gt_size <= 0):
                                # print(f"🚨 检测到尺寸为0的box - Batch {b}, Box {i}:")
                                # print(f"   预测box: pos={pred_box[:3]}, size={pred_size} (l={pred_size[0]:.6f}, w={pred_size[1]:.6f}, h={pred_size[2]:.6f})")
                                # print(f"   GT box:   pos={gt_box[:3]}, size={gt_size} (l={gt_size[0]:.6f}, w={gt_size[1]:.6f}, h={gt_size[2]:.6f})")
                                # print(f"   预测旋转: {pred_rot}")
                                # print(f"   GT旋转:   {gt_rot}")
                                pass
                            
                            # 使用OBB IoU计算
                            iou = self._compute_box_iou(pred_box, gt_box, pred_rot, gt_rot)
                            sample_ious.append(iou)
                            
                            # 🔍 详细调试信息：打印每个box的详细信息
                            # print(f"   Box {i}:")
                            # print(f"     预测box: pos=({pred_box[0]:.3f}, {pred_box[1]:.3f}, {pred_box[2]:.3f}), size=({pred_box[3]:.3f}, {pred_box[4]:.3f}, {pred_box[5]:.3f})")
                            # print(f"     GT box:   pos=({gt_box[0]:.3f}, {gt_box[1]:.3f}, {gt_box[2]:.3f}), size=({gt_box[3]:.3f}, {gt_box[4]:.3f}, {gt_box[5]:.3f})")
                            # print(f"     预测旋转: roll={pred_rot[0]:.3f}, pitch={pred_rot[1]:.3f}, yaw={pred_rot[2]:.3f}")
                            # print(f"     GT旋转:   roll={gt_rot[0]:.3f}, pitch={gt_rot[1]:.3f}, yaw={gt_rot[2]:.3f}")
                            
                            # 🔍 如果有等效box信息，显示最优等效box
                            if equivalent_boxes is not None and b < len(equivalent_boxes) and i < len(equivalent_boxes[b]):
                                try:
                                    # 构建预测box用于等效box选择
                                    pred_box_for_equiv = [
                                        pred_box[0], pred_box[1], pred_box[2],  # pos
                                        pred_box[3], pred_box[4], pred_box[5],  # size
                                        pred_rot[0], pred_rot[1], pred_rot[2]   # rot
                                    ]
                                    
                                    # 选择最优等效box
                                    equiv_boxes = equivalent_boxes[b][i]
                                    if len(equiv_boxes) > 0:
                                        best_box, min_loss = select_best_equivalent_representation(pred_box_for_equiv, equiv_boxes)
                                        
                                        # 🔍 调试：检查best_box的结构
                                        # print(f"     🔍 best_box结构调试:")
                                        # print(f"       best_box类型: {type(best_box)}")
                                        # print(f"       best_box长度: {len(best_box)}")
                                        # print(f"       best_box内容: {best_box}")
                                        
                                        # print(f"     最优等效box: pos=({best_box[0]:.3f}, {best_box[1]:.3f}, {best_box[2]:.3f}), size=({best_box[3]:.3f}, {best_box[4]:.3f}, {best_box[5]:.3f})")
                                        # print(f"     最优等效旋转: roll={best_box[6]:.3f}, pitch={best_box[7]:.3f}, yaw={best_box[8]:.3f}")
                                        # print(f"     原始等效box数量: {len(equiv_boxes)}, 筛选后等效box数量: {len([eq for eq in equiv_boxes if abs(eq[6]) <= math.pi/3 and abs(eq[7]) <= math.pi/3 and abs(eq[8]) <= math.pi/3])}, 最小loss: {min_loss:.6f}")
                                        
                                        # 🔍 调试：检查pred_box和best_box的结构
                                        # print(f"     🔍 误差计算调试:")
                                        # print(f"       pred_box类型: {type(pred_box)}, 长度: {len(pred_box)}")
                                        # print(f"       pred_box内容: {pred_box}")
                                        # print(f"       best_box类型: {type(best_box)}, 长度: {len(best_box)}")
                                        # print(f"       best_box内容: {best_box}")
                                        
                                        # 🔧 修复：pred_box只有6个元素，需要正确切片
                                        pred_pos_array = np.array(pred_box[:3])
                                        best_pos_array = np.array(best_box[:3])
                                        # print(f"       pred_pos_array形状: {pred_pos_array.shape}")
                                        # print(f"       best_pos_array形状: {best_pos_array.shape}")
                                        
                                        equiv_pos_error = np.sqrt(sum((pred_pos_array - best_pos_array)**2))
                                        
                                        # pred_box只有6个元素，best_box有9个元素
                                        pred_size_array = np.array(pred_box[3:])  # [l, w, h]
                                        best_size_array = np.array(best_box[3:6])  # [l, w, h] - 只取尺寸部分
                                        # print(f"       pred_size_array形状: {pred_size_array.shape}")
                                        # print(f"       best_size_array形状: {best_size_array.shape}")
                                        
                                        equiv_size_error = np.sqrt(sum((pred_size_array - best_size_array)**2))
                                        
                                        # 🔍 调试：检查旋转部分的形状
                                        # print(f"     🔍 旋转部分调试:")
                                        # print(f"       pred_rot类型: {type(pred_rot)}, 长度: {len(pred_rot)}")
                                        # print(f"       pred_rot内容: {pred_rot}")
                                        best_box_rot_slice = best_box[6:9]
                                        # print(f"       best_box[6:9]类型: {type(best_box_rot_slice)}, 长度: {len(best_box_rot_slice)}")
                                        # print(f"       best_box[6:9]内容: {best_box_rot_slice}")
                                        
                                        # 🔧 修复：确保旋转部分是正确的格式
                                        pred_rot_array = np.array(pred_rot)
                                        best_rot_array = np.array([best_box[6], best_box[7], best_box[8]])
                                        
                                        # print(f"       pred_rot_array形状: {pred_rot_array.shape}")
                                        # print(f"       best_rot_array形状: {best_rot_array.shape}")
                                        
                                        equiv_rot_error = np.sqrt(sum((pred_rot_array - best_rot_array)**2))
                                        
                                        # print(f"     与最优等效box误差: pos={equiv_pos_error:.3f}, size={equiv_size_error:.3f}, rot={equiv_rot_error:.3f}")
                                except Exception as e:
                                    import traceback
                                    # print(f"     ⚠️ 等效box调试出错: {e}")
                                    # print(f"     详细错误信息:")
                                    # traceback.print_exc()
                                    pass
                            
                            # 计算各维度误差
                            pos_error = np.sqrt(sum((np.array(pred_box[:3]) - np.array(gt_box[:3]))**2))
                            size_error = np.sqrt(sum((np.array(pred_box[3:]) - np.array(gt_box[3:]))**2))
                            rot_error = np.sqrt(sum((np.array(pred_rot) - np.array(gt_rot))**2))
                            
                            # print(f"     位置误差: {pos_error:.3f}")
                            # print(f"     尺寸误差: {size_error:.3f}")
                            # print(f"     旋转误差: {rot_error:.3f}")
                            # print(f"     IoU: {iou:.4f}")
                            
                            # 如果IoU异常高，额外检查
                            # if iou > 0.5:
                            #     print(f"     🚨 异常高IoU警告!")
                            #     print(f"       尺寸差异: l={abs(pred_box[3]-gt_box[3]):.3f}, w={abs(pred_box[4]-gt_box[4]):.3f}, h={abs(pred_box[5]-gt_box[5]):.3f}")
                            #     print(f"       位置差异: x={abs(pred_box[0]-gt_box[0]):.3f}, y={abs(pred_box[1]-gt_box[1]):.3f}, z={abs(pred_box[2]-gt_box[2]):.3f}")
                            
                        except Exception as e:
                            # print(f"⚠️  计算box IoU时出错: {e}")
                            sample_ious.append(0.0)
                    
                    if sample_ious:
                        sample_mean_iou = sum(sample_ious) / len(sample_ious)
                        total_iou += sample_mean_iou
                        valid_samples += 1
                        # print(f"   样本 {b} 平均IoU: {sample_mean_iou:.4f} (有效IoU数量: {len(sample_ious)})")
                    else:
                        # print(f"   样本 {b} 没有有效的IoU计算")
                        pass
            
            if valid_samples == 0:
                return 0.0
            
            mean_iou = total_iou / valid_samples
            # print(f"\n🎯 生成Overall Mean IoU: {mean_iou:.4f} (from {valid_samples} samples)")
            return float(mean_iou)
            
        except Exception as e:
            if verbose:
                # print(f"计算生成IoU时出错: {e}")
                pass
            return 0.0
    
    def _compute_box_iou(self, box1: List[float], box2: List[float], rot1: List[float] = None, rot2: List[float] = None) -> float:
        """
        计算两个3D box的IoU，使用OBB（有向包围盒）计算
        Args:
            box1: [x, y, z, l, w, h] - 第一个box的中心坐标和尺寸
            box2: [x, y, z, l, w, h] - 第二个box的中心坐标和尺寸 
            rot1: [roll, pitch, yaw] - 第一个box的欧拉角旋转（弧度）
            rot2: [roll, pitch, yaw] - 第二个box的欧拉角旋转（弧度）
        Returns:
            IoU值 (0.0-1.0)
        """
        # 检查旋转信息是否完整
        if rot1 is None or rot2 is None:
            raise ValueError("旋转信息缺失，无法计算OBB IoU")
        
        # 使用OBB IoU计算
        return self._compute_obb_iou(box1, box2, rot1, rot2)
    
    
    def _compute_obb_iou(self, box1: List[float], box2: List[float], rot1: List[float], rot2: List[float]) -> float:
        """
        计算两个有向包围盒(OBB)的IoU
        Args:
            box1: [x, y, z, l, w, h] - 第一个box的中心坐标和尺寸
            box2: [x, y, z, l, w, h] - 第二个box的中心坐标和尺寸 
            rot1: [roll, pitch, yaw] - 第一个box的欧拉角旋转（弧度）
            rot2: [roll, pitch, yaw] - 第二个box的欧拉角旋转（弧度）
        Returns:
            IoU值 (0.0-1.0)
        """
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation
            
            # 确保输入格式正确
            if len(box1) != 6 or len(box2) != 6 or len(rot1) != 3 or len(rot2) != 3:
                # print(f"⚠️  OBB输入格式错误: box1={len(box1)}, box2={len(box2)}, rot1={len(rot1)}, rot2={len(rot2)}")
                return 0.0
            
            # 检查输入参数的有效性
            if np.any(np.isnan(box1)) or np.any(np.isinf(box1)):
                # print(f"⚠️  Box1包含无效值: box1={box1}")
                return 0.0
            if np.any(np.isnan(box2)) or np.any(np.isinf(box2)):
                # print(f"⚠️  Box2包含无效值: box2={box2}")
                return 0.0
            if np.any(np.isnan(rot1)) or np.any(np.isinf(rot1)):
                # print(f"⚠️  Rot1包含无效值: rot1={rot1}")
                return 0.0
            if np.any(np.isnan(rot2)) or np.any(np.isinf(rot2)):
                # print(f"⚠️  Rot2包含无效值: rot2={rot2}")
                return 0.0
            
            # 提取box参数
            center1 = np.array(box1[:3])  # [x, y, z]
            size1 = np.array(box1[3:])    # [l, w, h]
            center2 = np.array(box2[:3])  # [x, y, z]
            size2 = np.array(box2[3:])    # [l, w, h]
            
            # 检查尺寸是否有效（避免尺寸为0的box）
            if np.any(size1 <= 0) or np.any(size2 <= 0):
                # print(f"🚨 在OBB IoU计算中检测到无效尺寸的box:")
                # print(f"  Box1: center={center1}, size={size1} (l={size1[0]:.6f}, w={size1[1]:.6f}, h={size1[2]:.6f})")
                # print(f"  Box2: center={center2}, size={size2} (l={size2[0]:.6f}, w={size2[1]:.6f}, h={size2[2]:.6f})")
                # print(f"  Rot1: {rot1}")
                # print(f"  Rot2: {rot2}")
                # print(f"  返回IoU=0.0")
                return 0.0
            
            # 创建旋转矩阵
            rot_matrix1 = Rotation.from_euler('xyz', rot1).as_matrix()
            rot_matrix2 = Rotation.from_euler('xyz', rot2).as_matrix()
            
            # 计算OBB的8个顶点
            def get_obb_vertices(center, size, rot_matrix):
                # 局部坐标系的8个顶点
                half_size = size / 2
                vertices_local = np.array([
                    [-half_size[0], -half_size[1], -half_size[2]],
                    [ half_size[0], -half_size[1], -half_size[2]],
                    [ half_size[0],  half_size[1], -half_size[2]],
                    [-half_size[0],  half_size[1], -half_size[2]],
                    [-half_size[0], -half_size[1],  half_size[2]],
                    [ half_size[0], -half_size[1],  half_size[2]],
                    [ half_size[0],  half_size[1],  half_size[2]],
                    [-half_size[0],  half_size[1],  half_size[2]]
                ])
                
                # 旋转并平移到世界坐标系
                vertices_world = vertices_local @ rot_matrix.T + center
                return vertices_world
            
            vertices1 = get_obb_vertices(center1, size1, rot_matrix1)
            vertices2 = get_obb_vertices(center2, size2, rot_matrix2)
            
            # 使用凸包计算交集体积
            from scipy.spatial import ConvexHull
            
            # 计算两个OBB的凸包
            try:
                # 验证顶点数据的有效性
                if not self._validate_vertices(vertices1):
                    # print(f"⚠️  Box1顶点数据无效:")
                    # print(f"  center1: {center1}")
                    # print(f"  size1: {size1}")
                    # print(f"  rot1: {rot1}")
                    # print(f"  vertices1 shape: {vertices1.shape}")
                    # print(f"  vertices1 sample: {vertices1[:3]}")
                    raise ValueError("Box1顶点数据无效，无法计算OBB IoU")
                
                if not self._validate_vertices(vertices2):
                    # print(f"⚠️  Box2顶点数据无效:")
                    # print(f"  center2: {center2}")
                    # print(f"  size2: {size2}")
                    # print(f"  rot2: {rot2}")
                    # print(f"  vertices2 shape: {vertices2.shape}")
                    # print(f"  vertices2 sample: {vertices2[:3]}")
                    raise ValueError("Box2顶点数据无效，无法计算OBB IoU")
                
                hull1 = ConvexHull(vertices1)
                hull2 = ConvexHull(vertices2)
                
                # 计算体积
                volume1 = hull1.volume
                volume2 = hull2.volume
                
                # 计算交集体积（简化方法：使用AABB近似）
                # 这里使用简化的方法，实际应用中可能需要更复杂的算法
                return self._compute_obb_intersection_volume(vertices1, vertices2, volume1, volume2)
                
            except Exception as e:
                raise RuntimeError(f"凸包计算出错: {e}")
                
        except Exception as e:
            raise RuntimeError(f"OBB IoU计算出错: {e}")
    
    def _validate_vertices(self, vertices):
        """
        验证顶点数据的有效性
        Args:
            vertices: [8, 3] 顶点坐标
        Returns:
            bool: 是否有效
        """
        try:
            # 检查是否有足够的点
            if len(vertices) < 4:
                return False
            
            # 检查是否有重复点
            unique_vertices = np.unique(vertices, axis=0)
            if len(unique_vertices) < 4:
                return False
            
            # 检查是否有NaN或Inf值
            if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
                return False
            
            # 检查点是否共面（简化检查）
            if len(unique_vertices) == 4:
                # 如果只有4个唯一点，检查是否共面
                vectors = unique_vertices[1:] - unique_vertices[0]
                if np.linalg.matrix_rank(vectors) < 3:
                    return False
            
            return True
            
        except Exception:
            return False
    
    
    def _compute_obb_intersection_volume(self, vertices1: np.ndarray, vertices2: np.ndarray, volume1: float, volume2: float) -> float:
        """
        使用trimesh计算两个OBB的真实交集体积
        使用manifold3d进行精确的布尔运算
        """
        try:
            import trimesh
            import numpy as np
            
            # 创建两个OBB的trimesh对象
            try:
                # 创建第一个OBB的mesh
                mesh1 = trimesh.convex.convex_hull(vertices1)
                # 创建第二个OBB的mesh  
                mesh2 = trimesh.convex.convex_hull(vertices2)
                
                # 使用trimesh的布尔运算计算交集
                intersection = mesh1.intersection(mesh2)
                
                if intersection is None or intersection.volume <= 0:
                    return 0.0
                
                intersection_volume = intersection.volume
                
                # 计算IoU
                union_volume = volume1 + volume2 - intersection_volume
                if union_volume <= 0:
                    return 0.0
                
                iou = intersection_volume / union_volume
                return max(0.0, min(1.0, iou))
                
            except Exception as e:
                print(f"⚠️  Trimesh布尔运算计算出错: {e}")
                # 回退到AABB方法
                return self._compute_aabb_iou(vertices1, vertices2, volume1, volume2)
            
        except Exception as e:
            print(f"⚠️  OBB交集体积计算出错: {e}")
            return 0.0
    
    
    
    def _compute_aabb_iou(self, vertices1: np.ndarray, vertices2: np.ndarray, volume1: float, volume2: float) -> float:
        """
        计算AABB IoU（备用方法）
        """
        try:
            import numpy as np
            
            # 计算AABB包围盒
            min1 = np.min(vertices1, axis=0)
            max1 = np.max(vertices1, axis=0)
            min2 = np.min(vertices2, axis=0)
            max2 = np.max(vertices2, axis=0)
            
            # 计算交集AABB
            inter_min = np.maximum(min1, min2)
            inter_max = np.minimum(max1, max2)
            
            # 检查是否有交集
            if np.any(inter_min >= inter_max):
                return 0.0
            
            # 计算交集体积
            inter_volume = np.prod(inter_max - inter_min)
            
            # 计算IoU
            union_volume = volume1 + volume2 - inter_volume
            if union_volume <= 0:
                return 0.0
            
            iou = inter_volume / union_volume
            return max(0.0, min(1.0, iou))
            
        except Exception as e:
            print(f"⚠️  AABB IoU计算出错: {e}")
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
            # print(f"⚠️  异常IoU值: {iou:.6f}")
            # print(f"  intersection_count: {intersection_count}, dV: {dV:.6f}")
            # print(f"  vol1: {vol1:.6f}, vol2: {vol2:.6f}")
            # print(f"  intersection_volume: {intersection_volume:.6f}")
            # print(f"  union_volume: {union_volume:.6f}")
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
                    'roll': batch['roll'].to(self.device),
                    'pitch': batch['pitch'].to(self.device),
                    'yaw': batch['yaw'].to(self.device),
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
                            temperature=self.incremental_temperature,
                            eos_threshold=self.eos_threshold
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
                
                # 计算每个样本的生成指标
                batch_size = targets['x'].size(0)
                batch_x_error = 0.0
                batch_y_error = 0.0
                batch_z_error = 0.0
                batch_w_error = 0.0
                batch_h_error = 0.0
                batch_l_error = 0.0
                batch_overall_error = 0.0
                
                for sample_idx in range(batch_size):
                    # 提取单个样本的结果
                    sample_gen_results = {}
                    sample_targets = {}
                    
                    for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
                        if attr in gen_results:
                            sample_gen_results[attr] = gen_results[attr][sample_idx:sample_idx+1]  # [1, seq_len]
                        if attr in targets:
                            sample_targets[attr] = targets[attr][sample_idx:sample_idx+1]  # [1, seq_len]
                    
                    # 计算单个样本的指标
                    # 🔧 修复：使用与验证集相同的等效box逻辑
                    if batch.get('equivalent_boxes'):
                        sample_equivalent_boxes = [batch['equivalent_boxes'][sample_idx]]
                    else:
                        sample_equivalent_boxes = None
                    sample_metrics = self._compute_generation_metrics(sample_gen_results, sample_targets, None, verbose=False, equivalent_boxes=sample_equivalent_boxes)
                    
                    # 打印每个样本的结果
                    if self.is_main_process:
                        actual_sample_idx = batch_idx * batch_size + sample_idx + 1
                        print(f"   Test Sample {actual_sample_idx}: Mean IoU = {sample_metrics['iou']:.4f} ({sample_metrics['num_generated_boxes']:.0f} boxes)")
                    
                    # 累积统计
                    total_gen_iou += sample_metrics['iou']
                    total_generated_boxes += sample_metrics['num_generated_boxes']
                    total_gt_boxes += sample_metrics['num_gt_boxes']
                    
                    # 累积误差指标
                    batch_x_error += sample_metrics.get('x_error', 0.0)
                    batch_y_error += sample_metrics.get('y_error', 0.0)
                    batch_z_error += sample_metrics.get('z_error', 0.0)
                    batch_w_error += sample_metrics.get('w_error', 0.0)
                    batch_h_error += sample_metrics.get('h_error', 0.0)
                    batch_l_error += sample_metrics.get('l_error', 0.0)
                    batch_overall_error += sample_metrics.get('overall_mean_error', 0.0)
                
                # 累积到总误差 - 🔧 修复：直接累积，不要多除batch_size
                total_x_error += batch_x_error
                total_y_error += batch_y_error
                total_z_error += batch_z_error
                total_w_error += batch_w_error
                total_h_error += batch_h_error
                total_l_error += batch_l_error
                total_overall_error += batch_overall_error
                
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
                        'roll': targets['roll'].cpu().tolist(),
                        'pitch': targets['pitch'].cpu().tolist(),
                        'yaw': targets['yaw'].cpu().tolist(),
                    },
                    'metrics': {
                        'iou': total_gen_iou / batch_size if batch_size > 0 else 0.0,
                        'num_generated_boxes': total_generated_boxes / batch_size if batch_size > 0 else 0.0,
                        'num_gt_boxes': total_gt_boxes / batch_size if batch_size > 0 else 0.0,
                        'x_error': batch_x_error / batch_size if batch_size > 0 else 0.0,
                        'y_error': batch_y_error / batch_size if batch_size > 0 else 0.0,
                        'z_error': batch_z_error / batch_size if batch_size > 0 else 0.0,
                        'w_error': batch_w_error / batch_size if batch_size > 0 else 0.0,
                        'h_error': batch_h_error / batch_size if batch_size > 0 else 0.0,
                        'l_error': batch_l_error / batch_size if batch_size > 0 else 0.0,
                        'overall_mean_error': batch_overall_error / batch_size if batch_size > 0 else 0.0
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
                        'test_best_val/roll_error': test_results.get('avg_roll_error', 0.0),
                        'test_best_val/pitch_error': test_results.get('avg_pitch_error', 0.0),
                        'test_best_val/yaw_error': test_results.get('avg_yaw_error', 0.0),
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
                        'test_best_gen/roll_error': test_results.get('avg_roll_error', 0.0),
                        'test_best_gen/pitch_error': test_results.get('avg_pitch_error', 0.0),
                        'test_best_gen/yaw_error': test_results.get('avg_yaw_error', 0.0),
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
                if self.current_epoch > phase_start_epoch + epoch_in_phase:
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
                        'train/eos_loss': train_stats.train_eos_loss,
                        'train/mean_iou': train_stats.train_mean_iou,
                        
                        # 🔍 添加旋转角度损失记录
                        'train/roll_cls_loss': train_stats.train_roll_cls_loss,
                        'train/pitch_cls_loss': train_stats.train_pitch_cls_loss,
                        'train/yaw_cls_loss': train_stats.train_yaw_cls_loss,
                        'train/roll_delta_loss': train_stats.train_roll_delta_loss,
                        'train/pitch_delta_loss': train_stats.train_pitch_delta_loss,
                        'train/yaw_delta_loss': train_stats.train_yaw_delta_loss,
                        
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
                        'val/roll_error': val_results.get('avg_roll_error', 0.0),
                        'val/pitch_error': val_results.get('avg_pitch_error', 0.0),
                        'val/yaw_error': val_results.get('avg_yaw_error', 0.0),
                        
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
                    print(f"   旋转误差: Roll={val_results.get('avg_roll_error', 0.0):.4f} | Pitch={val_results.get('avg_pitch_error', 0.0):.4f} | Yaw={val_results.get('avg_yaw_error', 0.0):.4f}")
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
                'test/roll_error': test_results.get('avg_roll_error', 0.0),
                'test/pitch_error': test_results.get('avg_pitch_error', 0.0),
                'test/yaw_error': test_results.get('avg_yaw_error', 0.0),
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