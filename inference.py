#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D物体检测推理脚本
加载保存的checkpoint.pth，对指定数据进行推理，并保存为JSON格式
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入项目模块
from config_loader import ConfigLoader
from primitive_anything_3d import PrimitiveTransformer3D
from dataloader_3d import Box3DDataset, create_dataloader


class InferenceEngine:
    """3D物体检测推理引擎"""
    
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = "cuda",
        use_incremental: bool = True,
        temperature: float = None,  # 改为None，从配置文件读取
        eos_threshold: float = 0.5
    ):
        """
        初始化推理引擎
        
        Args:
            config_path: 配置文件路径
            checkpoint_path: checkpoint文件路径
            device: 推理设备
            use_incremental: 是否使用增量推理
            temperature: 采样温度
            eos_threshold: EOS阈值
        """
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_incremental = use_incremental
        self.eos_threshold = eos_threshold
        
        # 加载配置
        self.config_loader = ConfigLoader()
        self.config_loader.load_unified_config(config_path)
        
        # 从配置文件读取温度参数
        if temperature is None:
            opt_config = self.config_loader.get_training_config().get('optimizations', {})
            inference_config = opt_config.get('incremental_inference', {})
            self.temperature = inference_config.get('temperature', 0.3)
        else:
            self.temperature = temperature
        
        # 创建模型
        self.model = self._create_model()
        
        # 加载checkpoint
        self._load_checkpoint()
        
        # 设置模型为评估模式
        self.model.eval()
        
        print(f"✅ 推理引擎初始化完成")
        print(f"📁 配置文件: {config_path}")
        print(f"💾 Checkpoint: {checkpoint_path}")
        print(f"🖥️  设备: {device}")
        print(f"🔄 增量推理: {'启用' if use_incremental else '禁用'}")
        print(f"🌡️  温度: {temperature}")
        print(f"🛑 EOS阈值: {eos_threshold}")
    
    def _create_model(self) -> PrimitiveTransformer3D:
        """根据配置创建模型"""
        model_config = self.config_loader.get_model_config()
        global_config = self.config_loader.get_global_config()
        data_config = self.config_loader.get_data_config()
        
        # 提取模型参数
        discretization = model_config.get('discretization', {})
        embeddings = model_config.get('embeddings', {})
        transformer = model_config.get('transformer', {})
        image_encoder = model_config.get('image_encoder', {})
        conditioning = model_config.get('conditioning', {})
        advanced = model_config.get('advanced', {})
        
        # 连续值范围
        continuous_ranges = data_config.get('continuous_ranges', {})
        
        # 创建模型
        model = PrimitiveTransformer3D(
            # 离散化参数
            num_discrete_x=discretization.get('num_discrete_x', 128),
            num_discrete_y=discretization.get('num_discrete_y', 128),
            num_discrete_z=discretization.get('num_discrete_z', 128),
            num_discrete_w=discretization.get('num_discrete_w', 64),
            num_discrete_h=discretization.get('num_discrete_h', 64),
            num_discrete_l=discretization.get('num_discrete_l', 64),
            
            # 连续值范围
            continuous_range_x=continuous_ranges.get('x', [0.5, 3.0]),
            continuous_range_y=continuous_ranges.get('y', [-2.5, 2.5]),
            continuous_range_z=continuous_ranges.get('z', [-1.5, 0.5]),
            continuous_range_w=continuous_ranges.get('w', [0.3, 0.7]),
            continuous_range_h=continuous_ranges.get('h', [0.3, 0.7]),
            continuous_range_l=continuous_ranges.get('l', [0.3, 0.7]),
            
            # 嵌入维度
            dim_x_embed=embeddings.get('dim_x_embed', 64),
            dim_y_embed=embeddings.get('dim_y_embed', 64),
            dim_z_embed=embeddings.get('dim_z_embed', 64),
            dim_w_embed=embeddings.get('dim_w_embed', 64),
            dim_h_embed=embeddings.get('dim_h_embed', 64),
            dim_l_embed=embeddings.get('dim_l_embed', 64),
            
            # Transformer参数
            dim=transformer.get('dim', 512),
            max_primitive_len=global_config.get('max_seq_len', 12),
            attn_depth=transformer.get('depth', 6),
            attn_dim_head=transformer.get('dim_head', 64),
            attn_heads=transformer.get('heads', 8),
            attn_dropout=transformer.get('attn_dropout', 0.1),
            ff_dropout=transformer.get('ff_dropout', 0.1),
            
            # 图像编码器
            image_encoder_dim=image_encoder.get('output_dim', 256),
            use_fpn=image_encoder.get('use_fpn', True),
            backbone=image_encoder.get('backbone', 'resnet50'),
            pretrained=image_encoder.get('pretrained', True),
            
            # 条件化
            condition_on_image=conditioning.get('condition_on_image', True),
            
            # 高级配置
            gateloop_depth=advanced.get('gateloop', {}).get('depth', 2),
            gateloop_use_heinsen=advanced.get('gateloop_use_heinsen', False),
            
            # 其他参数
            pad_id=global_config.get('pad_id', -1.0)
        )
        
        print(f"✅ 模型创建完成")
        print(f"📊 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def _load_checkpoint(self):
        """加载checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint文件不存在: {self.checkpoint_path}")
        
        print(f"📥 加载checkpoint: {self.checkpoint_path}")
        
        # 加载checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # 检查checkpoint格式
        if 'model' in checkpoint:
            # 标准checkpoint格式
            model_state_dict = checkpoint['model']
            print(f"📊 Checkpoint信息:")
            print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
            print(f"  - Loss: {checkpoint.get('loss', 'unknown')}")
            print(f"  - 模型状态: {'✅ 已保存' if 'model' in checkpoint else '❌ 缺失'}")
        elif isinstance(checkpoint, dict) and any(key.startswith('image_encoder') or key.startswith('decoder') for key in checkpoint.keys()):
            # 直接是模型状态字典
            model_state_dict = checkpoint
            print(f"📊 Checkpoint格式: 直接模型状态字典")
        else:
            raise ValueError(f"无法识别的checkpoint格式: {type(checkpoint)}")
        
        # 加载模型权重
        try:
            self.model.load_state_dict(model_state_dict, strict=True)
            print(f"✅ 模型权重加载成功")
        except Exception as e:
            print(f"⚠️  严格加载失败，尝试非严格加载: {e}")
            try:
                self.model.load_state_dict(model_state_dict, strict=False)
                print(f"✅ 模型权重非严格加载成功")
            except Exception as e2:
                raise RuntimeError(f"模型权重加载失败: {e2}")
        
        # 移动到指定设备
        self.model = self.model.to(self.device)
        print(f"✅ 模型已移动到设备: {self.device}")
    
    def create_dataloader(
        self,
        data_root: str,
        stage: str = "test",
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False
    ) -> DataLoader:
        """创建数据加载器"""
        global_config = self.config_loader.get_global_config()
        data_config = self.config_loader.get_data_config()
        
        # 创建数据集
        dataset = Box3DDataset(
            data_root=data_root,
            stage=stage,
            max_boxes=global_config.get('max_seq_len', 12),
            image_size=global_config.get('image_size', [640, 640]),
            continuous_ranges=data_config.get('continuous_ranges', {}),
            augmentation_config={},  # 推理时不使用数据增强
            augment=False,
            pad_id=global_config.get('pad_id', -1.0)
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        print(f"✅ 数据加载器创建完成")
        print(f"📊 数据集大小: {len(dataset)}")
        print(f"📦 批次大小: {batch_size}")
        print(f"🔄 数据阶段: {stage}")
        
        return dataloader
    
    def inference_single_batch(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        对单个批次进行推理
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            推理结果字典
        """
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
        
        # 检查输入数据有效性
        if torch.isnan(rgbxyz).any() or torch.isinf(rgbxyz).any():
            print(f"⚠️  批次 {batch_idx}: 输入数据包含NaN或Inf值，跳过")
            return None
        
        with torch.no_grad():
            try:
                # 根据配置选择推理方法
                if self.use_incremental:
                    print(f"🔄 使用增量推理")
                    # 使用增量推理
                    max_len = self.config_loader.get_global_config()['max_seq_len']
                    gen_results = self.model.generate_incremental(
                        image=rgbxyz,
                        max_seq_len=max_len,
                        temperature=self.temperature,
                        eos_threshold=self.eos_threshold
                    )
                else:
                    print(f"🔄 使用传统推理")
                    # 使用传统推理方法
                    max_len = self.config_loader.get_global_config()['max_seq_len']
                    gen_results = self.model.generate(
                        image=rgbxyz,
                        max_seq_len=max_len,
                        temperature=1.0,  # 传统推理使用温度1.0，与训练时一致
                        eos_threshold=self.eos_threshold
                    )
                
                # 检查生成结果的有效性
                if not gen_results or not isinstance(gen_results, dict):
                    print(f"⚠️  批次 {batch_idx}: 生成结果无效，跳过")
                    return None
                
                # 构建结果字典
                result = {
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
                    }
                }
                
                return result
                
            except Exception as e:
                print(f"❌ 批次 {batch_idx}: 推理失败 - {e}")
                return None
    
    def inference_dataset(
        self,
        data_root: str,
        stage: str = "test",
        batch_size: int = 1,
        max_samples: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        对整个数据集进行推理
        
        Args:
            data_root: 数据根目录
            stage: 数据阶段 (test/val)
            batch_size: 批次大小
            max_samples: 最大样本数（None表示全部）
            save_path: 保存路径（None表示不保存）
            
        Returns:
            完整的推理结果
        """
        print(f"🚀 开始数据集推理")
        print(f"📁 数据根目录: {data_root}")
        print(f"🔄 数据阶段: {stage}")
        print(f"📦 批次大小: {batch_size}")
        if max_samples:
            print(f"🔢 最大样本数: {max_samples}")
        
        # 创建数据加载器
        dataloader = self.create_dataloader(
            data_root=data_root,
            stage=stage,
            batch_size=batch_size,
            shuffle=False
        )
        
        # 存储所有结果
        all_results = []
        processed_samples = 0
        total_batches = len(dataloader)
        
        # 推理循环
        with tqdm(total=total_batches, desc="推理进度") as pbar:
            for batch_idx, batch_data in enumerate(dataloader):
                # 检查是否达到最大样本数
                if max_samples and processed_samples >= max_samples:
                    print(f"✅ 已达到最大样本数 {max_samples}，停止推理")
                    break
                
                # 进行推理
                batch_result = self.inference_single_batch(
                    batch=batch_data,
                    batch_idx=batch_idx
                )
                
                if batch_result is not None:
                    all_results.append(batch_result)
                    processed_samples += 1
                
                pbar.update(1)
                pbar.set_postfix({
                    'processed': processed_samples,
                    'batches': len(all_results)
                })
        
        # 构建最终结果
        final_result = {
            'test_summary': {
                'num_samples': processed_samples,
                'config_path': self.config_path,
                'checkpoint_path': self.checkpoint_path,
                'device': self.device,
                'use_incremental': self.use_incremental,
                'temperature': self.temperature,
                'eos_threshold': self.eos_threshold,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': all_results
        }
        
        # 保存结果
        if save_path:
            self._save_results(final_result, save_path)
        
        print(f"✅ 推理完成")
        print(f"📊 处理样本数: {processed_samples}")
        print(f"📦 成功批次: {len(all_results)}")
        
        return final_result
    
    def _save_results(self, results: Dict[str, Any], save_path: str):
        """保存推理结果到JSON文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 保存结果到: {save_path}")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='3D物体检测推理脚本')
    
    # 必需参数
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Checkpoint文件路径 (.pt 或 .pth)')
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据根目录路径')
    
    # 可选参数
    parser.add_argument('--output', type=str, default=None,
                       help='输出JSON文件路径（默认：checkpoint目录/inference_results.json）')
    parser.add_argument('--stage', type=str, default='test', choices=['test', 'val'],
                       help='数据阶段 (默认: test)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='批次大小 (默认: 1)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大样本数 (默认: 全部)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='推理设备 (默认: cuda)')
    parser.add_argument('--no-incremental', action='store_true',
                       help='禁用增量推理')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='采样温度 (默认: 0.3)')
    parser.add_argument('--eos-threshold', type=float, default=0.5,
                       help='EOS阈值 (默认: 0.5)')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='数据加载器工作进程数 (默认: 0)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.data_root):
        print(f"❌ 数据根目录不存在: {args.data_root}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = checkpoint_dir / 'inference_results.json'
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        args.device = 'cpu'
    
    try:
        # 创建推理引擎
        print("🔧 初始化推理引擎...")
        engine = InferenceEngine(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            use_incremental=not args.no_incremental,
            temperature=args.temperature,
            eos_threshold=args.eos_threshold
        )
        
        # 进行推理
        print("🚀 开始推理...")
        results = engine.inference_dataset(
            data_root=args.data_root,
            stage=args.stage,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            save_path=str(args.output)
        )
        
        print("🎉 推理完成！")
        print(f"📊 结果已保存到: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⚠️  推理被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
