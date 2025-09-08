#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置加载器 - 统一管理模型、损失函数和数据加载器的配置
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from primitive_anything_3d import PrimitiveTransformer3D
from loss_3d import AdaptivePrimitiveTransformer3DLoss
from dataloader_3d import create_dataloader


class ConfigLoader:
    def __init__(self):
        self.config = {}
        self._unified_mode = True  # 🔧 修复：强制使用统一模式
    
    def load_unified_config(self, config_path: str = 'training_config.yaml'):
        """加载统一配置文件 - 现在是唯一的配置加载方式"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            self._unified_mode = True
            
            # 验证配置完整性
            self._validate_unified_config()
            
            print(f"✅ 成功加载统一配置文件: {config_path}")
            print(f"📊 配置版本: {self.config.get('version', 'unknown')}")
            
            # 显示训练阶段信息
            phases = self.config.get('training', {}).get('phases', {})
            total_epochs = sum(phase.get('epochs', 0) for phase in phases.values())
            print(f"🎯 训练阶段: {len(phases)}个阶段，总计{total_epochs}个epoch")
            for name, phase in phases.items():
                print(f"  - {name}: {phase.get('epochs', 0)} epochs")
            
            return self.config
            
        except FileNotFoundError:
            print(f"❌ 配置文件未找到: {config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"❌ 配置文件格式错误: {e}")
            raise
    
    def _validate_unified_config(self):
        """验证统一配置的完整性"""
        required_sections = ['version', 'global', 'model', 'training', 'data', 'loss']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少必需的section: {section}")
        
        # 验证关键参数
        global_config = self.config['global']
        required_global = ['max_seq_len', 'image_size']
        
        for param in required_global:
            if param not in global_config:
                raise ValueError(f"全局配置缺少必需参数: {param}")
    
    # ======================================================================
    # 配置访问方法
    # ======================================================================
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.config.get('model', {})
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        return self.config.get('loss', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """获取训练配置"""
        return self.config.get('training', {})
    
    def get_global_config(self) -> Dict[str, Any]:
        """获取全局配置"""
        return self.config.get('global', {})
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """获取实验配置"""
        return self.config.get('experiment', {})
    
    def get_training_phases(self) -> Dict[str, Any]:
        """获取训练阶段配置"""
        return self.config.get('training', {}).get('phases', {})
    
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置"""
        return self.config.get('training', {}).get('optimizer', {})
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取学习率调度配置"""
        return self.config.get('training', {}).get('scheduler', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """获取训练优化配置"""
        return self.config.get('training', {}).get('optimizations', {})
    
    # ======================================================================
    # 便捷访问方法
    # ======================================================================
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点记法 (e.g., 'global.max_seq_len')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值，支持点记法"""
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save_config(self, config_path: str):
        """保存当前配置到文件"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ 配置已保存到: {config_path}")
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*50)
        print("📋 配置摘要")
        print("="*50)
        
        # 全局配置
        global_config = self.get_global_config()
        print(f"🌍 全局配置:")
        print(f"  • 最大序列长度: {global_config.get('max_seq_len', 'N/A')}")
        print(f"  • 图像尺寸: {global_config.get('image_size', 'N/A')}")
        
        # 模型配置
        model_config = self.get_model_config()
        print(f"🧠 模型配置:")
        print(f"  • 模型维度: {model_config.get('dim', 'N/A')}")
        print(f"  • Transformer层数: {model_config.get('depth', 'N/A')}")
        print(f"  • 注意力头数: {model_config.get('heads', 'N/A')}")
        
        # 训练配置
        training_config = self.get_training_config()
        print(f"🎯 训练配置:")
        batch_size = training_config.get('dataloader', {}).get('batch_size', 'N/A')
        print(f"  • 批次大小: {batch_size}")
        
        optimizer_config = training_config.get('optimizer', {})
        print(f"  • 优化器: {optimizer_config.get('name', 'N/A')}")
        print(f"  • 学习率: {optimizer_config.get('lr', 'N/A')}")
        
        # 数据配置
        data_config = self.get_data_config()
        augmentation = data_config.get('augmentation', {}).get('enabled', False)
        print(f"📊 数据配置:")
        print(f"  • 数据增强: {'启用' if augmentation else '禁用'}")
        
        print("="*50)


# 向后兼容的便捷函数
def load_config(config_path: str = 'training_config.yaml') -> ConfigLoader:
    """便捷函数：加载配置"""
    loader = ConfigLoader()
    loader.load_unified_config(config_path)
    return loader 