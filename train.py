#!/usr/bin/env python3
"""
3D物体检测训练脚本 - 使用统一配置
"""

import argparse
import sys
import os
from pathlib import Path

from config_loader import ConfigLoader
from trainer import AdvancedTrainer


def main():
    parser = argparse.ArgumentParser(description='3D物体检测训练')
    parser.add_argument('--config', type=str, default='training_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--experiment', type=str, default='test_unified',
                       help='实验名称')
    parser.add_argument('--disable-swanlab', action='store_true',
                       help='禁用SwanLab日志')
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定checkpoint恢复训练')
    
    args = parser.parse_args()
    
    # 从环境变量获取分布式训练参数（torchrun自动设置）
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    # 设置CUDA设备
    if world_size > 1:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            
            # 初始化分布式环境
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl')
                
            print(f"🚀 进程 {rank}/{world_size} 使用 GPU {local_rank}")
    
    # 检查配置文件
    if not Path(args.config).exists():
        if local_rank == 0:  # 只在主进程打印错误
            print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    try:
        # 加载统一配置
        if local_rank == 0:
            print("🔧 加载配置文件...")
        config_loader = ConfigLoader()
        config_loader.load_unified_config(args.config)
        
        # 只在主进程打印配置摘要
        if local_rank == 0:
            config_loader.print_summary()
        
        # 确定是否使用SwanLab（只在主进程）
        use_swanlab = not args.disable_swanlab and (local_rank == 0)
        
        # 创建训练器
        if local_rank == 0:
            print(f"🚀 初始化训练器: {args.experiment}")
        trainer = AdvancedTrainer(
            config_loader=config_loader,
            experiment_name=args.experiment,
            use_swanlab=use_swanlab,
            local_rank=local_rank,
            world_size=world_size,
            resume_from=args.resume
        )
        
        # 开始训练
        if local_rank == 0:
            print("🎯 开始训练...")
        trainer.train()
        
        if local_rank == 0:
            print("🎉 训练完成！")
        
    except KeyboardInterrupt:
        if local_rank == 0:
            print("\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        if local_rank == 0:
            print(f"❌ 训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 