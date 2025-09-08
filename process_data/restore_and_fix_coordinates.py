#!/usr/bin/env python3
"""
恢复原始文件并重新处理坐标
"""

import os
import subprocess
from pathlib import Path

def restore_backup_files():
    """恢复所有备份文件"""
    print("🔄 恢复备份文件...")
    
    # 查找所有备份文件
    backup_files = []
    for root, dirs, files in os.walk('sim_data'):
        for file in files:
            if file.endswith('.backup'):
                backup_path = os.path.join(root, file)
                original_path = backup_path[:-7]  # 移除.backup后缀
                backup_files.append((backup_path, original_path))
    
    print(f"📁 找到 {len(backup_files)} 个备份文件")
    
    # 恢复文件
    for backup_path, original_path in backup_files:
        try:
            # 复制备份文件到原始位置
            subprocess.run(['cp', backup_path, original_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 恢复失败 {original_path}: {e}")
    
    print("✅ 备份文件恢复完成")

def main():
    print("🔧 修正坐标处理逻辑")
    print("=" * 50)
    
    # 1. 恢复备份文件
    restore_backup_files()
    
    # 2. 重新运行修正后的脚本
    print("\n🔄 重新处理坐标...")
    subprocess.run(['python', 'modify_box_coordinates.py'])
    
    print("\n✅ 坐标修正完成！")

if __name__ == "__main__":
    main() 