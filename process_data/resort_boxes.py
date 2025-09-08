#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新排序sim_data中所有json文件的box
按照新的position进行排序：
1. 按x坐标从小到大（主排序）
2. 如果x相差小于tolerance，按z坐标从大到小（次排序）
3. 如果z也相差小于tolerance，按y坐标从小到大（第三排序）
"""

import json
import argparse
from pathlib import Path
import os
from tqdm import tqdm

def sort_boxes(boxes, tolerance=0.03):
    """
    对boxes进行排序：
    1. 按x坐标从小到大
    2. 如果x相差小于tolerance，按z坐标从大到小
    3. 如果z也相差小于tolerance，按y坐标从小到大
    
    Args:
        boxes: box列表，每个box包含position字段
        tolerance: 容差值，默认0.11
    
    Returns:
        list: 排序后的boxes
    """
    if not boxes:
        return boxes
    
    # 简化思路：使用元组排序，但需要处理容差逻辑
    def sort_key(box):
        pos = box['position']
        x, y, z = pos[0], pos[1], pos[2]
        
        # 将连续值离散化为分组，然后在组内进行精确排序
        x_group = round(x / tolerance)
        z_group = round(z / tolerance)
        y_group = round(y / tolerance)
        
        return (
            -z_group,  # z坐标分组，负号实现从大到小（次排序）
            x_group,   # x坐标分组（主排序）
            y_group,   # y坐标分组（第三排序）
            x,         # 组内按精确x排序
            -z,        # 组内按精确z排序（大到小）
            y          # 组内按精确y排序
        )
    
    # 使用排序键
    sorted_boxes = sorted(boxes, key=sort_key)
    
    return sorted_boxes

def process_json_file(file_path, backup=True):
    """处理单个json文件"""
    try:
        # 读取json文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建备份
        if backup:
            backup_path = str(file_path) + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        # 排序boxes
        if 'boxes' in data:
            original_boxes = data['boxes'].copy()
            sorted_boxes = sort_boxes(data['boxes'])
            data['boxes'] = sorted_boxes
            
            # 检查是否有变化
            has_changes = False
            for i, (orig, sorted_box) in enumerate(zip(original_boxes, sorted_boxes)):
                if orig != sorted_box:
                    has_changes = True
                    break
            
            if has_changes:
                print(f"  📊 {file_path}: {len(sorted_boxes)} boxes reordered")
        
        # 保存修改后的文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return True, len(data.get('boxes', []))
        
    except Exception as e:
        return False, str(e)

def find_json_files(data_dir):
    """查找所有json文件"""
    json_files = []
    data_path = Path(data_dir)
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    return json_files

def main():
    parser = argparse.ArgumentParser(description='重新排序sim_data中所有json文件的box')
    parser.add_argument('--data_dir', type=str, default='sim_data', 
                       help='数据目录路径 (默认: sim_data)')
    parser.add_argument('--no_backup', action='store_true', 
                       help='不创建备份文件')
    parser.add_argument('--dry_run', action='store_true', 
                       help='只显示将要处理的文件，不实际修改')
    
    args = parser.parse_args()
    
    print("🔍 查找json文件...")
    json_files = find_json_files(args.data_dir)
    
    if not json_files:
        print(f"❌ 在 {args.data_dir} 中没有找到json文件")
        return
    
    print(f"📁 找到 {len(json_files)} 个json文件")
    
    if args.dry_run:
        print("\n🔍 干运行模式 - 只显示将要处理的文件:")
        for file_path in json_files:
            print(f"  {file_path}")
        return
    
    # 处理文件
    success_count = 0
    total_boxes = 0
    failed_files = []
    
    print(f"\n🔄 开始处理文件...")
    for file_path in tqdm(json_files, desc="处理文件"):
        success, result = process_json_file(file_path, backup=not args.no_backup)
        
        if success:
            success_count += 1
            total_boxes += result
        else:
            failed_files.append((file_path, result))
    
    print(f"\n✅ 处理完成!")
    print(f"📊 统计信息:")
    print(f"  成功处理文件: {success_count}/{len(json_files)}")
    print(f"  总box数量: {total_boxes}")
    print(f"  失败文件: {len(failed_files)}")
    
    if failed_files:
        print(f"\n❌ 失败的文件:")
        for file_path, error in failed_files:
            print(f"  {file_path}: {error}")
    
    if not args.no_backup:
        print(f"\n💾 备份文件已创建 (添加.backup后缀)")

if __name__ == "__main__":
    main() 