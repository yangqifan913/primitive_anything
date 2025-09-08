#!/usr/bin/env python3
"""
修改sim_data中所有json标注文件的box坐标
将box的中心点重新定义为：
1. 沿length边的方向平移length/2，方向取使x最小的方向
2. 沿height边的方向平移height/2，方向取使z最大的方向
"""

import json
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def quaternion_to_rotation_matrix(q):
    """将四元数转换为旋转矩阵"""
    x, y, z, w = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
        [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
    ])

def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([x, y, z, w])

def modify_box_coordinates(box):
    """
    修改box的坐标定义
    将中心点重新定义为：
    1. 沿length方向平移length/2，方向取使x最小的方向
    2. 沿height方向平移height/2，方向取使z最大的方向
    """
    # 获取原始数据
    position = np.array(box['position'])  # [x, y, z] - 原始中心点
    rotation = np.array(box['rotation'])   # [qx, qy, qz, qw] - 四元数
    size = np.array(box['size'])          # [length, width, height] - 尺寸
    
    # 解析尺寸
    length, width, height = size
    
    # 将四元数转换为旋转矩阵
    R = quaternion_to_rotation_matrix(rotation)
    
    # 定义局部坐标轴（相对于box的坐标系）
    # 根据simple_rgbxyz.py的定义：
    # length是沿x方向的边，width是沿y方向的边，height是沿z方向的边
    local_x_axis = R[:, 0]  # length方向（沿x轴）
    local_y_axis = R[:, 1]  # width方向（沿y轴）
    local_z_axis = R[:, 2]  # height方向（沿z轴）
    
    # 计算平移向量
    # 1. 沿length方向平移length/2，方向取使x最小的方向
    length_direction = local_x_axis
    if length_direction[0] > 0:  # 如果length方向在x轴正方向
        length_direction = -length_direction  # 取反，使x最小
    length_translation = length_direction * (length / 2)
    
    # 2. 沿width方向平移width/2，方向取使y最大的方向
    width_direction = local_y_axis
    if width_direction[1] < 0:  # 如果width方向在y轴负方向
        width_direction = -width_direction  # 取反，使y最大
    width_translation = width_direction * (width / 2)
    
    # 3. 沿height方向平移height/2，方向取使z最大的方向
    height_direction = local_z_axis
    if height_direction[2] < 0:  # 如果height方向在z轴负方向
        height_direction = -height_direction  # 取反，使z最大
    height_translation = height_direction * (height / 2)
    
    # 计算新的中心点
    new_position = position + length_translation + width_translation + height_translation
    
    # 更新box数据
    box['position'] = new_position.tolist()
    
    return box

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
        
        # 修改每个box的坐标
        if 'boxes' in data:
            for box in data['boxes']:
                box = modify_box_coordinates(box)
        
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
    parser = argparse.ArgumentParser(description='修改sim_data中所有json文件的box坐标')
    parser.add_argument('--data_dir', type=str, default='sim_data', 
                       help='数据目录路径 (默认: sim_data)')
    parser.add_argument('--no_backup', action='store_true', 
                       help='不创建备份文件')
    parser.add_argument('--dry_run', action='store_true', 
                       help='只显示将要修改的文件，不实际修改')
    
    args = parser.parse_args()
    
    print("🔍 查找json文件...")
    json_files = find_json_files(args.data_dir)
    
    if not json_files:
        print(f"❌ 在 {args.data_dir} 中没有找到json文件")
        return
    
    print(f"📁 找到 {len(json_files)} 个json文件")
    
    if args.dry_run:
        print("\n🔍 干运行模式 - 只显示将要修改的文件:")
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
    
    # 输出结果
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