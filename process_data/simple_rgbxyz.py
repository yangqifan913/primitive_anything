#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量RGBXYZ生成和box pose转换脚本
将RGB图像和深度图合并为640×640×6的数组
计算相机坐标系下的box pose，重新定义长宽高和旋转角度
"""

import numpy as np
import json
import os
from PIL import Image
from pathlib import Path
import argparse
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: [x, y, z, w] 格式的四元数
    """
    return R.from_quat(q).as_matrix()

def rotation_matrix_to_quaternion(rot_mat):
    """
    将旋转矩阵转换为四元数
    """
    return R.from_matrix(rot_mat).as_quat()

def standardize_box_dimensions(box_position, box_rotation, box_size):
    """
    根据当前box的pose，通过计算三个边与世界坐标轴的夹角来定义长宽高：
    - 长边(length)：与x轴夹角最小的边
    - 宽边(width)：与y轴夹角最小的边  
    - 高边(height)：与z轴夹角最小的边
    - 然后重新定义box的pose，让长边作为新的x轴方向
    
    Args:
        box_position: [x, y, z] box中心位置
        box_rotation: [x, y, z, w] 四元数旋转
        box_size: [w, h, l] 原始尺寸
    
    Returns:
        dict: 包含重新定义后的position, rotation, size
    """
    # 获取原始旋转矩阵
    original_rot_matrix = quaternion_to_rotation_matrix(box_rotation)
    
    # box的三个局部坐标轴（列向量）
    local_x = original_rot_matrix[:, 0]  # box的局部x轴在世界坐标系中的方向
    local_y = original_rot_matrix[:, 1]  # box的局部y轴在世界坐标系中的方向
    local_z = original_rot_matrix[:, 2]  # box的局部z轴在世界坐标系中的方向
    
    # 世界坐标系的三个轴
    world_x = np.array([1, 0, 0])
    world_y = np.array([0, 1, 0])
    world_z = np.array([0, 0, 1])
    
    # 计算每个局部轴与世界轴的夹角（余弦值，越接近1夹角越小）
    angles_to_world_x = [
        abs(np.dot(local_x, world_x)),  # local_x与world_x的夹角余弦
        abs(np.dot(local_y, world_x)),  # local_y与world_x的夹角余弦
        abs(np.dot(local_z, world_x))   # local_z与world_x的夹角余弦
    ]
    
    angles_to_world_y = [
        abs(np.dot(local_x, world_y)),
        abs(np.dot(local_y, world_y)),
        abs(np.dot(local_z, world_y))
    ]
    
    angles_to_world_z = [
        abs(np.dot(local_x, world_z)),
        abs(np.dot(local_y, world_z)),
        abs(np.dot(local_z, world_z))
    ]
    
    # 找到与各世界轴夹角最小的局部轴
    length_axis_idx = np.argmax(angles_to_world_x)  # 与x轴夹角最小 -> 长边
    width_axis_idx = np.argmax(angles_to_world_y)   # 与y轴夹角最小 -> 宽边
    height_axis_idx = np.argmax(angles_to_world_z)  # 与z轴夹角最小 -> 高边
    
    # 获取对应的尺寸值
    original_sizes = np.array(box_size)  # [w, h, l]
    
    # 按照长宽高顺序重新排列尺寸
    standardized_size = [
        original_sizes[length_axis_idx],   # 长边（与x轴夹角最小）
        original_sizes[width_axis_idx],    # 宽边（与y轴夹角最小）
        original_sizes[height_axis_idx]    # 高边（与z轴夹角最小）
    ]
    
    # 构建新的局部坐标系，让长边作为新的x轴
    local_axes = [local_x, local_y, local_z]
    new_x_axis = local_axes[length_axis_idx]  # 长边方向作为新的x轴
    new_y_axis = local_axes[width_axis_idx]   # 宽边方向作为新的y轴
    new_z_axis = local_axes[height_axis_idx]  # 高边方向作为新的z轴
    
    # 确保构成右手坐标系
    cross_product = np.cross(new_x_axis, new_y_axis)
    if np.dot(cross_product, new_z_axis) < 0:
        # 如果不是右手坐标系，调整z轴方向
        new_z_axis = -new_z_axis
    
    # 构建新的旋转矩阵
    new_rot_matrix = np.column_stack([new_x_axis, new_y_axis, new_z_axis])
    
    # 将旋转矩阵转换为四元数
    new_rotation = rotation_matrix_to_quaternion(new_rot_matrix)
    
    return {
        'position': box_position,  # 位置不变
        'rotation': new_rotation.tolist(),  # 新的旋转四元数
        'size': standardized_size  # 按照[长, 宽, 高]顺序
    }

def sort_boxes(boxes, tolerance=0.11):
    """
    对boxes进行排序：
    1. 按x坐标从小到大
    2. 如果x相差小于tolerance(5cm)，按z坐标从大到小
    3. 如果z也相差小于tolerance，按y坐标从小到大
    
    Args:
        boxes: box列表，每个box包含position字段
        tolerance: 容差值，默认0.05(5cm)
    
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

def transform_box_to_new_world_coordinate(box, camera_position, camera_rotation):
    """
    将box从原世界坐标系转换到新的世界坐标系
    只做平移：将世界坐标系的原点移动到相机位置，坐标轴方向保持不变
    
    Args:
        box: 包含position, rotation, size的box信息
        camera_position: 相机位置 [x, y, z]
        camera_rotation: 相机旋转四元数 [x, y, z, w]（未使用，保留接口）
    
    Returns:
        dict: 包含转换后的position, rotation, size
    """
    # 步骤1：平移到新的世界坐标系（以相机位置为原点）
    box_world_pos = np.array(box['position'])
    transformed_pos = box_world_pos - np.array(camera_position)
    
    # 步骤2：保持旋转不变
    # 因为世界坐标系只是平移，坐标轴方向没有改变
    transformed_rotation = box['rotation']  # 保持原始旋转不变
    
    # 步骤3：保持原始尺寸不变
    transformed_size = box['size']  # 保持原始尺寸不变
    
    return {
        'position': transformed_pos.tolist(),
        'rotation': transformed_rotation,  # 保持原始格式
        'size': transformed_size
    }

def transform_pointcloud_to_new_world_coordinate(x, y, z, camera_position, camera_rotation):
    """
    将相机坐标系下的点云转换到新的世界坐标系
    新世界坐标系：以相机位置为原点，坐标轴方向与原世界坐标系相同
    
    转换过程：相机坐标系 -> 新世界坐标系（只做旋转对齐，不做平移）
    
    Args:
        x, y, z: 相机坐标系下的点云坐标 (H, W)
        camera_position: 相机位置 [x, y, z]（未使用，保留接口）
        camera_rotation: 相机旋转四元数 [x, y, z, w]
    
    Returns:
        tuple: (x_new, y_new, z_new) 新世界坐标系下的点云坐标
    """
    # 获取相机的旋转矩阵
    camera_rot_matrix = quaternion_to_rotation_matrix(camera_rotation)
    
    # 将点云数据展平以便矩阵运算
    height, width = x.shape
    points_camera = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=0)  # (3, N)
    
    points_new_world = camera_rot_matrix @ points_camera
    
    # 重新整形回原始形状
    x_new = points_new_world[0].reshape(height, width)
    y_new = points_new_world[1].reshape(height, width)
    z_new = points_new_world[2].reshape(height, width)
    
    return x_new, y_new, z_new

def create_rgbxyz_simple(rgb_file, depth_file, scene_file, output_file="rgbxyz.npz"):
    """
    生成RGBXYZ数据，点云从相机坐标系转换到新的世界坐标系
    
    Args:
        rgb_file: RGB图像文件路径
        depth_file: depth.npz文件路径
        scene_file: scene.json文件路径
        output_file: 输出文件名
    
    Returns:
        rgbxyz_array: (640, 640, 6) 数组，通道为 [R, G, B, X, Y, Z]
    """
    
    # 加载RGB图像
    rgb_image = Image.open(rgb_file)
    rgb_array = np.array(rgb_image)
    if rgb_array.shape[2] == 4:  # 如果是RGBA，去掉alpha通道
        rgb_array = rgb_array[:, :, :3]
    
    # 加载深度图
    depth_data = np.load(depth_file)
    depth_array = depth_data['arr_0']
    
    # 加载相机参数
    with open(scene_file, 'r') as f:
        scene_data = json.load(f)
    camera = scene_data['camera']
    intrinsics = np.array(camera['intrinsics'])
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # 获取相机位置和旋转
    camera_position = camera['position']
    camera_rotation = camera['rotation']
    
    # 生成像素坐标网格
    height, width = depth_array.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 转换为相机坐标系下的XYZ坐标
    x_camera = (u - cx) * depth_array / fx
    y_camera = (v - cy) * depth_array / fy
    z_camera = depth_array
    
    # 将点云从相机坐标系转换到新的世界坐标系
    x_new, y_new, z_new = transform_pointcloud_to_new_world_coordinate(
        x_camera, y_camera, z_camera, camera_position, camera_rotation
    )
    
    # 合并RGB和转换后的XYZ
    rgbxyz_array = np.concatenate([
        rgb_array,  # RGB: 3个通道
        np.stack([x_new, y_new, z_new], axis=2)  # XYZ: 3个通道（新世界坐标系）
    ], axis=2)
    
    # 保存为npz文件
    np.savez_compressed(output_file, rgbxyz=rgbxyz_array)
    
    depth_data.close()
    return rgbxyz_array

def process_single_folder(input_folder, output_folder, folder_name):
    """
    处理单个文件夹
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        folder_name: 文件夹名称（如"0000"）
    """
    print(f"正在处理文件夹: {folder_name}")
    
    # 构建文件路径
    rgb_file = os.path.join(input_folder, f"{folder_name}.png")
    depth_file = os.path.join(input_folder, "depth.npz")
    scene_file = os.path.join(input_folder, "scene.json")
    annotation_file = os.path.join(input_folder, f"{folder_name}.json")
    
    # 检查文件是否存在
    required_files = [rgb_file, depth_file, scene_file, annotation_file]
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"警告：文件不存在 {file_path}，跳过该文件夹")
            return
    
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 生成RGBXYZ
    rgbxyz_output = os.path.join(output_folder, f"{folder_name}.npz")
    try:
        rgbxyz_array = create_rgbxyz_simple(rgb_file, depth_file, scene_file, rgbxyz_output)
        print(f"生成RGBXYZ完成: {rgbxyz_output}, 形状: {rgbxyz_array.shape}")
    except Exception as e:
        print(f"生成RGBXYZ失败: {e}")
        return
    
    # 处理box pose
    try:
        # 加载场景数据
        with open(scene_file, 'r') as f:
            scene_data = json.load(f)
        
        # 加载标注数据
        with open(annotation_file, 'r') as f:
            annotation_data = json.load(f)
        
        # 获取相机参数
        camera = scene_data['camera']
        camera_position = camera['position']
        camera_rotation = camera['rotation']  # [x, y, z, w] 格式
        
        # 转换所有box的pose
        transformed_boxes = []
        
        # 使用annotation文件中的boxes（如果存在）
        boxes_to_process = annotation_data.get('boxes', [])
        if not boxes_to_process:
            # 如果annotation文件中没有boxes，使用scene文件中的
            boxes_to_process = scene_data.get('boxes', [])
        
        for box in boxes_to_process:
            # 跳过非box类别（比如container等）
            if box.get('category_id', 1) != 1:  # 假设category_id=1是box
                continue
                
            # 步骤1：转换到新的世界坐标系
            transformed_box = transform_box_to_new_world_coordinate(
                box, camera_position, camera_rotation
            )
            
            # 步骤2：标准化长宽高定义
            standardized_box = standardize_box_dimensions(
                transformed_box['position'],
                transformed_box['rotation'], 
                transformed_box['size']
            )
            
            # 保留原始的其他信息
            result_box = {
                'position': standardized_box['position'],
                'rotation': standardized_box['rotation'],
                'size': standardized_box['size'],  # 现在是[长, 宽, 高]顺序
                'instance_id': box.get('instance_id', 0),
                'category_id': box.get('category_id', 1)
            }
            
            # 如果有其他需要保留的字段，可以在这里添加
            for key in ['area', 'bbox']:
                if key in box:
                    result_box[key] = box[key]
            
            transformed_boxes.append(result_box)
        
        # 步骤3：对所有boxes进行排序
        transformed_boxes = sort_boxes(transformed_boxes)
        
        # 保存转换后的box pose
        output_data = {
            'boxes': transformed_boxes,
            'coordinate_system': {
                'description': '新的世界坐标系，原点平移到相机位置，坐标轴方向与原世界坐标系相同',
                'origin': '相机位置',
                'axes': '与原世界坐标系方向相同',
                'transformation': '只做平移变换：原点移到相机位置，旋转和尺寸保持不变'
            },
            'standardization': {
                'size_definition': 'size按[长, 宽, 高]顺序排列',
                'coordinate_system': '重新定义box的局部坐标系',
                'length': '最长的边，沿新的x轴方向',
                'width': '宽度，沿新的y轴方向',
                'height': '高度，沿新的z轴方向',
                'pose_redefinition': '宽和高组成的平面作为box的x轴方向（y-z平面是宽高平面）'
            },
            'sorting': {
                'primary': 'x坐标从小到大',
                'secondary': '如果x相差<5cm，按z坐标从大到小',
                'tertiary': '如果z也相差<5cm，按y坐标从小到大',
                'tolerance': '5cm (0.05)'
            }
        }
        
        pose_output = os.path.join(output_folder, f"{folder_name}.json")
        with open(pose_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"box pose转换完成: {pose_output}, 处理了 {len(transformed_boxes)} 个box")
        
    except Exception as e:
        print(f"box pose转换失败: {e}")

def batch_process(input_path, output_path):
    """
    批量处理指定路径下的所有文件夹
    
    Args:
        input_path: 输入路径（包含多个编号文件夹的目录）
        output_path: 输出路径
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"错误：输入路径不存在 {input_path}")
        return
    
    # 获取所有数字编号的文件夹
    folders = []
    for item in input_path.iterdir():
        if item.is_dir() and item.name.isdigit():
            folders.append(item.name)
    
    folders.sort()  # 按编号排序
    
    if not folders:
        print(f"错误：在 {input_path} 中没有找到数字编号的文件夹")
        return
    
    print(f"找到 {len(folders)} 个文件夹: {folders}")
    
    # 处理每个文件夹
    for folder_name in folders:
        input_folder = input_path / folder_name
        output_folder = output_path / folder_name
        
        try:
            process_single_folder(str(input_folder), str(output_folder), folder_name)
        except Exception as e:
            print(f"处理文件夹 {folder_name} 时出错: {e}")
            continue
    
    print(f"批量处理完成！结果保存在: {output_path}")

def load_and_split_rgbxyz(npz_file):
    """
    加载RGBXYZ文件并分离通道
    
    Returns:
        dict: 包含'rgb'和'xyz'的字典
    """
    data = np.load(npz_file)
    rgbxyz = data['rgbxyz']
    
    return {
        'rgb': rgbxyz[:, :, :3],  # RGB通道
        'xyz': rgbxyz[:, :, 3:6], # XYZ通道
        'full': rgbxyz            # 完整数据
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='批量生成RGBXYZ和box pose')
    parser.add_argument('--input', '-i', required=True, help='输入路径（包含编号文件夹的目录）')
    parser.add_argument('--output', '-o', required=True, help='输出路径')
    parser.add_argument('--single', '-s', help='处理单个文件夹（指定文件夹名称，如"0000"）')
    
    args = parser.parse_args()
    
    if args.single:
        # 处理单个文件夹
        input_folder = os.path.join(args.input, args.single)
        output_folder = os.path.join(args.output, args.single)
        process_single_folder(input_folder, output_folder, args.single)
    else:
        # 批量处理
        batch_process(args.input, args.output)
