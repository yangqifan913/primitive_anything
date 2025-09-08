#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化RGBXYZ点云和Box的脚本
使用rerun-0.21 API将彩色点云和box进行可视化
每个frame显示一组数据
"""

import numpy as np
import json
import argparse
from pathlib import Path
import rerun as rr
from simple_rgbxyz import load_and_split_rgbxyz

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵
    q: [x, y, z, w] 格式的四元数
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])

def create_box_corners(size, position, rotation):
    """
    创建box的8个角点坐标
    
    Args:
        size: [长, 宽, 高] - 标准化后的尺寸顺序
        position: [x, y, z]
        rotation: [x, y, z, w] 四元数
    
    Returns:
        corners: (8, 3) 数组，包含8个角点的坐标
    """
    # box的8个角点（在box坐标系中）
    # size现在是[长, 宽, 高]，对应x、y、z轴方向
    length, width, height = size
    half_size = np.array([length, width, height]) / 2
    
    corners_local = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],  # 0: ---
        [+half_size[0], -half_size[1], -half_size[2]],  # 1: +--
        [-half_size[0], +half_size[1], -half_size[2]],  # 2: -+-
        [+half_size[0], +half_size[1], -half_size[2]],  # 3: ++-
        [-half_size[0], -half_size[1], +half_size[2]],  # 4: --+
        [+half_size[0], -half_size[1], +half_size[2]],  # 5: +-+
        [-half_size[0], +half_size[1], +half_size[2]],  # 6: -++
        [+half_size[0], +half_size[1], +half_size[2]],  # 7: +++
    ])
    
    # 转换到世界坐标系
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    corners_world = (rotation_matrix @ corners_local.T).T + np.array(position)
    
    return corners_world

def create_box_lines(corners):
    """
    创建box的12条边线
    
    Args:
        corners: (8, 3) 角点坐标
    
    Returns:
        lines: (12, 2, 3) 数组，每条线由两个点定义
    """
    # box的12条边线（连接哪些角点）
    edges = [
        # 底面4条边
        [0, 1], [1, 3], [3, 2], [2, 0],
        # 顶面4条边
        [4, 5], [5, 7], [7, 6], [6, 4],
        # 垂直4条边
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    lines = []
    for edge in edges:
        line = np.array([corners[edge[0]], corners[edge[1]]])
        lines.append(line)
    
    return np.array(lines)

def visualize_single_frame(rgbxyz_file, boxes_file, frame_name):
    """
    可视化单个frame的数据
    
    Args:
        rgbxyz_file: RGBXYZ npz文件路径
        boxes_file: boxes json文件路径
        frame_name: frame名称
    """
    print(f"可视化 {frame_name}...")
    
    # 加载RGBXYZ数据
    try:
        data = load_and_split_rgbxyz(rgbxyz_file)
        rgb = data['rgb']  # (640, 640, 3)
        xyz = data['xyz']  # (640, 640, 3)
    except Exception as e:
        print(f"加载RGBXYZ数据失败: {e}")
        return
    
    # 加载boxes数据
    try:
        with open(boxes_file, 'r') as f:
            boxes_data = json.load(f)
        boxes = boxes_data['boxes']
    except Exception as e:
        print(f"加载boxes数据失败: {e}")
        return
    
    # 设置时间轴
    rr.set_time_sequence("frame", int(frame_name))
    
    # 1. 可视化点云
    # 将图像格式的数据转换为点云格式
    height, width = xyz.shape[:2]
    
    # 展平数据
    points = xyz.reshape(-1, 3)  # (640*640, 3)
    colors = rgb.reshape(-1, 3)  # (640*640, 3)
    
    # 过滤掉无效点
    # 注意：由于坐标系转换，Z值可能为负，需要调整过滤条件
    valid_mask = (np.abs(points[:, 0]) < 50) & (np.abs(points[:, 1]) < 50) & (np.abs(points[:, 2]) < 50)
    valid_points = points[valid_mask]
    valid_colors = colors[valid_mask]
    
    # 进一步过滤：移除明显异常的点
    distances = np.linalg.norm(valid_points, axis=1)
    distance_mask = distances < 10.0  # 距离原点小于10米
    valid_points = valid_points[distance_mask]
    valid_colors = valid_colors[distance_mask]
    
    # 记录点云
    rr.log(
        "world/pointcloud",
        rr.Points3D(
            positions=valid_points,
            colors=valid_colors.astype(np.uint8)
        )
    )
    
    print(f"  点云: {len(valid_points)} 个有效点")
    
    # 2. 可视化boxes - 使用Boxes3D接口，带编号
    for i, box in enumerate(boxes):
        try:
            position = box['position']
            rotation = box['rotation']  # [x, y, z, w] 四元数
            size = box['size']  # [长, 宽, 高] - 标准化后的顺序
            
            # 记录box实体
            box_entity = f"world/boxes/box_{i:03d}"
            
            # 使用Boxes3D接口直接生成box
            # size现在是[长, 宽, 高]，需要正确映射到half_sizes
            length, width, height = size
            rr.log(
                box_entity,
                rr.Boxes3D(
                    half_sizes=[length/2, width/2, height/2],  # [长/2, 宽/2, 高/2]
                    centers=[position],
                    rotations=[rotation],  # 四元数格式 [x, y, z, w]
                    colors=[[0, 255, 0]],  # 绿色
                    labels=[f"Box{i+1}"]  # 直接在box上显示标签
                )
            )
            
            # 在box上方显示编号标签点（可选择性显示小点）
           
            label_position = [position[0], position[1], position[2] + height/2 + 0.05]  # box上方
            text_entity = f"world/boxes/box_{i:03d}/label"
            rr.log(
                text_entity,
                rr.Points3D(
                    positions=[label_position],
                    colors=[[255, 255, 0]],  # 黄色
                    radii=[0.00001],  # 非常小的点
                    # labels=[f"Box{i+1}"]
                )
                )
            
            # 为每个box添加详细信息作为实体描述
            # rr.log(
            #     box_entity,
            #     rr.AnyValues({
            #         "box_id": i+1,
            #         "length": length,
            #         "width": width, 
            #         "height": height,
            #         "position_x": position[0],
            #         "position_y": position[1],
            #         "position_z": position[2]
            #     })
            # )
            
        except Exception as e:
            print(f"  处理box {i} 时出错: {e}")
            continue
    
    print(f"  boxes: {len(boxes)} 个")
    
    # 3. 显示frame信息和box排序信息
    frame_info = f"""
# Frame {frame_name} 信息

## 点云统计
- 有效点数: {len(valid_points):,}
- 总像素数: {height * width:,}

## Box统计
- Box总数: {len(boxes)}
- 排序规则: x坐标→z坐标(大到小)→y坐标
- 尺寸定义: [长, 宽, 高] (重新定义的局部坐标系)
- 坐标系定义: x轴=长度方向，y-z平面=宽高平面
- 长度: 最大的边，沿box的x轴方向

## Box详情
"""
    
    for i, box in enumerate(boxes):
        pos = box['position']
        size = box['size']
        rot = box['rotation']
        frame_info += f"- **Box {i+1}**: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], size=[L:{size[0]:.2f}, W:{size[1]:.2f}, H:{size[2]:.2f}]\n"
    
    # 记录frame信息
    rr.log(
        f"info/frame_{frame_name}",
        rr.TextDocument(
            text=frame_info,
            media_type=rr.MediaType.MARKDOWN
        )
    )

def visualize_dataset(input_path, output_name="rgbxyz_visualization"):
    """
    可视化整个数据集
    
    Args:
        input_path: 输入路径（包含处理后的数据）
        output_name: rerun应用名称
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"错误：输入路径不存在 {input_path}")
        return
    
    # 初始化rerun
    rr.init(output_name, spawn=True)
    
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
    
    # 设置坐标系 - 使用右手坐标系，Z轴向上
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    
    # 可视化每个frame
    for folder_name in folders:
        folder_path = input_path / folder_name
        
        rgbxyz_file = folder_path / f"{folder_name}.npz"
        boxes_file = folder_path / f"{folder_name}.json"
        
        # 检查文件是否存在
        if not rgbxyz_file.exists():
            print(f"警告：文件不存在 {rgbxyz_file}")
            continue
        if not boxes_file.exists():
            print(f"警告：文件不存在 {boxes_file}")
            continue
        
        # 可视化这一帧
        visualize_single_frame(str(rgbxyz_file), str(boxes_file), folder_name)
    
    print(f"\n可视化完成！")
    print(f"在rerun viewer中可以:")
    print(f"1. 使用时间轴控制查看不同的frame")
    print(f"2. 旋转、缩放查看3D场景")
    print(f"3. 在左侧面板中切换显示/隐藏不同的实体")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化RGBXYZ点云和boxes')
    parser.add_argument('--input', '-i', required=True, help='输入路径（处理后的数据目录）')
    parser.add_argument('--name', '-n', default='rgbxyz_visualization', help='rerun应用名称')
    
    args = parser.parse_args()
    
    visualize_dataset(args.input, args.name) 