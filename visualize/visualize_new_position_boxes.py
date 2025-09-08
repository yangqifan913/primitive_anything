#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化新定义position的Box和角点的脚本
使用rerun-0.21 API可视化box和position所在的角点
每个frame显示一组数据，突出显示新定义的position角点
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

def create_box_corners_from_new_position(size, position, rotation):
    """
    从新定义的position创建box的8个角点坐标
    新position是box的一个角点，我们需要计算其他7个角点
    
    Args:
        size: [length, width, height] - 新定义的尺寸顺序
        position: [x, y, z] - 新定义的position（角点坐标）
        rotation: [x, y, z, w] 四元数
    
    Returns:
        corners: (8, 3) 数组，包含8个角点的坐标
        corner_indices: 新position对应的角点索引
    """
    # 新position是box的一个角点，根据定义：
    # - 沿length方向使x最小
    # - 沿width方向使y最大  
    # - 沿height方向使z最大
    # 这意味着新position是角点 [-, +, +] 即索引6
    
    length, width, height = size
    
    # 从新position计算box中心
    # 新position是通过以下步骤从box中心得到的：
    # 1. 沿length方向平移length/2，选择x最小的方向
    # 2. 沿width方向平移width/2，选择y最大的方向
    # 3. 沿height方向平移height/2，选择z最大的方向
    # 
    # 所以要从新position计算box中心，需要反向操作：
    # 1. 沿length方向平移-length/2，选择x最大的方向
    # 2. 沿width方向平移-width/2，选择y最小的方向
    # 3. 沿height方向平移-height/2，选择z最小的方向
    
    length, width, height = size
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    
    # 获取局部坐标轴
    local_x_axis = rotation_matrix[:, 0]  # length方向
    local_y_axis = rotation_matrix[:, 1]  # width方向
    local_z_axis = rotation_matrix[:, 2]  # height方向
    
    # 反向计算：从新position回到box中心
    # 1. 沿length方向平移-length/2，选择x最大的方向
    length_direction = local_x_axis
    if length_direction[0] < 0:  # 如果length方向在x轴负方向
        length_direction = -length_direction  # 取反，使x最大
    length_translation = length_direction * (length / 2)
    
    # 2. 沿width方向平移-width/2，选择y最小的方向
    width_direction = local_y_axis
    if width_direction[1] > 0:  # 如果width方向在y轴正方向
        width_direction = -width_direction  # 取反，使y最小
    width_translation = width_direction * (width / 2)
    
    # 3. 沿height方向平移-height/2，选择z最小的方向
    height_direction = local_z_axis
    if height_direction[2] > 0:  # 如果height方向在z轴正方向
        height_direction = -height_direction  # 取反，使z最小
    height_translation = height_direction * (height / 2)
    
    # 计算box中心
    box_center = np.array(position) + length_translation + width_translation + height_translation
    
    # 计算box的8个角点（在box坐标系中）
    half_size = np.array([length, width, height]) / 2
    
    corners_local = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],  # 0: ---
        [+half_size[0], -half_size[1], -half_size[2]],  # 1: +--
        [-half_size[0], +half_size[1], -half_size[2]],  # 2: -+-
        [+half_size[0], +half_size[1], -half_size[2]],  # 3: ++-
        [-half_size[0], -half_size[1], +half_size[2]],  # 4: --+
        [+half_size[0], -half_size[1], +half_size[2]],  # 5: +-+
        [-half_size[0], +half_size[1], +half_size[2]],  # 6: -++ (新position)
        [+half_size[0], +half_size[1], +half_size[2]],  # 7: +++
    ])
    
    # 转换到世界坐标系
    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    corners_world = (rotation_matrix @ corners_local.T).T + box_center
    
    return corners_world, 6  # 返回角点和新position对应的索引

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
    
    # 2. 可视化boxes和角点
    all_corners = []
    all_new_positions = []
    
    for i, box in enumerate(boxes):
        try:
            position = box['position']  # 新定义的position（角点坐标）
            rotation = box['rotation']  # [x, y, z, w] 四元数
            size = box['size']  # [length, width, height]
            
            # 记录box实体
            box_entity = f"world/boxes/box_{i:03d}"
            
            # 从新position计算box中心
            # 新position是通过以下步骤从box中心得到的：
            # 1. 沿length方向平移length/2，选择x最小的方向
            # 2. 沿width方向平移width/2，选择y最大的方向
            # 3. 沿height方向平移height/2，选择z最大的方向
            # 
            # 所以要从新position计算box中心，需要反向操作：
            # 1. 沿length方向平移-length/2，选择x最大的方向
            # 2. 沿width方向平移-width/2，选择y最小的方向
            # 3. 沿height方向平移-height/2，选择z最小的方向
            
            length, width, height = size
            rotation_matrix = quaternion_to_rotation_matrix(rotation)
            
            # 获取局部坐标轴
            local_x_axis = rotation_matrix[:, 0]  # length方向
            local_y_axis = rotation_matrix[:, 1]  # width方向
            local_z_axis = rotation_matrix[:, 2]  # height方向
            
            # 反向计算：从新position回到box中心
            # 1. 沿length方向平移-length/2，选择x最大的方向
            length_direction = local_x_axis
            if length_direction[0] < 0:  # 如果length方向在x轴负方向
                length_direction = -length_direction  # 取反，使x最大
            length_translation = length_direction * (length / 2)
            
            # 2. 沿width方向平移-width/2，选择y最小的方向
            width_direction = local_y_axis
            if width_direction[1] > 0:  # 如果width方向在y轴正方向
                width_direction = -width_direction  # 取反，使y最小
            width_translation = width_direction * (width / 2)
            
            # 3. 沿height方向平移-height/2，选择z最小的方向
            height_direction = local_z_axis
            if height_direction[2] > 0:  # 如果height方向在z轴正方向
                height_direction = -height_direction  # 取反，使z最小
            height_translation = height_direction * (height / 2)
            
            # 计算box中心
            box_center = np.array(position) + length_translation + width_translation + height_translation
            
            # 使用Boxes3D接口直接生成box
            rr.log(
                box_entity,
                rr.Boxes3D(
                    half_sizes=[length/2, width/2, height/2],  # [长/2, 宽/2, 高/2]
                    centers=[box_center],  # 使用计算出的box中心
                    rotations=[rotation],  # 四元数格式 [x, y, z, w]
                    colors=[[0, 255, 0]],  # 绿色
                    labels=[f"Box{i+1}"]  # 直接在box上显示标签
                )
            )
            
            # 可视化新position点（红色大点）
            rr.log(
                f"world/boxes/box_{i:03d}/new_position",
                rr.Points3D(
                    positions=[position],
                    colors=[[255, 0, 0]],  # 红色
                    radii=[0.03],  # 大点
                    labels=[f"NewPosition{i+1}"]
                )
            )
            
            # 在box上方显示编号标签点
            label_position = [box_center[0], box_center[1], box_center[2] + height/2 + 0.05]
            text_entity = f"world/boxes/box_{i:03d}/label"
            rr.log(
                text_entity,
                rr.Points3D(
                    positions=[label_position],
                    colors=[[255, 255, 0]],  # 黄色
                    radii=[0.00001],  # 非常小的点
                )
            )
            
        except Exception as e:
            print(f"  处理box {i} 时出错: {e}")
            continue
    
    print(f"  boxes: {len(boxes)} 个")
    
    # 3. 显示frame信息和新position说明
    frame_info = f"""
# Frame {frame_name} 信息

## 点云统计
- 有效点数: {len(valid_points):,}
- 总像素数: {height * width:,}

## Box统计
- Box总数: {len(boxes)}
- 新Position定义: 通过三步平移选择的角点
- 选择步骤: 1)沿length方向选择x最小 2)沿width方向选择y最大 3)沿height方向选择z最大
- 尺寸定义: [length, width, height]
- 坐标系定义: x轴=length方向，y轴=width方向，z轴=height方向

## 新Position说明
- 红色大点: 新定义的position位置
- 绿色box: 使用计算出的box中心显示的box
- 黄色标签: box编号标签

## Box详情
"""
    
    for i, box in enumerate(boxes):
        pos = box['position']
        size = box['size']
        rot = box['rotation']
        frame_info += f"- **Box {i+1}**: pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], size=[L:{size[0]:.3f}, W:{size[1]:.3f}, H:{size[2]:.3f}]\n"
    
    # 记录frame信息
    rr.log(
        f"info/frame_{frame_name}",
        rr.TextDocument(
            text=frame_info,
            media_type=rr.MediaType.MARKDOWN
        )
    )

def visualize_dataset(input_path, output_name="new_position_visualization"):
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
    print(f"4. 红色角点表示新定义的position位置")
    print(f"5. 白色角点表示box的其他角点")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='可视化新定义position的Box和角点')
    parser.add_argument('--input', '-i', required=True, help='输入路径（处理后的数据目录）')
    parser.add_argument('--name', '-n', default='new_position_visualization', help='rerun应用名称')
    
    args = parser.parse_args()
    
    visualize_dataset(args.input, args.name) 