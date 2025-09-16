# -*- coding: utf-8 -*-
"""
3D Box Detection DataLoader
加载processed数据：RGBXYZ图像(npz) + 3D Box标注(json)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

def custom_collate_fn(batch):
    """自定义collate函数，处理字典格式的batch"""
    if not batch:
        return {}
    
    # 获取所有键
    keys = batch[0].keys()
    
    # 对每个键进行处理
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        
        # 特殊处理非tensor类型
        if key in ['folder_name']:
            # 字符串类型，直接保留为列表
            collated[key] = values
        elif key in ['equivalent_boxes']:
            # 列表类型，直接保留为列表
            collated[key] = values
        else:
            # tensor类型，进行stack
            tensor_values = []
            for value in values:
                if isinstance(value, torch.Tensor):
                    tensor_values.append(value)
                else:
                    # 其他类型转换为tensor
                    tensor_values.append(torch.tensor(value))
            
            collated[key] = torch.stack(tensor_values, dim=0)
    
    return collated
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
import math

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """将四元数转换为旋转矩阵"""
    import numpy as np
    
    # 归一化四元数
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    
    return R

def rotation_matrix_to_euler_scipy(R):
    """使用scipy将旋转矩阵转换为欧拉角 (ZYX顺序)"""
    from scipy.spatial.transform import Rotation as R_scipy
    
    # 使用scipy转换
    r = R_scipy.from_matrix(R)
    euler = r.as_euler('xyz', degrees=False)  # 返回 (roll, pitch, yaw)
    
    return euler[0], euler[1], euler[2]

def apply_local_rotation_scipy(roll, pitch, yaw, axis, angle):
    """
    在局部坐标系中应用旋转 (使用scipy，避免万向锁)
    
    Args:
        roll, pitch, yaw: 当前欧拉角（弧度）
        axis: 旋转轴 ('x', 'y', 'z')
        angle: 旋转角度（弧度）
    
    Returns:
        new_roll, new_pitch, new_yaw: 新的欧拉角（弧度）
    """
    from scipy.spatial.transform import Rotation as R_scipy
    import numpy as np
    import warnings
    
    # 使用四元数避免万向锁问题
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 忽略万向锁警告
        
        # 将欧拉角转换为四元数（输入弧度）
        r_original = R_scipy.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        
        # 创建局部旋转四元数（输入弧度）
        if axis == 'x':
            r_local = R_scipy.from_euler('x', angle, degrees=False)
        elif axis == 'y':
            r_local = R_scipy.from_euler('y', angle, degrees=False)
        elif axis == 'z':
            r_local = R_scipy.from_euler('z', angle, degrees=False)
        else:
            raise ValueError(f"Invalid axis: {axis}")
        
        # 应用局部旋转：q_new = q_original * q_local
        r_new = r_original * r_local
        
        # 转换回欧拉角（输出弧度）
        euler_new = r_new.as_euler('xyz', degrees=False)
    
    return euler_new[0], euler_new[1], euler_new[2]

def signed_perm_matrices(det_keep=1):
    """生成所有 signed permutation 矩阵，保留 det == det_keep（det_keep=+1 -> 24 个）"""
    import numpy as np
    from itertools import permutations, product
    
    mats = []
    for perm in permutations([0, 1, 2]):
        for signs in product([1, -1], repeat=3):
            S = np.zeros((3, 3), dtype=int)
            for i, p in enumerate(perm):
                S[i, p] = signs[i]
            d = int(round(np.linalg.det(S)))
            if d == det_keep:
                mats.append(S)
    return mats

def generate_equivalent_box_representations(x, y, z, l, w, h, roll, pitch, yaw):
    """
    生成box的所有等价表示（基于test_box2.py的24个等效box）
    
    每个box有24个等效表示：
    - 使用signed permutation matrices生成24个orientation-preserving等效box
    
    Args:
        x, y, z: 位置坐标
        l, w, h: 长度、宽度、高度
        roll, pitch, yaw: 旋转角度（弧度）
    
    Returns:
        List of equivalent boxes: [(x, y, z, l, w, h, roll, pitch, yaw), ...]
    """
    import math
    import numpy as np
    from scipy.spatial.transform import Rotation as R_scipy
    
    equivalent_boxes = []
    
    # 获取原始旋转矩阵和半尺寸
    half_sizes = np.array([l, w, h]) / 2.0
    Rmat = R_scipy.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    
    # 生成24个signed permutation matrices
    Ss = signed_perm_matrices(det_keep=1)  # 24个矩阵
    
    for S in Ss:
        # R' = R * S^T
        Rprime = Rmat @ S.T
        half_prime = np.abs(S @ half_sizes)  # positive half sizes
        
        # 转换回欧拉角
        euler_new = R_scipy.from_matrix(Rprime).as_euler('xyz')
        
        # 转换为我们的格式
        new_l, new_w, new_h = half_prime[0] * 2, half_prime[1] * 2, half_prime[2] * 2
        new_roll, new_pitch, new_yaw = euler_new[0], euler_new[1], euler_new[2]
        
        # 归一化角度到[-π, π]
        r_norm = normalize_angle(new_roll)
        p_norm = normalize_angle(new_pitch)
        y_norm = normalize_angle(new_yaw)
        
        # 添加等效box
        equivalent_boxes.append((x, y, z, new_l, new_w, new_h, r_norm, p_norm, y_norm))
    
    return equivalent_boxes

def equivalent_box_from_columns(position, rpy, size, perm):
    """
    计算等效box（完全基于test_box.py的实现）
    参数:
        position: (3,) array-like
        rpy: (3,) array-like, radians, euler xyz (roll,pitch,yaw)
        size: (3,) array-like, [l, w, h]
        perm: tuple of 3 ints, 表示 new_axis_i 对应 old_axis perm[i]
              例如 perm=(1,0,2) 表示 new_x=old_y, new_y=old_x, new_z=old_z
    返回:
        center: (3,)
        half_sizes: (3,)  (注意 Rerun 的 Boxes3D 接受的是 half_sizes)
        quat_xyzw: (4,) 四元数按 [x,y,z,w]
        euler_xyz: (3,) rpy（便于打印 / 调试）
        was_reflection_fix: bool (是否做了 -1 列修正)
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R_scipy
    
    pos = np.array(position, dtype=float)
    size = np.array(size, dtype=float)
    Rm = R_scipy.from_euler("xyz", rpy).as_matrix()            # world_from_local 的矩阵，列是局部 x,y,z 在世界坐标里的向量

    # 新旋转矩阵的列直接从原矩阵按 permute 取列
    Rnew = Rm[:, list(perm)].copy()

    # 如果置换导致左手系（det < 0），把第 3 列乘 -1 修正为右手系
    was_reflection_fix = False
    if np.linalg.det(Rnew) < 0:
        Rnew[:, 2] *= -1.0
        was_reflection_fix = True

    new_size = size[list(perm)]
    half_sizes = new_size / 2.0

    quat_xyzw = R_scipy.from_matrix(Rnew).as_quat()  # SciPy 保证是 [x,y,z,w]
    euler_xyz = R_scipy.from_matrix(Rnew).as_euler("xyz")

    return pos, half_sizes.tolist(), quat_xyzw.tolist(), euler_xyz.tolist(), was_reflection_fix

def normalize_angle(angle):
    """将角度归一化到[-π, π]范围（弧度）"""
    import math
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def quaternion_to_euler_angles(quaternion):
    """
    将四元数转换为欧拉角 (roll, pitch, yaw)
    
    Args:
        quaternion: 四元数 [x, y, z, w] 或 (N, 4) 数组
        
    Returns:
        euler_angles: 欧拉角 [roll, pitch, yaw] 或 (N, 3) 数组，单位为弧度
    """
    if quaternion.ndim == 1:
        # 单个四元数
        x, y, z, w = quaternion
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    else:
        # 批量四元数
        x, y, z, w = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))  # 限制在[-1,1]范围内
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.column_stack([roll, pitch, yaw])

def load_config(config_path: str = "data_config.yaml") -> Dict:
    """加载数据配置文件 - 已废弃，保留以防兼容性问题"""
    print(f"⚠️  警告: load_config已废弃，现在使用统一配置文件training_config.yaml")
    return {}

class Box3DDataset(Dataset):
    """3D Box检测数据集"""
    
    def __init__(
        self,
        data_root: str,
        config_path: str = "data_config.yaml",  # 已废弃，保留以防兼容性
        stage: str = "train",  # train, val, test
        # 必需参数（从训练配置传入）
        max_boxes: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        continuous_ranges: Optional[Dict] = None,
        augmentation_config: Optional[Dict] = None,
        # 其他可选参数
        augment: Optional[bool] = None,
        augment_intensity: Optional[float] = None,
        pad_id: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            data_root: processed数据根目录路径
            config_path: 配置文件路径（已废弃，保留以防兼容性）
            stage: 训练阶段 (train/val/test)
            max_boxes: 最大box数量
            image_size: 图像尺寸
            continuous_ranges: 连续值范围配置
            augmentation_config: 数据增强配置
            其他参数: 可选参数会覆盖默认设置
        """
        self.data_root = Path(data_root)
        self.stage = stage
        
        # 🔧 修复：优先使用传入的参数，而不是读取不存在的配置文件
        if config_path != "data_config.yaml":
            print(f"⚠️  警告: config_path参数已废弃，现在使用统一配置文件")
        
        # 设置参数（优先级：直接参数 > 默认值）
        self.max_boxes = max_boxes if max_boxes is not None else 10
        self.image_size = image_size if image_size is not None else (640, 640)
        self.pad_id = pad_id if pad_id is not None else -1
        
        # 🔧 修复：数据增强设置 - 优先使用传入的augmentation_config
        if augmentation_config is not None:
            self.aug_config = augmentation_config
            self.augment = augment if augment is not None else augmentation_config.get('enabled', False)
            base_intensity = augmentation_config.get('intensity', 1.0)
            self.augment_intensity = augment_intensity if augment_intensity is not None else base_intensity
        else:
            # 回退到默认值
            self.aug_config = {}
            self.augment = augment if augment is not None else False
            self.augment_intensity = augment_intensity if augment_intensity is not None else 1.0
        
        # 🔧 修复：连续值范围 - 优先使用传入的continuous_ranges
        if continuous_ranges is not None:
            self.continuous_range_x = continuous_ranges.get('x', [0.5, 2.5])
            self.continuous_range_y = continuous_ranges.get('y', [-2, 2])
            self.continuous_range_z = continuous_ranges.get('z', [-1.5, 1.5])
            self.continuous_range_w = continuous_ranges.get('w', [0.3, 0.7])
            self.continuous_range_h = continuous_ranges.get('h', [0.3, 0.7])
            self.continuous_range_l = continuous_ranges.get('l', [0.3, 0.7])
            # 旋转范围 - 支持两种格式
            import math
            if 'rotation' in continuous_ranges:
                # 新格式：rotation是3x2数组 [[roll_range], [pitch_range], [yaw_range]]
                rotation_ranges = continuous_ranges['rotation']
                roll_range = rotation_ranges[0]  # [min, max]
                pitch_range = rotation_ranges[1]  # [min, max]
                yaw_range = rotation_ranges[2]  # [min, max]
            else:
                # 旧格式：分别的roll, pitch, yaw键
                roll_range = continuous_ranges.get('roll', [-90.0, 90.0])
                pitch_range = continuous_ranges.get('pitch', [-90.0, 90.0])
                yaw_range = continuous_ranges.get('yaw', [-90.0, 90.0])
            
            # 从角度转换为弧度（config用角度，模型用弧度）
            self.continuous_range_roll = [math.radians(roll_range[0]), math.radians(roll_range[1])]
            self.continuous_range_pitch = [math.radians(pitch_range[0]), math.radians(pitch_range[1])]
            self.continuous_range_yaw = [math.radians(yaw_range[0]), math.radians(yaw_range[1])]
        else:
            # 使用默认值
            self.continuous_range_x = [0.5, 2.5]
            self.continuous_range_y = [-2, 2]
            self.continuous_range_z = [-1.5, 1.5]
            self.continuous_range_w = [0.3, 0.7]
            self.continuous_range_h = [0.3, 0.7]
            self.continuous_range_l = [0.3, 0.7]
            # 旋转范围默认值
            # 默认角度范围转换为弧度
            import math
            self.continuous_range_roll = [math.radians(-90.0), math.radians(90.0)]
            self.continuous_range_pitch = [math.radians(-90.0), math.radians(90.0)]
            self.continuous_range_yaw = [math.radians(-90.0), math.radians(90.0)]
        
        # 打印配置信息
        if self.augment:
            print(f"🎨 数据增强已启用 (强度: {self.augment_intensity})")
        else:
            print(f"❌ 数据增强已禁用")
        
        # 扫描数据文件
        self.samples = self._scan_data()
        
        print(f"Box3DDataset: 找到 {len(self.samples)} 个样本")
        print(f"  数据根目录: {self.data_root}")
        print(f"  图像尺寸: {self.image_size}")
        print(f"  最大box数: {self.max_boxes}")
        print(f"  数据增强: {self.augment} (强度: {self.augment_intensity})")
    
    def _scan_data(self) -> List[Dict]:
        """扫描数据目录，返回样本列表"""
        samples = []
        
        # 根据stage确定要加载的数据子目录
        if self.stage == "train":
            data_subdir = self.data_root / "train"
        elif self.stage == "val":
            data_subdir = self.data_root / "val"
        elif self.stage == "test":
            data_subdir = self.data_root / "test"
        else:
            # 默认加载train文件夹
            data_subdir = self.data_root / "train"
            print(f"警告: 未知的stage '{self.stage}'，默认加载train文件夹")
        
        # 检查数据子目录是否存在
        if not data_subdir.exists():
            print(f"错误: 数据子目录不存在: {data_subdir}")
            return samples
        
        # 遍历数据子目录中的所有编号文件夹
        for folder in sorted(data_subdir.iterdir()):
            if not folder.is_dir():
                continue
            
            # 接受纯数字和带下划线的数字文件夹名 (如 0000, 0000_2)
            folder_name = folder.name
            if not (folder_name.isdigit() or (folder_name.replace('_', '').isdigit() and '_' in folder_name)):
                continue
                
            folder_name = folder.name
            npz_file = folder / f"{folder_name}.npz"
            json_file = folder / f"{folder_name}.json"
            
            # 检查文件是否存在
            if npz_file.exists() and json_file.exists():
                samples.append({
                    'folder_name': folder_name,
                    'npz_file': str(npz_file),
                    'json_file': str(json_file)
                })
            else:
                print(f"警告: 文件夹 {folder_name} 缺少必要文件")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_rgbxyz_image(self, npz_file: str) -> np.ndarray:
        """加载RGBXYZ图像数据"""
        try:
            data = np.load(npz_file)
            rgbxyz = data['rgbxyz']  # Shape: (H, W, 6) - [R, G, B, X, Y, Z]
            data.close()
            return rgbxyz
        except Exception as e:
            print(f"加载图像失败 {npz_file}: {e}")
            # 返回空图像
            return np.zeros((*self.image_size, 6), dtype=np.float32)
    
    def _load_boxes(self, json_file: str) -> List[Dict]:
        """加载3D box标注"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data.get('boxes', [])
        except Exception as e:
            print(f"加载标注失败 {json_file}: {e}")
            return []
    
    def _normalize_coordinates(self, boxes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取box坐标和尺寸，进行范围裁剪，同时将四元数旋转转换为欧拉角，并计算所有等价表示"""
        if not boxes:
            # 返回空数组
            return (np.array([]), np.array([]), np.array([]), 
                   np.array([]), np.array([]), np.array([]), 
                   np.array([]), np.array([]), np.array([]))  # roll, pitch, yaw
        
        # 提取坐标、尺寸和旋转
        positions = np.array([box['position'] for box in boxes])  # (N, 3)
        sizes = np.array([box['size'] for box in boxes])  # (N, 3)
        rotations = np.array([box['rotation'] for box in boxes])  # (N, 4) - quaternion [x,y,z,w]
        
        # 分离xyz坐标和whl尺寸
        x = positions[:, 0]
        y = positions[:, 1] 
        z = positions[:, 2]
        w = sizes[:, 1]  # width
        h = sizes[:, 2]  # height
        l = sizes[:, 0]  # length
        
        # 🔧 修复：不做归一化，保持原始数值
        # 只进行范围检查和裁剪到有效范围
        x_clipped = np.clip(x, self.continuous_range_x[0], self.continuous_range_x[1])
        y_clipped = np.clip(y, self.continuous_range_y[0], self.continuous_range_y[1])
        z_clipped = np.clip(z, self.continuous_range_z[0], self.continuous_range_z[1])
        w_clipped = np.clip(w, self.continuous_range_w[0], self.continuous_range_w[1])
        h_clipped = np.clip(h, self.continuous_range_h[0], self.continuous_range_h[1])
        l_clipped = np.clip(l, self.continuous_range_l[0], self.continuous_range_l[1])
        
        # 🔍 添加日志：检查原始数据和裁剪后的尺寸
        for i in range(len(boxes)):
            original_w, original_h, original_l = w[i], h[i], l[i]
            clipped_w, clipped_h, clipped_l = w_clipped[i], h_clipped[i], l_clipped[i]
            
            if original_w <= 0 or original_h <= 0 or original_l <= 0:
                print(f"🚨 数据预处理中发现无效原始尺寸 - Box {i}:")
                print(f"   原始尺寸: w={original_w:.6f}, h={original_h:.6f}, l={original_l:.6f}")
                print(f"   裁剪后尺寸: w={clipped_w:.6f}, h={clipped_h:.6f}, l={clipped_l:.6f}")
                print(f"   原始位置: x={x[i]:.6f}, y={y[i]:.6f}, z={z[i]:.6f}")
                print(f"   裁剪后位置: x={x_clipped[i]:.6f}, y={y_clipped[i]:.6f}, z={z_clipped[i]:.6f}")
            
            if clipped_w <= 0 or clipped_h <= 0 or clipped_l <= 0:
                print(f"🚨 数据预处理后仍有无效尺寸 - Box {i}:")
                print(f"   裁剪后尺寸: w={clipped_w:.6f}, h={clipped_h:.6f}, l={clipped_l:.6f}")
                print(f"   尺寸范围: w_range={self.continuous_range_w}, h_range={self.continuous_range_h}, l_range={self.continuous_range_l}")
        
        # 将四元数转换为欧拉角
        euler_angles = quaternion_to_euler_angles(rotations)  # (N, 3) - [roll, pitch, yaw]
        roll = euler_angles[:, 0]
        pitch = euler_angles[:, 1]
        yaw = euler_angles[:, 2]
        
        # 裁剪旋转角度到有效范围
        roll_clipped = np.clip(roll, self.continuous_range_roll[0], self.continuous_range_roll[1])
        pitch_clipped = np.clip(pitch, self.continuous_range_pitch[0], self.continuous_range_pitch[1])
        yaw_clipped = np.clip(yaw, self.continuous_range_yaw[0], self.continuous_range_yaw[1])
        
        # 计算所有等价表示
        all_equivalent_boxes = []
        for i in range(len(x_clipped)):
            equiv_boxes = generate_equivalent_box_representations(
                x_clipped[i], y_clipped[i], z_clipped[i],
                l_clipped[i], w_clipped[i], h_clipped[i],
                roll_clipped[i], pitch_clipped[i], yaw_clipped[i]
            )
            all_equivalent_boxes.append(equiv_boxes)
        
        # 暂时返回原始表示，等价表示将在训练时使用
        return x_clipped, y_clipped, z_clipped, w_clipped, h_clipped, l_clipped, roll_clipped, pitch_clipped, yaw_clipped
    
    def _pad_sequences(self, x, y, z, w, h, l, roll, pitch, yaw) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将序列pad到固定长度"""
        current_len = len(x)
        
        if current_len == 0:
            # 如果没有box，返回全padding的tensor
            return (torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32))
        
        if current_len > self.max_boxes:
            # 如果超过最大长度，随机采样
            indices = np.random.choice(current_len, self.max_boxes, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
            w = w[indices]
            h = h[indices]
            l = l[indices]
            roll = roll[indices]
            pitch = pitch[indices]
            yaw = yaw[indices]
        else:
            # 如果不足最大长度，进行padding
            pad_len = self.max_boxes - current_len
            x = np.concatenate([x, np.full(pad_len, self.pad_id)])
            y = np.concatenate([y, np.full(pad_len, self.pad_id)])
            z = np.concatenate([z, np.full(pad_len, self.pad_id)])
            w = np.concatenate([w, np.full(pad_len, self.pad_id)])
            h = np.concatenate([h, np.full(pad_len, self.pad_id)])
            l = np.concatenate([l, np.full(pad_len, self.pad_id)])
            
            roll = np.concatenate([roll, np.full(pad_len, self.pad_id)])
            pitch = np.concatenate([pitch, np.full(pad_len, self.pad_id)])
            yaw = np.concatenate([yaw, np.full(pad_len, self.pad_id)])
        
        return (torch.from_numpy(x).float(),
               torch.from_numpy(y).float(),
               torch.from_numpy(z).float(),
               torch.from_numpy(w).float(),
               torch.from_numpy(h).float(),
               torch.from_numpy(l).float(),
               torch.from_numpy(roll).float(),
               torch.from_numpy(pitch).float(),
               torch.from_numpy(yaw).float())
    
    def _augment_data(self, rgbxyz: np.ndarray, boxes: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """数据增强 - 包括图像噪点和点云噪声"""
        if not self.augment:
            return rgbxyz, boxes
        
        # 注意：数据是从文件新加载的，无需复制
        # 但为了安全起见，仍然复制boxes（因为它们包含嵌套的numpy数组）
        boxes = [box.copy() for box in boxes]
        for box in boxes:
            box['position'] = box['position'].copy()
            box['size'] = box['size'].copy()
        
        # 2. RGB图像增强
        rgb_enh = self.aug_config.get('rgb_enhancement', {})
        
        # 亮度调整
        brightness_cfg = rgb_enh.get('brightness', {})
        if brightness_cfg.get('enabled', True) and random.random() < brightness_cfg.get('probability', 0.7):
            factor_range = brightness_cfg.get('factor_range', [0.8, 1.2])
            brightness_factor = random.uniform(*factor_range)
            rgbxyz[:, :, :3] = np.clip(rgbxyz[:, :, :3] * brightness_factor, 0, 255)
        
        # 对比度调整
        contrast_cfg = rgb_enh.get('contrast', {})
        if contrast_cfg.get('enabled', True) and random.random() < contrast_cfg.get('probability', 0.7):
            factor_range = contrast_cfg.get('factor_range', [0.8, 1.2])
            contrast_factor = random.uniform(*factor_range)
            mean_rgb = rgbxyz[:, :, :3].mean()
            rgbxyz[:, :, :3] = np.clip((rgbxyz[:, :, :3] - mean_rgb) * contrast_factor + mean_rgb, 0, 255)
        
        # 色调偏移
        hue_cfg = rgb_enh.get('hue', {})  # 修正配置键名
        if hue_cfg.get('enabled', True) and random.random() < hue_cfg.get('probability', 0.7):
            shift_range = hue_cfg.get('shift_range', [-10, 10])
            hue_shift = random.uniform(*shift_range)
            rgbxyz[:, :, :3] = np.clip(rgbxyz[:, :, :3] + hue_shift, 0, 255)
        
        # 3. RGB图像噪点
        rgb_noise = self.aug_config.get('rgb_noise', {})
        
        # 高斯噪声
        gaussian_cfg = rgb_noise.get('gaussian', {})
        if gaussian_cfg.get('enabled', True) and random.random() < gaussian_cfg.get('probability', 0.6):
            std_range = gaussian_cfg.get('std_range', [2, 8])
            noise_std = random.uniform(*std_range) * self.augment_intensity
            gaussian_noise = np.random.normal(0, noise_std, rgbxyz[:, :, :3].shape)
            rgbxyz[:, :, :3] = np.clip(rgbxyz[:, :, :3] + gaussian_noise, 0, 255)
        
        # 椒盐噪声
        salt_pepper_cfg = rgb_noise.get('salt_pepper', {})
        if salt_pepper_cfg.get('enabled', True) and random.random() < salt_pepper_cfg.get('probability', 0.4):
            ratio_range = salt_pepper_cfg.get('ratio_range', [0.001, 0.01])
            noise_ratio = random.uniform(*ratio_range) * self.augment_intensity
            mask = np.random.random(rgbxyz[:, :, :3].shape) < noise_ratio
            rgbxyz[:, :, :3][mask] = np.random.choice([0, 255], size=mask.sum())
        
        # 斑点噪声
        speckle_cfg = rgb_noise.get('speckle', {})
        if speckle_cfg.get('enabled', True) and random.random() < speckle_cfg.get('probability', 0.3):
            std_range = speckle_cfg.get('std_range', [0.05, 0.15])
            noise_std = random.uniform(*std_range) * self.augment_intensity
            speckle_noise = np.random.normal(0, noise_std, rgbxyz[:, :, :3].shape)
            rgbxyz[:, :, :3] = np.clip(rgbxyz[:, :, :3] * (1 + speckle_noise), 0, 255)
        

            rgbxyz[:, :, :3] = np.clip(rgbxyz[:, :, :3] * (1 + speckle_noise), 0, 255)
        
        # 4. 点云(XYZ)噪声
        pc_config = self.aug_config.get('point_cloud', {})
        
        # XYZ坐标高斯噪声
        gaussian_cfg = pc_config.get('gaussian_noise', {})
        if gaussian_cfg.get('enabled', True) and random.random() < gaussian_cfg.get('probability', 0.7):
            std_ranges = gaussian_cfg.get('std_ranges', {'x': [0.01, 0.05], 'y': [0.01, 0.05], 'z': [0.005, 0.03]})
            xyz_noise_std = [
                random.uniform(*std_ranges.get('x', [0.01, 0.05])) * self.augment_intensity,
                random.uniform(*std_ranges.get('y', [0.01, 0.05])) * self.augment_intensity,
                random.uniform(*std_ranges.get('z', [0.005, 0.03])) * self.augment_intensity
            ]
            
            for i, std in enumerate(xyz_noise_std):
                noise = np.random.normal(0, std, rgbxyz[:, :, 3+i].shape)
                rgbxyz[:, :, 3+i] += noise
        
        # 点云随机丢失
        dropout_cfg = pc_config.get('dropout', {})
        if dropout_cfg.get('enabled', True) and random.random() < dropout_cfg.get('probability', 0.5):
            ratio_range = dropout_cfg.get('ratio_range', [0.001, 0.01])
            dropout_ratio = random.uniform(*ratio_range) * self.augment_intensity
            dropout_mask = np.random.random(rgbxyz[:, :, 3:6].shape[:2]) < dropout_ratio
            rgbxyz[dropout_mask, 3:6] = 0
        
        # 深度量化噪声
        quant_cfg = pc_config.get('quantization', {})
        if quant_cfg.get('enabled', True) and random.random() < quant_cfg.get('probability', 0.4):
            step_range = quant_cfg.get('step_range', [0.005, 0.02])
            depth_quantization = random.uniform(*step_range) * self.augment_intensity
            # 避免除零错误
            if depth_quantization > 0:
                rgbxyz[:, :, 5] = np.round(rgbxyz[:, :, 5] / depth_quantization) * depth_quantization
        
        # 系统性偏移
        bias_cfg = pc_config.get('coordinate_bias', {})
        if bias_cfg.get('enabled', True) and random.random() < bias_cfg.get('probability', 0.3):
            bias_ranges = bias_cfg.get('bias_ranges', {'x': [-0.02, 0.02], 'y': [-0.02, 0.02], 'z': [-0.01, 0.01]})
            xyz_bias = [
                random.uniform(*bias_ranges.get('x', [-0.02, 0.02])) * self.augment_intensity,
                random.uniform(*bias_ranges.get('y', [-0.02, 0.02])) * self.augment_intensity,
                random.uniform(*bias_ranges.get('z', [-0.01, 0.01])) * self.augment_intensity
            ]
            for i, bias in enumerate(xyz_bias):
                rgbxyz[:, :, 3+i] += bias
        
        # 5. 3D Box标注扰动
        box_pert = self.aug_config.get('box_augmentation', {})
        if box_pert.get('enabled', True) and random.random() < box_pert.get('probability', 0.2):
            pos_noise_cfg = box_pert.get('position_noise', {})
            size_noise_cfg = box_pert.get('size_noise', {})
            
            for box in boxes:
                # 位置扰动
                if pos_noise_cfg.get('enabled', True):
                    noise_ranges = pos_noise_cfg.get('noise_ranges', {'x': [-0.02, 0.02], 'y': [-0.02, 0.02], 'z': [-0.01, 0.01]})
                    position_noise = [
                        random.uniform(*noise_ranges.get('x', [-0.02, 0.02])),
                        random.uniform(*noise_ranges.get('y', [-0.02, 0.02])),
                        random.uniform(*noise_ranges.get('z', [-0.01, 0.01]))
                    ]
                    for i, noise in enumerate(position_noise):
                        box['position'][i] += noise
                        # 确保在有效范围内
                        if i == 0:  # X
                            box['position'][i] = np.clip(box['position'][i], 
                                                       self.continuous_range_x[0], 
                                                       self.continuous_range_x[1])
                        elif i == 1:  # Y
                            box['position'][i] = np.clip(box['position'][i], 
                                                       self.continuous_range_y[0], 
                                                       self.continuous_range_y[1])
                        elif i == 2:  # Z
                            box['position'][i] = np.clip(box['position'][i], 
                                                       self.continuous_range_z[0], 
                                                       self.continuous_range_z[1])
                
                # 尺寸扰动
                if size_noise_cfg.get('enabled', True):
                    noise_ranges = size_noise_cfg.get('noise_ranges', {'w': [-0.01, 0.01], 'h': [-0.01, 0.01], 'l': [-0.01, 0.01]})
                    size_noise = [
                        random.uniform(*noise_ranges.get('w', [-0.01, 0.01])),
                        random.uniform(*noise_ranges.get('h', [-0.01, 0.01])),
                        random.uniform(*noise_ranges.get('l', [-0.01, 0.01]))
                    ]
                    for i, noise in enumerate(size_noise):
                        box['size'][i] += noise
                        # 确保尺寸为正且在有效范围内
                        if i == 0:  # width
                            box['size'][i] = np.clip(box['size'][i], 
                                                    self.continuous_range_w[0], 
                                                    self.continuous_range_w[1])
                        elif i == 1:  # height
                            box['size'][i] = np.clip(box['size'][i], 
                                                    self.continuous_range_h[0], 
                                                    self.continuous_range_h[1])
                        elif i == 2:  # length
                            box['size'][i] = np.clip(box['size'][i], 
                                                    self.continuous_range_l[0], 
                                                    self.continuous_range_l[1])
        
        return rgbxyz, boxes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 加载RGBXYZ图像
        rgbxyz = self._load_rgbxyz_image(sample['npz_file'])  # (H, W, 6)
        
        # 加载3D box标注
        boxes = self._load_boxes(sample['json_file'])
        
        # 数据增强
        rgbxyz, boxes = self._augment_data(rgbxyz, boxes)
        
        # 归一化坐标并提取旋转信息
        x, y, z, w, h, l, roll, pitch, yaw = self._normalize_coordinates(boxes)
        
        # 计算所有等价表示
        equivalent_boxes = []
        for i in range(len(x)):
            equiv_boxes = generate_equivalent_box_representations(
                x[i], y[i], z[i], l[i], w[i], h[i], roll[i], pitch[i], yaw[i]
            )
            equivalent_boxes.append(equiv_boxes)
        
        # Pad序列到固定长度
        x_padded, y_padded, z_padded, w_padded, h_padded, l_padded, roll_padded, pitch_padded, yaw_padded = self._pad_sequences(x, y, z, w, h, l, roll, pitch, yaw)
        
        # 🔧 修复：等效box也需要padding到固定长度
        # 为padding位置添加空的等效box列表
        while len(equivalent_boxes) < self.max_boxes:
            equivalent_boxes.append([])  # 空列表表示padding位置
        
        # 转换图像格式：(H, W, 6) -> (6, H, W)
        rgbxyz_tensor = torch.from_numpy(rgbxyz).permute(2, 0, 1).float()
        
        # 归一化RGB通道到[0,1]
        rgbxyz_tensor[:3] = rgbxyz_tensor[:3] / 255.0
        
        # 🔧 修复：不归一化XYZ点云通道，保持原始数值
        # 只进行范围裁剪，确保在有效范围内
        rgbxyz_tensor[3] = torch.clamp(rgbxyz_tensor[3], self.continuous_range_x[0], self.continuous_range_x[1])  # X
        rgbxyz_tensor[4] = torch.clamp(rgbxyz_tensor[4], self.continuous_range_y[0], self.continuous_range_y[1])  # Y  
        rgbxyz_tensor[5] = torch.clamp(rgbxyz_tensor[5], self.continuous_range_z[0], self.continuous_range_z[1])  # Z
        
        return {
            'image': rgbxyz_tensor,  # (6, H, W) - RGBXYZ
            'x': x_padded,          # (max_boxes,)
            'y': y_padded,          # (max_boxes,)
            'z': z_padded,          # (max_boxes,)
            'w': w_padded,          # (max_boxes,)
            'h': h_padded,          # (max_boxes,)
            'l': l_padded,          # (max_boxes,)
            'roll': roll_padded,     # (max_boxes,) - 欧拉角旋转
            'pitch': pitch_padded,   # (max_boxes,)
            'yaw': yaw_padded,      # (max_boxes,)
            'equivalent_boxes': equivalent_boxes,  # 等价表示列表
            'folder_name': sample['folder_name']  # 用于调试
        }

def create_dataloader(
    data_root: str,
    config_path: str = "data_config.yaml",  # 已废弃，保留以防兼容性
    stage: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    drop_last: Optional[bool] = None,
    **kwargs
) -> DataLoader:
    """创建DataLoader - 现在使用直接参数而不是配置文件"""
    
    # 🔧 修复：直接使用传入的参数，而不是读取配置文件
    if config_path != "data_config.yaml":
        print(f"⚠️  警告: config_path参数已废弃，现在使用直接参数传递")
    
    # 设置默认值
    batch_size = batch_size if batch_size is not None else 8
    num_workers = num_workers if num_workers is not None else 4
    pin_memory = pin_memory if pin_memory is not None else True
    prefetch_factor = prefetch_factor if prefetch_factor is not None else 2
    persistent_workers = persistent_workers if persistent_workers is not None else True
    
    # 根据阶段设置默认行为
    shuffle = shuffle if shuffle is not None else (stage == 'train')
    drop_last = drop_last if drop_last is not None else (stage == 'train')
    
    dataset = Box3DDataset(
        data_root=data_root,
        config_path=config_path,
        stage=stage,
        **kwargs
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        collate_fn=custom_collate_fn  # 添加自定义collate函数
    )
    
    return dataloader

# 测试代码 - 已更新为使用统一配置
if __name__ == "__main__":
    print("⚠️  注意：dataloader_3d.py的测试代码已过时，请使用train.py进行训练")
    print("现在所有配置都通过training_config.yaml统一管理")
    
    # 简单测试（使用默认参数）
    data_root = "sim_data"  # 根据实际路径调整
    
    print("=== 基础功能测试 ===")
    # 创建训练数据集（使用默认参数）
    train_dataset = Box3DDataset(data_root, stage="train")
    
    if len(train_dataset) > 0:
        # 测试单个样本
        sample = train_dataset[0]
        print("\n=== 训练样本测试 ===")
        print(f"图像形状: {sample['image'].shape}")
        print(f"x坐标: {sample['x'][:5]}...")  # 只显示前5个
        print(f"y坐标: {sample['y'][:5]}...")
        print(f"z坐标: {sample['z'][:5]}...")
        print(f"w尺寸: {sample['w'][:5]}...")
        print(f"h尺寸: {sample['h'][:5]}...")
        print(f"l尺寸: {sample['l'][:5]}...")
        print(f"文件夹: {sample['folder_name']}")
        
        print(f"✅ 数据集可用，包含 {len(train_dataset)} 个样本")
        print(f"🎨 数据增强状态: {'启用' if train_dataset.augment else '禁用'}")
        print(f"📏 最大boxes数量: {train_dataset.max_boxes}")
        print(f"🖼️ 图像尺寸: {train_dataset.image_size}")
        
    else:
        print("数据集为空，请检查数据路径！") 