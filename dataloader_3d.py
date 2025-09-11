# -*- coding: utf-8 -*-
"""
3D Box Detection DataLoader
加载processed数据：RGBXYZ图像(npz) + 3D Box标注(json)
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import random
from collections.abc import Sequence, Mapping

def load_config(config_path: str = "data_config.yaml") -> Dict:
    """加载数据配置文件 - 已废弃，保留以防兼容性问题"""
    print(f"⚠️  警告: load_config已废弃，现在使用统一配置文件training_config.yaml")
    return {}

# 从segment_dataset.py复制的变换类
class GridSample(object):
    def __init__(self, grid_size=0.05, mode="train"):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec
        # 为所有模式设置keys，包括val模式
        self.keys = ("coord",)

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        grid_coord = np.floor(scaled_coord).astype(int)
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        scaled_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)
        key = self.hash(grid_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        idx_select = (
            np.cumsum(np.insert(count, 0, 0)[0:-1])
            + np.random.randint(0, count.max(), count.size) % count
        )
        idx_unique = idx_sort[idx_select]
        data_dict["grid_coord"] = grid_coord[idx_unique]
        for key in self.keys:
            data_dict[key] = data_dict[key][idx_unique]
        
        # 保持像素坐标映射关系
        if "pixel_coords" in data_dict:
            data_dict["pixel_coords"] = data_dict["pixel_coords"][idx_unique]
        
        return data_dict
    
    @staticmethod
    def fnv_hash_vec(arr):
        assert arr.ndim == 2
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr

class ToTensor(object):
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, str):
            return data
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, bool):
            return torch.from_numpy(data)
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.integer):
            return torch.from_numpy(data).long()
        elif isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.floating):
            return torch.from_numpy(data).float()
        elif isinstance(data, Mapping):
            result = {sub_key: self(item) for sub_key, item in data.items()}
            return result
        elif isinstance(data, Sequence):
            result = [self(item) for item in data]
            return result
        else:
            raise TypeError(f"type {type(data)} cannot be converted to tensor.")

class Collect(object):
    def __init__(self, keys, offset_keys_dict=None, **kwargs):
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        self.keys = keys
        self.offset_keys = offset_keys_dict
        self.kwargs = kwargs

    def __call__(self, data_dict):
        data = dict()
        if isinstance(self.keys, str):
            self.keys = [self.keys]
        
        for key in self.keys:
            if key in data_dict:
                data[key] = data_dict[key]
        
        for key, value in self.offset_keys.items():
            if value in data_dict:
                # offset应该是累积偏移量，对于单个样本，offset就是[0, num_points]
                num_points = data_dict[value].shape[0]
                data[key] = torch.tensor([0, num_points], dtype=torch.long)
        
        for name, keys in self.kwargs.items():
            name = name.replace("_keys", "")
            assert isinstance(keys, Sequence)
            if len(keys) > 3: 
                keys = [keys]
            # 如果只有一个key，直接使用该key的数据
            if len(keys) == 1:
                data[name] = data_dict[keys[0]].float()
            else:
                data[name] = torch.cat([data_dict[key].float() for key in keys], dim=1)
        
        # 保持像素坐标映射关系
        if "pixel_coords" in data_dict:
            data["pixel_coords"] = data_dict["pixel_coords"]
        
        return data

class RandomCrop(object):
    def __init__(self, point_max=80000, continuous_ranges=None, random_fluctuation=0.1):
        self.point_max = point_max
        self.continuous_ranges = continuous_ranges or {
            'x': [0.5, 2.5],
            'y': [-2.0, 2.0], 
            'z': [-1.5, 1.5]
        }
        self.random_fluctuation = random_fluctuation  # 0.1m的随机波动

    def __call__(self, data_dict):
        point_max = self.point_max
        assert "coord" in data_dict.keys()
        
        # 如果点云数量超过限制，进行基于范围的切割
        if data_dict["coord"].shape[0] > point_max:
            # 生成带随机波动的切割范围
            crop_ranges = {}
            for axis, (min_val, max_val) in self.continuous_ranges.items():
                # 添加0.1m的随机波动
                fluctuation = np.random.uniform(-self.random_fluctuation, self.random_fluctuation)
                min_crop = min_val + fluctuation
                max_crop = max_val + fluctuation
                
                # 确保范围仍然合理
                min_crop = max(min_crop, min_val - self.random_fluctuation)
                max_crop = min(max_crop, max_val + self.random_fluctuation)
                
                crop_ranges[axis] = [min_crop, max_crop]
            
            # 根据切割范围过滤点云
            coord = data_dict["coord"]
            valid_mask = np.ones(coord.shape[0], dtype=bool)
            
            # 对每个轴进行范围过滤
            for i, axis in enumerate(['x', 'y', 'z']):
                if axis in crop_ranges:
                    min_val, max_val = crop_ranges[axis]
                    axis_mask = (coord[:, i] >= min_val) & (coord[:, i] <= max_val)
                    valid_mask = valid_mask & axis_mask
            
            # 如果过滤后的点云仍然太多，随机采样
            if valid_mask.sum() > point_max:
                valid_indices = np.where(valid_mask)[0]
                selected_indices = np.random.choice(valid_indices, point_max, replace=False)
                idx_crop = selected_indices
            else:
                # 如果过滤后的点云不够，使用所有有效点
                idx_crop = np.where(valid_mask)[0]
            
            # 应用切割
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "grid_coord" in data_dict.keys():
                data_dict["grid_coord"] = data_dict["grid_coord"][idx_crop]
            
            # 保持像素坐标映射关系
            if "pixel_coords" in data_dict:
                data_dict["pixel_coords"] = data_dict["pixel_coords"][idx_crop]
                
        return data_dict

class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            # 确保至少保留1个点，避免空点云
            keep_count = max(1, int(n * (1 - self.dropout_ratio)))
            idx = np.random.choice(n, keep_count, replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            
            # 保持像素坐标映射关系
            if "pixel_coords" in data_dict:
                data_dict["pixel_coords"] = data_dict["pixel_coords"][idx]
        return data_dict

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
            self.augmentation_config = augmentation_config
            self.aug_config = augmentation_config  # 保持向后兼容
            self.augment = augment if augment is not None else augmentation_config.get('enabled', False)
            base_intensity = augmentation_config.get('intensity', 1.0)
            self.augment_intensity = augment_intensity if augment_intensity is not None else base_intensity
        else:
            # 回退到默认值
            self.augmentation_config = {}
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
        else:
            # 使用默认值
            self.continuous_range_x = [0.5, 2.5]
            self.continuous_range_y = [-2, 2]
            self.continuous_range_z = [-1.5, 1.5]
            self.continuous_range_w = [0.3, 0.7]
            self.continuous_range_h = [0.3, 0.7]
            self.continuous_range_l = [0.3, 0.7]
        
        # 打印配置信息
        if self.augment:
            print(f"🎨 数据增强已启用 (强度: {self.augment_intensity})")
        else:
            print(f"❌ 数据增强已禁用")
        
        # 点云变换，参考segment_dataset.py
        self.grid = GridSample(grid_size=0.05, mode=stage)
        self.totensor = ToTensor()
        
        # 使用continuous_ranges进行点云切割
        # 从augmentation_config中获取裁剪参数
        cropping_config = self.augmentation_config.get('point_cloud', {}).get('cropping', {})
        point_max = cropping_config.get('max_points', 50000)
        random_fluctuation = cropping_config.get('random_fluctuation', 0.1)
        
        self.crop = RandomCrop(
            point_max=point_max,
            continuous_ranges={
                'x': self.continuous_range_x,
                'y': self.continuous_range_y,
                'z': self.continuous_range_z
            },
            random_fluctuation=random_fluctuation
        )
        self.drop = RandomDropout(dropout_ratio=0.3)
        
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
    
    def _normalize_coordinates(self, boxes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """提取box坐标和尺寸，进行范围裁剪，同时提取旋转信息（保持原始物理数值，不归一化）"""
        if not boxes:
            # 返回空数组
            return (np.array([]), np.array([]), np.array([]), 
                   np.array([]), np.array([]), np.array([]), 
                   np.array([]).reshape(0, 4))  # 旋转四元数
        
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
        
        return x_clipped, y_clipped, z_clipped, w_clipped, h_clipped, l_clipped, rotations
    
    def _pad_sequences(self, x, y, z, w, h, l, rotations) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """将序列pad到固定长度"""
        current_len = len(x)
        
        if current_len == 0:
            # 如果没有box，返回全padding的tensor
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数 (x,y,z,w)
            return (torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.full((self.max_boxes,), self.pad_id, dtype=torch.float32),
                   torch.from_numpy(np.tile(identity_quat, (self.max_boxes, 1))).float())
        
        if current_len > self.max_boxes:
            # 如果超过最大长度，随机采样
            indices = np.random.choice(current_len, self.max_boxes, replace=False)
            x = x[indices]
            y = y[indices]
            z = z[indices]
            w = w[indices]
            h = h[indices]
            l = l[indices]
            rotations = rotations[indices]
        else:
            # 如果不足最大长度，进行padding
            pad_len = self.max_boxes - current_len
            x = np.concatenate([x, np.full(pad_len, self.pad_id)])
            y = np.concatenate([y, np.full(pad_len, self.pad_id)])
            z = np.concatenate([z, np.full(pad_len, self.pad_id)])
            w = np.concatenate([w, np.full(pad_len, self.pad_id)])
            h = np.concatenate([h, np.full(pad_len, self.pad_id)])
            l = np.concatenate([l, np.full(pad_len, self.pad_id)])
            
            # 为旋转数据padding使用单位四元数
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 单位四元数 (x,y,z,w)
            pad_rotations = np.tile(identity_quat, (pad_len, 1))
            rotations = np.concatenate([rotations, pad_rotations], axis=0)
        
        return (torch.from_numpy(x).float(),
               torch.from_numpy(y).float(),
               torch.from_numpy(z).float(),
               torch.from_numpy(w).float(),
               torch.from_numpy(h).float(),
               torch.from_numpy(l).float(),
               torch.from_numpy(rotations).float())
    
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
    
    def _convert_xyz_to_point_cloud(self, xyz_data: np.ndarray) -> dict:
        """将XYZ数据转换为点云格式，并保持像素坐标映射关系"""
        H, W, _ = xyz_data.shape
        
        # 创建像素坐标网格
        pixel_y, pixel_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        pixel_coords = np.stack([
            pixel_x.flatten(),  # X像素坐标
            pixel_y.flatten()   # Y像素坐标
        ], axis=1)  # (H*W, 2) - 像素坐标
        
        # 展平并组合坐标 - 使用PointTransformer期望的格式
        coord = np.stack([
            xyz_data[:, :, 0].flatten(),  # X坐标（来自XYZ数据）
            xyz_data[:, :, 1].flatten(),  # Y坐标（来自XYZ数据）
            xyz_data[:, :, 2].flatten()   # Z坐标（来自XYZ数据）
        ], axis=1)  # (H*W, 3) - 这是PointTransformer期望的coord格式
        
        # 返回data_dict格式，参考segment_dataset.py
        data_dict = {
            'coord': coord,  # (N, 3) - XYZ坐标
            'pixel_coords': pixel_coords,  # (N, 2) - 对应的像素坐标 [x, y]
            'name': 'point_cloud'  # 可选的名字
        }
        
        return data_dict
    
    
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
        x, y, z, w, h, l, rotations = self._normalize_coordinates(boxes)
        
        # Pad序列到固定长度
        x_padded, y_padded, z_padded, w_padded, h_padded, l_padded, rotations_padded = self._pad_sequences(x, y, z, w, h, l, rotations)
        
        # 分离RGB和点云数据
        rgb_image = rgbxyz[:, :, :3]  # (H, W, 3) - RGB通道
        xyz_data = rgbxyz[:, :, 3:6]  # (H, W, 3) - XYZ通道
        
        # 转换RGB图像格式：(H, W, 3) -> (3, H, W)
        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float()
        rgb_tensor = rgb_tensor / 255.0  # 归一化RGB通道到[0,1]
        
        # 处理点云数据：将(H, W, 3)转换为点云格式
        point_cloud_dict = self._convert_xyz_to_point_cloud(xyz_data)
        
        # 应用点云变换，参考segment_dataset.py的流程
        if self.stage == "train":
            point_cloud_dict = self.drop(point_cloud_dict)
        point_cloud_dict = self.grid(point_cloud_dict)
        point_cloud_dict = self.crop(point_cloud_dict)
        point_cloud_dict = self.totensor(point_cloud_dict)
        # 手动处理点云数据，不使用Collect类
        # 确保coord和grid_coord存在
        if 'coord' not in point_cloud_dict:
            raise ValueError("point_cloud_dict中缺少'coord'字段")
        if 'grid_coord' not in point_cloud_dict:
            raise ValueError("point_cloud_dict中缺少'grid_coord'字段")
        
        # 生成offset: [0, num_points]
        num_points = point_cloud_dict['coord'].shape[0]
        point_cloud_dict['offset'] = torch.tensor([0, num_points], dtype=torch.long)
        
        # 生成feat: 使用coord作为特征
        point_cloud_dict['feat'] = point_cloud_dict['coord'].float()
        
        # 提取处理后的点云数据
        # point_cloud = point_cloud_dict['coord']  # (N, 3) - XYZ坐标
        
        # 合并RGB图像和点云数据，参考segment_dataset.py的格式
        result = {
            'rgb_image': rgb_tensor,    # (3, H, W) - RGB图像
            'x': x_padded,             # (max_boxes,)
            'y': y_padded,             # (max_boxes,)
            'z': z_padded,          # (max_boxes,)
            'w': w_padded,          # (max_boxes,)
            'h': h_padded,          # (max_boxes,)
            'l': l_padded,          # (max_boxes,)
            'rotations': rotations_padded,  # (max_boxes, 4) - 四元数旋转
            'folder_name': sample['folder_name']  # 用于调试
        }
        # print(f"point_cloud_dict: {point_cloud_dict['grid_coord'].shape}")
        # 添加点云数据字段到结果中
        result.update(point_cloud_dict)
        
        return result

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
        collate_fn=collate_fn  # 使用自定义collate函数
    )
    
    return dataloader

def collate_fn(batch):
    """自定义collate函数，与segment_dataset.py保持一致"""
    # 使用默认的PyTorch collate行为，但处理变长点云数据
    # 固定大小的张量会被自动stack，变长的保持为列表
    
    # 分离固定大小和变长的数据
    fixed_size_data = {}
    variable_size_data = {}
    
    # 固定大小的字段（会被stack）
    fixed_fields = ['rgb_image', 'x', 'y', 'z', 'w', 'h', 'l', 'rotations']
    # 变长的字段（保持为列表）
    variable_fields = ['coord', 'grid_coord', 'feat', 'pixel_coords', 'folder_name']
    
    for field in fixed_fields:
        if field in batch[0]:
            fixed_size_data[field] = [sample[field] for sample in batch]
    
    for field in variable_fields:
        if field in batch[0]:
            variable_size_data[field] = [sample[field] for sample in batch]
    
    # 特殊处理offset字段 - 需要合并成累积偏移量
    offset_data = None
    if 'offset' in batch[0]:
        offsets_list = [sample['offset'] for sample in batch]
        # 合并所有样本的offset，生成累积偏移量
        # offset2bincount期望的格式：[cumulative_points]，不包含开头的0
        cumulative_offset = []
        current_offset = 0
        for offset in offsets_list:
            # offset是[0, num_points]，我们只需要num_points
            num_points = offset[-1].item()
            current_offset += num_points
            cumulative_offset.append(current_offset)
        offset_data = torch.tensor(cumulative_offset, dtype=torch.long)
    
    # 堆叠固定大小的张量
    for field, data_list in fixed_size_data.items():
        fixed_size_data[field] = torch.stack(data_list, dim=0)
    
    # 合并变长的张量 - 将所有样本的点云数据首尾相连
    for field, data_list in variable_size_data.items():
        if field in ['coord', 'grid_coord', 'feat', 'pixel_coords']:
            variable_size_data[field] = torch.cat(data_list, dim=0)
        else:
            # 对于非张量字段，保持为列表
            variable_size_data[field] = data_list
    
    # 合并结果
    result = {**fixed_size_data, **variable_size_data}
    
    # 添加处理后的offset数据
    if offset_data is not None:
        result['offset'] = offset_data
    
    return result

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