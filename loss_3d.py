# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from scipy.spatial.transform import Rotation


class AdaptivePrimitiveTransformer3DLoss(nn.Module):
    """
    自适应3D Primitive Transformer损失函数
    特色：分类损失权重根据IoU自适应调整
    - IoU高 → 分类权重低 (关注精细调优)
    - IoU低 → 分类权重高 (关注粗略定位)
    """
    
    def __init__(
        self,
        # 离散化参数
        num_discrete_x: int = 128,
        num_discrete_y: int = 128, 
        num_discrete_z: int = 128,
        num_discrete_w: int = 64,
        num_discrete_h: int = 64,
        num_discrete_l: int = 64,
        num_discrete_roll: int = 64,  # 新增旋转
        num_discrete_pitch: int = 64,
        num_discrete_yaw: int = 64,
        
        # 连续范围参数
        continuous_range_x: Tuple[float, float] = (0.5, 2.5),
        continuous_range_y: Tuple[float, float] = (-2, 2),
        continuous_range_z: Tuple[float, float] = (-1.5, 1.5),
        continuous_range_w: Tuple[float, float] = (0.3, 0.7),
        continuous_range_h: Tuple[float, float] = (0.3, 0.7),
        continuous_range_l: Tuple[float, float] = (0.3, 0.7),
        continuous_range_roll: Tuple[float, float] = (-1.5708, 1.5708),  # 新增旋转范围（弧度）
        continuous_range_pitch: Tuple[float, float] = (-1.5708, 1.5708),
        continuous_range_yaw: Tuple[float, float] = (-1.5708, 1.5708),
        
        # 基础损失权重
        base_classification_weight: float = 1.0,
        iou_weight: float = 2.0,
        delta_weight: float = 1.0,
        eos_weight: float = 0.5,
        
        # IoU自适应权重参数
        adaptive_classification: bool = True,
        min_classification_weight: float = 0.1,  # IoU很高时的最小分类权重
        max_classification_weight: float = 3.0,  # IoU很低时的最大分类权重
        iou_threshold_high: float = 0.7,         # 高IoU阈值
        iou_threshold_low: float = 0.3,          # 低IoU阈值
        adaptive_delta: bool = True,             # delta权重是否也自适应
        min_delta_weight: float = 0.1,           # Delta权重最小值
        max_delta_weight: float = 2.0,           # Delta权重最大值
        
        # 其他参数
        pad_id: float = -1.0,
        label_smoothing: float = 0.1,
        distance_aware_cls: bool = True,         # 是否使用距离感知分类损失
        distance_alpha: float = 2.0,            # 距离权重参数
        focal_gamma: float = 2.0,                # focal loss参数
        
        # IoU计算参数（简化：只使用投影方法）
        # 移除了复杂的obb_method和voxel_resolution参数
    ):
        super().__init__()
        
        # 保存所有参数
        self.num_discrete_x = num_discrete_x
        self.num_discrete_y = num_discrete_y
        self.num_discrete_z = num_discrete_z
        self.num_discrete_w = num_discrete_w
        self.num_discrete_h = num_discrete_h
        self.num_discrete_l = num_discrete_l
        self.num_discrete_roll = num_discrete_roll  # 新增旋转
        self.num_discrete_pitch = num_discrete_pitch
        self.num_discrete_yaw = num_discrete_yaw
        
        self.continuous_range_x = continuous_range_x
        self.continuous_range_y = continuous_range_y
        self.continuous_range_z = continuous_range_z
        self.continuous_range_w = continuous_range_w
        self.continuous_range_h = continuous_range_h
        self.continuous_range_l = continuous_range_l
        self.continuous_range_roll = continuous_range_roll  # 新增旋转范围
        self.continuous_range_pitch = continuous_range_pitch
        self.continuous_range_yaw = continuous_range_yaw
        
        # 基础权重
        self.base_classification_weight = base_classification_weight
        self.iou_weight = iou_weight
        self.delta_weight = delta_weight
        self.eos_weight = eos_weight
        
        # 自适应权重参数
        self.adaptive_classification = adaptive_classification
        self.adaptive_delta = adaptive_delta
        self.min_classification_weight = min_classification_weight
        self.max_classification_weight = max_classification_weight
        self.iou_threshold_high = iou_threshold_high
        self.iou_threshold_low = iou_threshold_low
        self.min_delta_weight = min_delta_weight
        self.max_delta_weight = max_delta_weight
        
        # 其他参数
        self.pad_id = pad_id
        self.label_smoothing = label_smoothing
        self.distance_aware_cls = distance_aware_cls
        self.distance_alpha = distance_alpha
        self.focal_gamma = focal_gamma
        
        # 损失函数
        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing
        )
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        
        # IoU计算参数（简化：只使用投影方法）
        self.iou_loss_epsilon = 1e-7
        
        print(f"AdaptivePrimitiveTransformer3DLoss初始化:")
        print(f"  自适应分类权重: {adaptive_classification}")
        if adaptive_classification:
            print(f"    权重范围: [{min_classification_weight}, {max_classification_weight}]")
            print(f"    IoU阈值: 低={iou_threshold_low}, 高={iou_threshold_high}")
        print(f"  距离感知分类: {distance_aware_cls}")
        print(f"  基础权重 - 分类:{base_classification_weight}, IoU:{iou_weight}, Delta:{delta_weight}")
    
    def compute_adaptive_weights(self, mean_iou: torch.Tensor) -> Dict[str, float]:
        """
        根据IoU计算自适应权重
        Args:
            mean_iou: 当前batch的平均IoU
        Returns:
            adaptive_weights: 包含各种损失的自适应权重
        """
        weights = {}
        
        if self.adaptive_classification:
            # 分类权重：IoU越高，权重越低
            if mean_iou >= self.iou_threshold_high:
                # 高IoU：使用最小分类权重
                cls_weight = self.min_classification_weight
            elif mean_iou <= self.iou_threshold_low:
                # 低IoU：使用最大分类权重
                cls_weight = self.max_classification_weight
            else:
                # 中等IoU：线性插值
                ratio = (mean_iou - self.iou_threshold_low) / (self.iou_threshold_high - self.iou_threshold_low)
                cls_weight = self.max_classification_weight - ratio * (self.max_classification_weight - self.min_classification_weight)
            
            weights['classification'] = cls_weight
        else:
            weights['classification'] = self.base_classification_weight
        
        if self.adaptive_delta:
            # Delta权重：IoU越高，权重越高（细节优化更重要）
            if mean_iou >= self.iou_threshold_high:
                # 高IoU：使用最大delta权重
                delta_weight = self.max_delta_weight
            elif mean_iou <= self.iou_threshold_low:
                # 低IoU：使用最小delta权重
                delta_weight = self.min_delta_weight
            else:
                # 中等IoU：线性插值
                ratio = (mean_iou - self.iou_threshold_low) / (self.iou_threshold_high - self.iou_threshold_low)
                delta_weight = self.min_delta_weight + ratio * (self.max_delta_weight - self.min_delta_weight)
            
            weights['delta'] = delta_weight
        else:
            weights['delta'] = self.delta_weight
        
        # IoU权重保持不变
        weights['iou'] = self.iou_weight
        weights['eos'] = self.eos_weight
        
        return weights
    
    def discretize(self, value: torch.Tensor, num_bins: int, value_range: Tuple[float, float]) -> torch.Tensor:
        """将连续值离散化"""
        min_val, max_val = value_range
        normalized = (value - min_val) / (max_val - min_val)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        discrete = (normalized * (num_bins - 1)).round().long()
        return discrete
    
    def continuous_from_discrete(self, discrete_idx: torch.Tensor, num_bins: int, value_range: Tuple[float, float]) -> torch.Tensor:
        """从离散索引转换回连续值"""
        min_val, max_val = value_range
        normalized = discrete_idx.float() / (num_bins - 1)
        continuous = normalized * (max_val - min_val) + min_val
        return continuous
    
    def compute_3d_iou(self, pred_boxes: torch.Tensor, gt_boxes_with_rotation: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """
        计算3D IoU - 简化为AABB IoU（忽略旋转）
        Args:
            pred_boxes: [B, S, 6] (x, y, z, w, h, l) - 轴对齐预测框
            gt_boxes_with_rotation: [B, S, 10] (x, y, z, w, h, l, qx, qy, qz, qw) - 带旋转的GT框
        Returns:
            iou: [B, S] IoU值
        """
        # 提取GT的位置和尺寸（忽略旋转）
        gt_pos = gt_boxes_with_rotation[..., :3]  # [B, S, 3]
        gt_size = gt_boxes_with_rotation[..., 3:6]  # [B, S, 3] 
        
        # 简化为AABB IoU计算
        return self._compute_aabb_iou(pred_boxes, gt_pos, gt_size, epsilon)
    
    def _get_box_corners(self, pos: torch.Tensor, size: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        """
        获取旋转box的8个顶点
        Args:
            pos: [B, S, 3] 中心位置
            size: [B, S, 3] 尺寸 (w, h, l)  
            rotation: [B, S, 4] 四元数 (x, y, z, w)
        Returns:
            corners: [B, S, 8, 3] 8个顶点坐标
        """

        batch_size, seq_len = pos.shape[:2]
        device = pos.device
        
        # 本地坐标系下的8个顶点 (相对于box中心)
        w, h, l = size[..., 0:1], size[..., 1:2], size[..., 2:3]  # [B, S, 1]
        
        # 8个顶点的本地坐标
        local_corners = torch.stack([
            torch.cat([-w/2, -h/2, -l/2], dim=-1),  # 顶点0
            torch.cat([+w/2, -h/2, -l/2], dim=-1),  # 顶点1
            torch.cat([+w/2, +h/2, -l/2], dim=-1),  # 顶点2
            torch.cat([-w/2, +h/2, -l/2], dim=-1),  # 顶点3
            torch.cat([-w/2, -h/2, +l/2], dim=-1),  # 顶点4
            torch.cat([+w/2, -h/2, +l/2], dim=-1),  # 顶点5
            torch.cat([+w/2, +h/2, +l/2], dim=-1),  # 顶点6
            torch.cat([-w/2, +h/2, +l/2], dim=-1),  # 顶点7
        ], dim=-2)  # [B, S, 8, 3]
        
        # 应用旋转
        rotated_corners = self._rotate_points(local_corners, rotation)  # [B, S, 8, 3]
        
        # 平移到世界坐标
        world_corners = rotated_corners + pos.unsqueeze(-2)  # [B, S, 8, 3]
        
        return world_corners
    
    def _rotate_points(self, points: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
        """
        使用四元数旋转点
        Args:
            points: [B, S, N, 3] 点坐标
            quaternion: [B, S, 4] 四元数 (x, y, z, w)
        Returns:
            rotated_points: [B, S, N, 3] 旋转后的点
        """
        # 归一化四元数
        q = quaternion / (torch.norm(quaternion, dim=-1, keepdim=True) + 1e-8)
        qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]  # [B, S] 去掉最后的维度
        
        # 构造旋转矩阵
        r11 = 1 - 2 * (qy*qy + qz*qz)
        r12 = 2 * (qx*qy - qz*qw)
        r13 = 2 * (qx*qz + qy*qw)
        r21 = 2 * (qx*qy + qz*qw)
        r22 = 1 - 2 * (qx*qx + qz*qz)
        r23 = 2 * (qy*qz - qx*qw)
        r31 = 2 * (qx*qz - qy*qw)
        r32 = 2 * (qy*qz + qx*qw)
        r33 = 1 - 2 * (qx*qx + qy*qy)
        
        # 旋转矩阵 [B, S, 3, 3]
        R = torch.stack([
            torch.stack([r11, r12, r13], dim=-1),
            torch.stack([r21, r22, r23], dim=-1),
            torch.stack([r31, r32, r33], dim=-1)
        ], dim=-2)
        
        # 应用旋转: R @ points^T -> [B, S, 3, N] -> [B, S, N, 3]
        rotated = torch.matmul(R, points.transpose(-1, -2)).transpose(-1, -2)
        
        return rotated
    
    def _rotate_single_points(self, points: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
        """旋转单组点"""
        # 添加batch维度
        points_batch = points.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 3]
        q_batch = quaternion.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        
        rotated = self._rotate_points(points_batch, q_batch)  # [1, 1, N, 3]
        return rotated[0, 0]  # [N, 3]
    
    def _euler_to_quaternion(self, roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """
        将欧拉角转换为四元数
        Args:
            roll: [B, S] 绕x轴旋转角度（弧度）
            pitch: [B, S] 绕y轴旋转角度（弧度）
            yaw: [B, S] 绕z轴旋转角度（弧度）
        Returns:
            quaternion: [B, S, 4] 四元数 (x, y, z, w)
        """
        batch_size, seq_len = roll.shape
        device = roll.device
        
        # 转换为numpy进行scipy计算
        roll_np = roll.detach().cpu().numpy()
        pitch_np = pitch.detach().cpu().numpy()
        yaw_np = yaw.detach().cpu().numpy()
        
        # 创建欧拉角数组 [B*S, 3]
        euler_angles = np.stack([roll_np.flatten(), pitch_np.flatten(), yaw_np.flatten()], axis=1)
        
        # 使用scipy转换为四元数
        rotations = Rotation.from_euler('xyz', euler_angles)
        quaternions = rotations.as_quat()  # [B*S, 4] (x, y, z, w)
        
        # 转换回torch张量并重塑
        quaternions_tensor = torch.from_numpy(quaternions).to(device).float()
        quaternions_tensor = quaternions_tensor.view(batch_size, seq_len, 4)
        
        return quaternions_tensor
    
    def _compute_aabb_iou(self, pred_boxes: torch.Tensor, gt_pos: torch.Tensor, gt_size: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """
        计算AABB IoU（轴对齐包围盒IoU）
        Args:
            pred_boxes: [B, S, 6] (x, y, z, w, h, l) - 预测框
            gt_pos: [B, S, 3] (x, y, z) - GT框中心位置
            gt_size: [B, S, 3] (w, h, l) - GT框尺寸
            epsilon: 数值稳定性参数
        Returns:
            iou: [B, S] IoU值
        """
        # 预测框的边界
        pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 3] / 2  # 左边界
        pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 4] / 2  # 下边界
        pred_z1 = pred_boxes[..., 2] - pred_boxes[..., 5] / 2  # 后边界
        pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 3] / 2  # 右边界
        pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 4] / 2  # 上边界
        pred_z2 = pred_boxes[..., 2] + pred_boxes[..., 5] / 2  # 前边界
        
        # GT框的边界
        gt_x1 = gt_pos[..., 0] - gt_size[..., 0] / 2  # 左边界
        gt_y1 = gt_pos[..., 1] - gt_size[..., 1] / 2  # 下边界
        gt_z1 = gt_pos[..., 2] - gt_size[..., 2] / 2  # 后边界
        gt_x2 = gt_pos[..., 0] + gt_size[..., 0] / 2  # 右边界
        gt_y2 = gt_pos[..., 1] + gt_size[..., 1] / 2  # 上边界
        gt_z2 = gt_pos[..., 2] + gt_size[..., 2] / 2  # 前边界
        
        # 计算交集边界
        inter_x1 = torch.max(pred_x1, gt_x1)
        inter_y1 = torch.max(pred_y1, gt_y1)
        inter_z1 = torch.max(pred_z1, gt_z1)
        inter_x2 = torch.min(pred_x2, gt_x2)
        inter_y2 = torch.min(pred_y2, gt_y2)
        inter_z2 = torch.min(pred_z2, gt_z2)
        
        # 计算交集体积
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_l = torch.clamp(inter_z2 - inter_z1, min=0)
        inter_volume = inter_w * inter_h * inter_l
        
        # 计算并集体积
        pred_volume = pred_boxes[..., 3] * pred_boxes[..., 4] * pred_boxes[..., 5]
        gt_volume = gt_size[..., 0] * gt_size[..., 1] * gt_size[..., 2]
        union_volume = pred_volume + gt_volume - inter_volume
        
        # 计算IoU
        iou = inter_volume / (union_volume + epsilon)
        
        return iou
    def _compute_projection_iou(self, pred_boxes: torch.Tensor, gt_pos: torch.Tensor, 
                               gt_size: torch.Tensor, gt_rotation: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """
        基于3维投影的IoU计算：分别计算XY、XZ、YZ平面的投影IoU，然后组合
        这种方法简单、高效且可微
        """
        device = pred_boxes.device
        batch_size, seq_len = pred_boxes.shape[:2]
        
        # 计算旋转GT框的8个顶点
        gt_corners = self._get_box_corners(gt_pos, gt_size, gt_rotation)  # [B, S, 8, 3]
        
        # 预测框的顶点（轴对齐）
        pred_corners = self._get_aabb_corners(pred_boxes[:, :, :3], pred_boxes[:, :, 3:6])  # [B, S, 8, 3]
        
        # 分别计算三个平面的投影IoU
        iou_xy = self._compute_2d_projection_iou(pred_corners[:, :, :, :2], gt_corners[:, :, :, :2], epsilon)  # XY平面
        iou_xz = self._compute_2d_projection_iou(
            torch.stack([pred_corners[:, :, :, 0], pred_corners[:, :, :, 2]], dim=-1),  # X,Z
            torch.stack([gt_corners[:, :, :, 0], gt_corners[:, :, :, 2]], dim=-1), epsilon)  # XZ平面
        iou_yz = self._compute_2d_projection_iou(
            torch.stack([pred_corners[:, :, :, 1], pred_corners[:, :, :, 2]], dim=-1),  # Y,Z
            torch.stack([gt_corners[:, :, :, 1], gt_corners[:, :, :, 2]], dim=-1), epsilon)  # YZ平面
        
        # 检查各平面IoU是否有异常
        if torch.any(iou_xy > 1) or torch.any(iou_xz > 1) or torch.any(iou_yz > 1):
            print(f"🚨 投影IoU异常:")
            print(f"    XY平面IoU最大值: {iou_xy.max():.6f}")
            print(f"    XZ平面IoU最大值: {iou_xz.max():.6f}") 
            print(f"    YZ平面IoU最大值: {iou_yz.max():.6f}")
            
            # 详细分析异常
            if torch.any(iou_xy > 1):
                print("🔍 XY平面详细信息:")
                max_idx = torch.argmax(iou_xy.view(-1))
                batch_idx = max_idx // iou_xy.shape[1]
                seq_idx = max_idx % iou_xy.shape[1]
                print(f"    异常位置: batch={batch_idx}, seq={seq_idx}")
                print(f"    GT corners XY: {gt_corners[batch_idx, seq_idx, :, :2].cpu().numpy()}")
                print(f"    Pred corners XY: {pred_corners[batch_idx, seq_idx, :, :2].cpu().numpy()}")
                
            # 打印原始box信息进行对比
            print("📦 原始box信息:")
            print(f"    GT pos: {gt_pos[0, 0].cpu().numpy()}")
            print(f"    GT size: {gt_size[0, 0].cpu().numpy()}")
            print(f"    GT rotation: {gt_rotation[0, 0].cpu().numpy()}")
            print(f"    Pred pos: {pred_boxes[0, 0, :3].cpu().numpy()}")
            print(f"    Pred size: {pred_boxes[0, 0, 3:6].cpu().numpy()}")
        
        # 组合三个投影IoU
        # 方法1: 几何平均 
        iou_product = iou_xy * iou_xz * iou_yz
        combined_iou = torch.pow(torch.clamp(iou_product, min=epsilon), 1/3)
        
        # 检查最终IoU
        if torch.any(combined_iou > 1):
            print(f"🚨 组合IoU异常 > 1.0: 最大值={combined_iou.max():.6f}")
            print(f"    IoU乘积: 最大值={iou_product.max():.6f}")
            print(f"    各平面IoU: XY={iou_xy.max():.6f}, XZ={iou_xz.max():.6f}, YZ={iou_yz.max():.6f}")
        
        # 方法2: 算术平均
        # combined_iou = (iou_xy + iou_xz + iou_yz) / 3
        
        # 方法3: 最小值 (保守估计)  
        # combined_iou = torch.min(torch.min(iou_xy, iou_xz), iou_yz)
        
        return combined_iou
    
    def _get_aabb_corners(self, center: torch.Tensor, size: torch.Tensor) -> torch.Tensor:
        """
        获取轴对齐box的8个顶点
        Args:
            center: [B, S, 3] 中心位置
            size: [B, S, 3] 尺寸 (w, h, l)
        Returns:
            corners: [B, S, 8, 3] 8个顶点坐标
        """
        w, h, l = size[..., 0:1], size[..., 1:2], size[..., 2:3]  # [B, S, 1]
        
        # 8个顶点的偏移量
        offsets = torch.stack([
            torch.cat([-w/2, -h/2, -l/2], dim=-1),  # 顶点0
            torch.cat([+w/2, -h/2, -l/2], dim=-1),  # 顶点1
            torch.cat([+w/2, +h/2, -l/2], dim=-1),  # 顶点2
            torch.cat([-w/2, +h/2, -l/2], dim=-1),  # 顶点3
            torch.cat([-w/2, -h/2, +l/2], dim=-1),  # 顶点4
            torch.cat([+w/2, -h/2, +l/2], dim=-1),  # 顶点5
            torch.cat([+w/2, +h/2, +l/2], dim=-1),  # 顶点6
            torch.cat([-w/2, +h/2, +l/2], dim=-1),  # 顶点7
        ], dim=-2)  # [B, S, 8, 3]
        
        # 加上中心位置
        corners = offsets + center.unsqueeze(-2)  # [B, S, 8, 3]
        return corners
    
    def _compute_2d_projection_iou(self, pred_corners_2d: torch.Tensor, gt_corners_2d: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """
        计算2D投影的IoU - 使用多边形IoU而不是外包矩形
        Args:
            pred_corners_2d: [B, S, 8, 2] 预测框在某个平面的8个投影点
            gt_corners_2d: [B, S, 8, 2] GT框在某个平面的8个投影点
        Returns:
            iou_2d: [B, S] 2D投影IoU
        """
        batch_size, seq_len = pred_corners_2d.shape[:2]
        device = pred_corners_2d.device
        
        iou_results = torch.zeros(batch_size, seq_len, device=device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # 获取预测框和GT框在2D平面的投影多边形
                pred_poly = pred_corners_2d[b, s]  # [8, 2]
                gt_poly = gt_corners_2d[b, s]      # [8, 2]
                
                # 计算多边形IoU
                iou_val = self._compute_polygon_iou(pred_poly, gt_poly, epsilon)
                iou_results[b, s] = iou_val
                
        return iou_results
    
    def _compute_polygon_iou(self, poly1: torch.Tensor, poly2: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
        """
        计算两个多边形的IoU
        Args:
            poly1: [N, 2] 多边形1的顶点
            poly2: [M, 2] 多边形2的顶点
        Returns:
            iou: 标量IoU值
        """
        try:
            # 转换到CPU numpy进行几何计算
            poly1_np = poly1.detach().cpu().numpy()
            poly2_np = poly2.detach().cpu().numpy()
            
            # 计算多边形面积
            area1 = self._polygon_area(poly1_np)
            area2 = self._polygon_area(poly2_np)
            
            if area1 <= epsilon or area2 <= epsilon:
                return torch.tensor(0.0, device=poly1.device)
            
            # 计算交集面积 (使用Sutherland-Hodgman裁剪算法的简化版本)
            intersection_area = self._polygon_intersection_area(poly1_np, poly2_np)
            
            # 计算IoU
            union_area = area1 + area2 - intersection_area
            if union_area <= epsilon:
                return torch.tensor(0.0, device=poly1.device)
                
            iou = intersection_area / union_area
            
            # 检查IoU异常
            if iou > 1.0:
                print(f"🚨 多边形IoU异常 > 1.0: {iou:.6f}")
                print(f"    面积1: {area1:.6f}, 面积2: {area2:.6f}")
                print(f"    交集面积: {intersection_area:.6f}, 并集面积: {union_area:.6f}")
                iou = min(iou, 1.0)  # 限制在1.0以内
            
            return torch.tensor(max(0.0, iou), device=poly1.device)
            
        except Exception as e:
            print(f"⚠️  多边形IoU计算出错: {e}")
            return torch.tensor(0.0, device=poly1.device)
    
    def _polygon_area(self, vertices: np.ndarray) -> float:
        """计算多边形面积 (Shoelace公式)"""
        if len(vertices) < 3:
            return 0.0
        
        n = len(vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        
        return abs(area) / 2.0
    
    def _polygon_intersection_area(self, poly1: np.ndarray, poly2: np.ndarray) -> float:
        """
        计算两个多边形的交集面积
        简化实现：对于复杂的多边形相交，这里使用一个近似方法
        """
        try:
            # 简化方法：使用网格采样来近似计算交集
            # 获取两个多边形的边界框
            min_x = min(poly1[:, 0].min(), poly2[:, 0].min())
            max_x = max(poly1[:, 0].max(), poly2[:, 0].max())
            min_y = min(poly1[:, 1].min(), poly2[:, 1].min())
            max_y = max(poly1[:, 1].max(), poly2[:, 1].max())
            
            if max_x <= min_x or max_y <= min_y:
                return 0.0
            
            # 网格采样
            resolution = 100  # 可以调整精度
            x_step = (max_x - min_x) / resolution
            y_step = (max_y - min_y) / resolution
            
            intersection_count = 0
            total_count = 0
            
            for i in range(resolution):
                for j in range(resolution):
                    x = min_x + (i + 0.5) * x_step
                    y = min_y + (j + 0.5) * y_step
                    
                    in_poly1 = self._point_in_polygon(x, y, poly1)
                    in_poly2 = self._point_in_polygon(x, y, poly2)
                    
                    if in_poly1 and in_poly2:
                        intersection_count += 1
                    total_count += 1
            
            # 计算交集面积
            total_area = (max_x - min_x) * (max_y - min_y)
            intersection_area = (intersection_count / total_count) * total_area
            
            return intersection_area
            
        except Exception as e:
            print(f"⚠️  多边形交集面积计算出错: {e}")
            return 0.0
    
    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """判断点是否在多边形内 (射线法)"""
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def create_valid_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """创建有效目标的掩码"""
        return targets != self.pad_id
    
    def distance_aware_classification_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """距离感知的分类损失"""
        if not self.distance_aware_cls:
            return self.cross_entropy(logits, targets)
        
        # 标准交叉熵
        # 确保targets是整数类型且有效
        if targets.dtype != torch.long:
            targets = targets.long()
        
        # 确保logits是正确的形状 [batch_size, num_classes]
        if logits.dim() == 1:
            # 如果logits是1D，暂时跳过这个样本
            return torch.tensor(0.0, device=logits.device)
        
        # 确保数据类型正确
        if logits.dtype != torch.float32:
            logits = logits.float()
        
        # 检查targets是否包含无效值
        if torch.any(torch.isnan(targets)) or torch.any(torch.isinf(targets)):
            # 将NaN/Inf值替换为0
            targets = torch.where(torch.isnan(targets) | torch.isinf(targets), 
                                torch.zeros_like(targets), targets)
        
        if torch.any(targets < 0) or torch.any(targets >= logits.shape[-1]):
            # 将无效值clamp到有效范围
            targets = torch.clamp(targets, 0, logits.shape[-1] - 1)
        
        # 使用nn.CrossEntropyLoss而不是F.cross_entropy
        # 但我们需要reduction='none'，所以使用F.cross_entropy
        try:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        except RuntimeError as e:
            # 回退到nn.CrossEntropyLoss
            ce_loss = self.cross_entropy(logits, targets)
            # 手动计算reduction='none'
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Focal权重
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # 距离权重
        pred_classes = torch.argmax(logits, dim=-1)
        distance = torch.abs(pred_classes - targets).float()
        num_classes = logits.shape[-1]
        distance_weight = 1.0 + self.distance_alpha * (distance / num_classes)
        
        # 组合损失
        loss = focal_weight * distance_weight * ce_loss
        return loss.mean()
    
    def classification_loss(self, logits_dict: Dict[str, torch.Tensor], targets_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算分类损失"""
        losses = {}
        total_loss = 0.0
        
        attributes = [
            ('x', self.num_discrete_x, self.continuous_range_x),
            ('y', self.num_discrete_y, self.continuous_range_y),
            ('z', self.num_discrete_z, self.continuous_range_z),
            ('w', self.num_discrete_w, self.continuous_range_w),
            ('h', self.num_discrete_h, self.continuous_range_h),
            ('l', self.num_discrete_l, self.continuous_range_l),
            ('roll', self.num_discrete_roll, self.continuous_range_roll),  # 新增旋转
            ('pitch', self.num_discrete_pitch, self.continuous_range_pitch),
            ('yaw', self.num_discrete_yaw, self.continuous_range_yaw),
        ]
        
        for attr_name, num_bins, value_range in attributes:
            logits = logits_dict[f'{attr_name}_logits']  # [B, seq_len, num_bins]
            targets = targets_dict[attr_name]  # [B, max_boxes] 归一化的[0,1]
            
            # 🔧 修复：使用原始continuous_range计算target labels
            min_val, max_val = value_range
            
            # 正确的序列对齐：
            # logits[0] 预测第1个box，logits[1] 预测第2个box，...，logits[14] 预测第15个box
            # 我们只使用前max_boxes个预测与targets对比
            target_seq_len = targets.shape[1]  # max_boxes = 15  
            pred_seq_len = logits.shape[1]
            
            # 处理序列长度不匹配的情况
            if pred_seq_len > target_seq_len:
                # 模型生成过长：截断到目标长度
                logits = logits[:, :target_seq_len, :]  # 取前15个预测 [B, 15, num_bins]
            elif pred_seq_len < target_seq_len:
                # 模型生成过短：用零填充到目标长度
                batch_size, _, num_bins = logits.shape
                padding_length = target_seq_len - pred_seq_len
                padding = torch.zeros(batch_size, padding_length, num_bins, 
                                    device=logits.device, dtype=logits.dtype)
                logits = torch.cat([logits, padding], dim=1)  # [B, 15, num_bins]
            
            # 创建有效掩码
            valid_mask = self.create_valid_mask(targets)
            
            if valid_mask.sum() == 0:
                losses[f'{attr_name}_cls'] = torch.tensor(0.0, device=logits.device)
                continue
            
            # 离散化目标
            discrete_targets = self.discretize(targets, num_bins, value_range)
            
            # 只计算有效位置的损失
            valid_logits = logits[valid_mask]  # [N_valid, num_bins]
            valid_targets = discrete_targets[valid_mask]  # [N_valid]
            
            # 使用距离感知分类损失
            loss = self.distance_aware_classification_loss(valid_logits, valid_targets)
            losses[f'{attr_name}_cls'] = loss
            total_loss += loss
        
        losses['total_classification'] = total_loss
        return losses
    
    def iou_loss(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor, gt_rotations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算3D IoU损失 - 使用简化的AABB IoU（暂时绕过旋转问题）
        Args:
            pred_boxes: [B, S, 6] 预测的轴对齐box (x, y, z, w, h, l)
            gt_boxes: [B, S, 6] GT box的位置和尺寸 (x, y, z, w, h, l) 
            gt_rotations: [B, S, 4] GT box的旋转四元数 (x, y, z, w)
        """
        valid_mask = self.create_valid_mask(gt_boxes[..., 0])
        
        if valid_mask.sum() == 0:
            return {
                'iou_loss': torch.tensor(0.0, device=pred_boxes.device),
                'mean_iou': torch.tensor(0.0, device=pred_boxes.device)
            }
        
        # 使用平面投影IoU（支持旋转）
        gt_boxes_with_rotation = torch.cat([gt_boxes, gt_rotations], dim=-1)  # [B, S, 10]
        iou = self.compute_3d_iou(pred_boxes, gt_boxes_with_rotation)
        
        # 处理IoU计算失败的情况
        if iou is None:
            print("⚠️  IoU计算返回None，使用零IoU")
            iou = torch.zeros_like(pred_boxes[:, :, 0])
        
        valid_iou = iou[valid_mask]
        
        iou_loss = 1.0 - valid_iou.mean()
        mean_iou = valid_iou.mean()
        
        return {
            'iou_loss': iou_loss,
            'mean_iou': mean_iou
        }
    
    def delta_regression_loss(self, delta_dict: Dict[str, torch.Tensor], targets_dict: Dict[str, torch.Tensor], discrete_preds: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算delta回归损失"""
        losses = {}
        total_loss = 0.0
        
        attributes = [
            ('x', self.num_discrete_x, self.continuous_range_x),
            ('y', self.num_discrete_y, self.continuous_range_y),
            ('z', self.num_discrete_z, self.continuous_range_z),
            ('w', self.num_discrete_w, self.continuous_range_w),
            ('h', self.num_discrete_h, self.continuous_range_h),
            ('l', self.num_discrete_l, self.continuous_range_l),
            ('roll', self.num_discrete_roll, self.continuous_range_roll),  # 新增旋转
            ('pitch', self.num_discrete_pitch, self.continuous_range_pitch),
            ('yaw', self.num_discrete_yaw, self.continuous_range_yaw),
        ]
        
        for attr_name, num_bins, value_range in attributes:
            delta_pred = delta_dict[f'{attr_name}_delta'].squeeze(-1)  # [B, seq_len, 1] -> [B, seq_len]
            targets = targets_dict[attr_name]  # [B, max_boxes] 归一化的targets [0,1]
            discrete_pred = discrete_preds[f'{attr_name}_discrete']  # [B, max_boxes] (已对齐)
            
            # 🔧 修复：使用原始continuous_range进行计算
            min_val, max_val = value_range
            
            # 对delta_pred也只使用前max_boxes个预测与targets对齐  
            target_seq_len = targets.shape[1]
            pred_seq_len = delta_pred.shape[1]
            
            if pred_seq_len > target_seq_len:
                delta_pred = delta_pred[:, :target_seq_len]  # [B, 15]
            elif pred_seq_len < target_seq_len:
                # 用零填充delta_pred到目标长度
                batch_size = delta_pred.shape[0]
                padding_length = target_seq_len - pred_seq_len
                padding = torch.zeros(batch_size, padding_length, 
                                    device=delta_pred.device, dtype=delta_pred.dtype)
                delta_pred = torch.cat([delta_pred, padding], dim=1)  # [B, 15]
            
            # 基于归一化的targets创建valid mask（pad_id=-1）
            valid_mask = self.create_valid_mask(targets)
            
            if valid_mask.sum() == 0:
                losses[f'{attr_name}_delta'] = torch.tensor(0.0, device=delta_pred.device)
                continue
            
            # 🔧 修复：基于GT计算target delta，而不是基于模型预测
            min_val, max_val = value_range
            bin_width = (max_val - min_val) / (num_bins - 1)
            
            # 首先检查GT是否在valid range内
            gt_in_range = (targets >= min_val) & (targets <= max_val) & valid_mask
            
            if gt_in_range.sum() == 0:
                # 没有在有效范围内的GT，跳过delta loss计算
                losses[f'{attr_name}_delta'] = torch.tensor(0.0, device=delta_pred.device)
                continue
            
            # 只对在范围内的GT计算delta loss
            valid_targets = targets[gt_in_range]
            valid_delta_pred = delta_pred[gt_in_range]
            
            # 将GT离散化，得到GT应该在的discrete bin
            # 使用更精确的数值处理避免精度问题
            normalized = (valid_targets - min_val) / (max_val - min_val)
            gt_discrete = torch.clamp(
                (normalized * (num_bins - 1)).round().long(),
                0, num_bins - 1
            )
            
            # 基于GT的discrete bin计算continuous base
            gt_continuous_base = self.continuous_from_discrete(gt_discrete, num_bins, value_range)
            
            # 计算GT的真实delta（GT相对于其应该在的discrete bin的偏移）
            target_delta = (valid_targets - gt_continuous_base) / bin_width
            
            # 现在target_delta应该在[-0.5, 0.5]范围内
            # 使用更宽松的阈值来减少数值精度导致的警告
            tolerance = 1e-5  # 增加容差
            if torch.any(torch.abs(target_delta) > 0.5 + tolerance):
                print(f"⚠️  仍有异常的target delta: {target_delta}")
                print(f"异常值索引: {torch.where(torch.abs(target_delta) > 0.5 + tolerance)[0]}")
                # 作为最后的安全措施，clamp到[-0.5, 0.5]
                target_delta = torch.clamp(target_delta, -0.5, 0.5)
            
            # 损失计算（已经筛选过的有效样本）
            # valid_delta_pred 和 target_delta 已经是匹配的
            
            # Smooth L1损失
            loss = self.smooth_l1_loss(valid_delta_pred, target_delta).mean()
            losses[f'{attr_name}_delta'] = loss
            total_loss += loss
        
        losses['total_delta'] = total_loss
        return losses
    
    def eos_loss(self, eos_logits: torch.Tensor, sequence_lengths: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """
        计算长度平衡的EOS损失
        Args:
            eos_logits: [B, N] - 每个位置的EOS预测logits
            sequence_lengths: [B] - 每个样本的真实序列长度
            max_seq_len: 最大序列长度
        """
        batch_size = eos_logits.shape[0]
        device = eos_logits.device
        
        total_loss = 0.0
        valid_samples = 0
        
        for b in range(batch_size):
            seq_len = sequence_lengths[b].item()
            seq_len = min(seq_len, max_seq_len - 1)  # 确保不越界
            
            if seq_len < 0:  # 跳过无效样本
                continue
                
            # 当前样本的有效位置数
            num_valid_positions = seq_len + 1
            
            # 分别计算"继续"和"结束"位置的损失
            continue_loss = 0.0
            eos_loss_val = 0.0
            
            # 1. "继续"位置的损失 (位置0到seq_len-1)
            if seq_len > 0:
                continue_logits = eos_logits[b, :seq_len]  # [seq_len]
                continue_targets = torch.zeros_like(continue_logits)  # 全部为0
                continue_loss = F.binary_cross_entropy_with_logits(
                    continue_logits, continue_targets, reduction='mean'
                )
            
            # 2. "结束"位置的损失 (位置seq_len)
            if seq_len < max_seq_len:
                eos_logit = eos_logits[b, seq_len]  # 标量
                eos_target = torch.tensor(1.0, device=device)  # EOS=1
                eos_loss_val = F.binary_cross_entropy_with_logits(
                    eos_logit, eos_target, reduction='mean'
                )
            
            # 3. 平衡组合：给"继续"和"结束"相等的权重
            # 这样长短序列的损失就相对平衡了
            if seq_len > 0 and seq_len < max_seq_len:
                # 两种损失都存在：各占50%权重
                sample_loss = 0.5 * continue_loss + 0.5 * eos_loss_val
            elif seq_len == 0:
                # 只有EOS损失
                sample_loss = eos_loss_val
            else:
                # 只有continue损失 (极少见情况)
                sample_loss = continue_loss
            
            total_loss += sample_loss
            valid_samples += 1
        
        if valid_samples == 0:
            return torch.tensor(0.0, device=device)
        
        # 返回batch平均损失
        return total_loss / valid_samples
    
    def _compute_sequence_lengths(self, targets_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算每个样本的有效序列长度"""
        x_targets = targets_dict['x']  # [B, max_boxes]
        sequence_lengths = []
        for b in range(x_targets.shape[0]):
            valid_count = (x_targets[b] != -999).sum().item()
            sequence_lengths.append(valid_count)
        return torch.tensor(sequence_lengths, device=x_targets.device)
    
    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        delta_dict: Dict[str, torch.Tensor], 
        eos_logits: torch.Tensor,
        targets_dict: Dict[str, torch.Tensor],
        sequence_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        loss_dict = {}
        
        # 1. 分类损失
        cls_losses = self.classification_loss(logits_dict, targets_dict)
        loss_dict.update(cls_losses)
        
        # 2. 获取离散预测
        discrete_preds = {}
        # 获取目标序列长度
        target_seq_len = targets_dict['x'].shape[1]  # max_boxes = 15
        
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
            logits = logits_dict[f'{attr}_logits']  # [B, seq_len, num_bins]
            pred_seq_len = logits.shape[1]
            
            # 处理序列长度不匹配的情况
            if pred_seq_len > target_seq_len:
                logits = logits[:, :target_seq_len, :]  # [B, 15, num_bins]
            elif pred_seq_len < target_seq_len:
                # 用零填充logits到目标长度
                batch_size, _, num_bins = logits.shape
                padding_length = target_seq_len - pred_seq_len
                padding = torch.zeros(batch_size, padding_length, num_bins, 
                                    device=logits.device, dtype=logits.dtype)
                logits = torch.cat([logits, padding], dim=1)  # [B, 15, num_bins]
                
            discrete_preds[f'{attr}_discrete'] = torch.argmax(logits, dim=-1)
        
        # 3. Delta回归损失
        delta_losses = self.delta_regression_loss(delta_dict, targets_dict, discrete_preds)
        loss_dict.update(delta_losses)
        
        # 4. 构建3D boxes进行IoU计算
        pred_boxes_list = []
        gt_boxes_list = []
        
        for attr, num_bins, value_range in [
            ('x', self.num_discrete_x, self.continuous_range_x),
            ('y', self.num_discrete_y, self.continuous_range_y),
            ('z', self.num_discrete_z, self.continuous_range_z),
            ('w', self.num_discrete_w, self.continuous_range_w),
            ('h', self.num_discrete_h, self.continuous_range_h),
            ('l', self.num_discrete_l, self.continuous_range_l),
            ('roll', self.num_discrete_roll, self.continuous_range_roll),  # 新增旋转
            ('pitch', self.num_discrete_pitch, self.continuous_range_pitch),
            ('yaw', self.num_discrete_yaw, self.continuous_range_yaw),
        ]:
            discrete_pred = discrete_preds[f'{attr}_discrete']  # [B, 15] (已裁剪)
            continuous_base = self.continuous_from_discrete(discrete_pred, num_bins, value_range)
            delta_pred = delta_dict[f'{attr}_delta'].squeeze(-1)  # [B, 16] (原始) -> [B, 16]
            
            # 确保delta_pred与discrete_pred长度一致
            pred_seq_len = delta_pred.shape[1]
            if pred_seq_len > target_seq_len:
                delta_pred = delta_pred[:, :target_seq_len]  # [B, 15]
            elif pred_seq_len < target_seq_len:
                # 用零填充delta_pred到目标长度
                batch_size = delta_pred.shape[0]
                padding_length = target_seq_len - pred_seq_len
                padding = torch.zeros(batch_size, padding_length, 
                                    device=delta_pred.device, dtype=delta_pred.dtype)
                delta_pred = torch.cat([delta_pred, padding], dim=1)  # [B, 15]
            
            min_val, max_val = value_range
            bin_width = (max_val - min_val) / (num_bins - 1)
            final_pred = continuous_base + delta_pred * bin_width
            
            pred_boxes_list.append(final_pred.unsqueeze(-1))
            
            # 🔧 修复：IoU计算使用原始范围的GT
            gt_boxes_list.append(targets_dict[attr].unsqueeze(-1))
        
        pred_boxes = torch.cat(pred_boxes_list, dim=-1)
        gt_boxes = torch.cat(gt_boxes_list, dim=-1)
        
        # IoU损失 (使用旋转感知计算)
        # 从欧拉角计算四元数
        gt_rotations = self._euler_to_quaternion(
            targets_dict['roll'], targets_dict['pitch'], targets_dict['yaw']
        )  # [B, S, 4] 四元数旋转
        iou_losses = self.iou_loss(pred_boxes, gt_boxes, gt_rotations)
        loss_dict.update(iou_losses)
        
        # 5. 🔥 关键：根据IoU计算自适应权重
        mean_iou = iou_losses['mean_iou']
        adaptive_weights = self.compute_adaptive_weights(mean_iou)
        
        # 记录自适应权重
        device = next(iter(loss_dict.values())).device
        loss_dict['adaptive_classification_weight'] = torch.tensor(adaptive_weights['classification'], device=device)
        loss_dict['adaptive_delta_weight'] = torch.tensor(adaptive_weights['delta'], device=device)
        
        # 6. EOS损失
        # 如果没有提供sequence_lengths，自动计算
        if sequence_lengths is None:
            sequence_lengths = self._compute_sequence_lengths(targets_dict)
        
        if sequence_lengths is not None:
            # 处理eos_logits长度不匹配的情况
            target_seq_len = targets_dict['x'].shape[1]  # 目标序列长度
            pred_seq_len = eos_logits.shape[1]
            
            if pred_seq_len > target_seq_len:
                eos_logits = eos_logits[:, :target_seq_len]
            elif pred_seq_len < target_seq_len:
                # 用零填充eos_logits到目标长度
                batch_size = eos_logits.shape[0]
                padding_length = target_seq_len - pred_seq_len
                padding = torch.zeros(batch_size, padding_length, 
                                    device=eos_logits.device, dtype=eos_logits.dtype)
                eos_logits = torch.cat([eos_logits, padding], dim=1)
            
            max_seq_len = eos_logits.shape[1]
            eos_loss_val = self.eos_loss(eos_logits, sequence_lengths, max_seq_len)
            loss_dict['eos_loss'] = eos_loss_val
        else:
            loss_dict['eos_loss'] = torch.tensor(0.0, device=eos_logits.device)
        
        # 7. 使用自适应权重计算总损失
        total_loss = (
            adaptive_weights['classification'] * loss_dict['total_classification'] +
            adaptive_weights['iou'] * loss_dict['iou_loss'] +
            adaptive_weights['delta'] * loss_dict['total_delta'] +
            adaptive_weights['eos'] * loss_dict['eos_loss']
        )
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict


def create_adaptive_loss_function(
    adaptive_config: Dict = None,
    model_config: Dict = None,
    loss_config: Dict = None
) -> AdaptivePrimitiveTransformer3DLoss:
    """创建自适应损失函数"""
    
    # 默认自适应配置
    default_adaptive_config = {
        'adaptive_classification': True,
        'adaptive_delta': True,
        'min_classification_weight': 0.1,
        'max_classification_weight': 3.0,
        'iou_threshold_high': 0.7,
        'iou_threshold_low': 0.3,
    }
    
    # 默认模型配置
    default_model_config = {
        'num_discrete_x': 128,
        'num_discrete_y': 128,
        'num_discrete_z': 128,
        'num_discrete_w': 64,
        'num_discrete_h': 64,
        'num_discrete_l': 64,
        'continuous_range_x': (0.5, 2.5),
        'continuous_range_y': (-2, 2),
        'continuous_range_z': (-1.5, 1.5),
        'continuous_range_w': (0.1, 1.0),
        'continuous_range_h': (0.1, 1.0),
        'continuous_range_l': (0.1, 1.0),
    }
    
    # 默认损失配置
    default_loss_config = {
        'base_classification_weight': 1.0,
        'iou_weight': 2.0,
        'delta_weight': 1.0,
        'eos_weight': 0.5,
        'pad_id': -1.0,
        'label_smoothing': 0.1,
        'distance_aware_cls': True,
        'distance_alpha': 2.0,
        'focal_gamma': 2.0
    }
    
    # 合并配置
    adaptive_config = {**default_adaptive_config, **(adaptive_config or {})}
    model_config = {**default_model_config, **(model_config or {})}
    loss_config = {**default_loss_config, **(loss_config or {})}
    
    # 创建损失函数
    loss_fn = AdaptivePrimitiveTransformer3DLoss(
        **model_config,
        **loss_config,
        **adaptive_config
    )
    
    return loss_fn


# 演示自适应权重机制
if __name__ == "__main__":
    print("=== IoU自适应权重机制演示 ===\n")
    
    # 创建损失函数
    loss_fn = create_adaptive_loss_function()
    
    # 模拟不同IoU场景
    iou_scenarios = [
        ("训练初期 - 很低IoU", 0.1),
        ("训练中期 - 中等IoU", 0.5), 
        ("训练后期 - 高IoU", 0.8),
        ("收敛阶段 - 很高IoU", 0.9)
    ]
    
    print("IoU自适应权重变化:")
    print("阶段                   IoU    分类权重  Delta权重  策略")
    print("-" * 65)
    
    for stage_name, iou_value in iou_scenarios:
        iou_tensor = torch.tensor(iou_value)
        weights = loss_fn.compute_adaptive_weights(iou_tensor)
        
        if iou_value <= 0.3:
            strategy = "重点关注粗定位"
        elif iou_value >= 0.7:
            strategy = "重点关注精调优"
        else:
            strategy = "平衡定位和精调"
        
        print(f"{stage_name:<20} {iou_value:.1f}    {weights['classification']:.2f}      {weights['delta']:.2f}     {strategy}")
    
    print(f"\n配置参数:")
    print(f"  分类权重范围: [{loss_fn.min_classification_weight}, {loss_fn.max_classification_weight}]")
    print(f"  IoU阈值: 低={loss_fn.iou_threshold_low}, 高={loss_fn.iou_threshold_high}")
    print(f"  自适应分类: {loss_fn.adaptive_classification}")
    print(f"  自适应Delta: {loss_fn.adaptive_delta}") 