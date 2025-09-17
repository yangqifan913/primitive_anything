# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from einops import rearrange, repeat, pack
from x_transformers import Decoder
from x_transformers.autoregressive_wrapper import eval_decorator
import torchvision.models as models
from gateloop_transformer import SimpleGateLoopLayer
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass

class FiLM(nn.Module):
    """Feature-wise Linear Modulation layer - 与PrimitiveAnything源码保持一致"""
    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim_out or dim
        
        self.to_gamma = nn.Linear(dim, dim_out, bias=False)
        self.to_beta = nn.Linear(dim, dim_out)
        
        self.gamma_mult = nn.Parameter(torch.zeros(1,))
        self.beta_mult = nn.Parameter(torch.zeros(1,))
        
    def forward(self, x, cond):
        """
        Args:
            x: [B, seq_len, dim] - 输入特征
            cond: [B, cond_dim] - 条件特征
        Returns:
            modulated_x: [B, seq_len, dim_out] - 调制后的特征
        """
        gamma, beta = self.to_gamma(cond), self.to_beta(cond)
        gamma, beta = tuple(rearrange(t, 'b d -> b 1 d') for t in (gamma, beta))
        
        # 初始化到恒等映射
        gamma = (1 + self.gamma_mult * gamma.tanh())
        beta = beta.tanh() * self.beta_mult
        
        # 经典FiLM
        return x * gamma + beta

class GateLoopBlock(nn.Module):
    """门控循环块 - 参考PrimitiveAnything实现，支持缓存"""
    def __init__(self, dim, depth=2, use_heinsen=False):
        super().__init__()
        self.gateloops = nn.ModuleList([])

        for _ in range(depth):
            gateloop = SimpleGateLoopLayer(dim=dim, use_heinsen=use_heinsen)
            self.gateloops.append(gateloop)

    def forward(self, x, cache=None):
        """
        参考PrimitiveAnything的GateLoopBlock实现
        
        Args:
            x: 输入tensor [B, seq_len, dim]
            cache: 门控循环的缓存列表
            
        Returns:
            x: 输出tensor
            new_caches: 更新后的缓存列表
        """
        received_cache = cache is not None and len(cache) > 0

        if x.numel() == 0:  # 空tensor检查
            return x, cache

        if received_cache:
            # 如果有缓存，分离之前的序列和新token
            prev, x = x[:, :-1], x[:, -1:]

        cache = cache if cache is not None else []
        cache_iter = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache_iter, None)
            # 检查gateloop是否支持cache和return_cache参数
            if hasattr(gateloop, 'forward') and 'cache' in gateloop.forward.__code__.co_varnames:
                try:
                    out, new_cache = gateloop(x, cache=layer_cache, return_cache=True)
                    new_caches.append(new_cache)
                except TypeError:
                    # 如果不支持return_cache，使用普通前向传播
                    out = gateloop(x)
                    new_caches.append(None)
            else:
                # 普通前向传播
                out = gateloop(x)
                new_caches.append(None)
            
            x = x + out

        if received_cache:
            # 如果有缓存，将之前的序列与新处理的token连接
            x = torch.cat((prev, x), dim=-2)

        return x, new_caches

def build_2d_sine_positional_encoding(H, W, dim):
    """
    构建 [H, W, dim] 的 2D sine-cosine 位置编码
    顺序：先左到右，再由下到上
    """
    # 修改顺序：先x（左到右），再y（下到上）
    x_embed = torch.linspace(0, 1, steps=W).unsqueeze(0).repeat(H, 1)
    y_embed = torch.linspace(0, 1, steps=H).unsqueeze(1).repeat(1, W)
    
    dim_t = torch.arange(dim // 4, dtype=torch.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / (dim // 2))

    pos_x = x_embed[..., None] / dim_t
    pos_y = y_embed[..., None] / dim_t

    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)

    pos = torch.cat((pos_x, pos_y), dim=-1)  # [H, W, dim] - 先x后y
    return pos

class EnhancedFPN(nn.Module):
    """超轻量版Feature Pyramid Network - 大幅降低内存占用"""
    def __init__(self, in_channels, out_channels=32, attention_heads=2, attention_layers=None):  # 支持配置化
        super().__init__()
        self.out_channels = out_channels
        
        # 侧边连接层 - 将不同层的特征统一到相同通道数
        # ResNet50的通道数: [256, 512, 1024, 2048]
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),   # layer1
            nn.Conv2d(512, out_channels, 1),   # layer2
            nn.Conv2d(1024, out_channels, 1),  # layer3
            nn.Conv2d(2048, out_channels, 1),  # layer4
        ])
        
        # 简化的平滑层 - 只保留一层卷积
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(4)
        ])
        
        # 移除额外的卷积层
        self.extra_convs = nn.ModuleList([
            nn.Identity() for _ in range(4)
        ])
        
        # 注意力机制配置化
        if attention_layers is None:
            attention_layers = [2, 3]  # 默认在layer3和layer4使用注意力
        self.attention_layers = attention_layers
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads=attention_heads, batch_first=True)
            for _ in range(len(attention_layers))
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of tensors from different layers
                     [layer1_feat, layer2_feat, layer3_feat, layer4_feat]
        Returns:
            fpn_features: List of FPN features at different scales
        """
        # 从高层到低层处理
        laterals = []
        for i, (feat, lateral_conv) in enumerate(zip(features, self.lateral_convs)):
            lateral = lateral_conv(feat)
            laterals.append(lateral)
        
        # 自顶向下路径
        for i in range(len(laterals) - 2, -1, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[-2:], 
                mode='nearest'
            )
            # 添加侧边连接
            laterals[i] = laterals[i] + upsampled
        
        # 平滑处理和注意力
        fpn_features = []
        attention_idx = 0  # 注意力模块的索引
        
        for i, (lateral, smooth_conv, extra_conv) in enumerate(
            zip(laterals, self.smooth_convs, self.extra_convs)
        ):
            # 平滑处理
            smoothed = F.relu(smooth_conv(lateral))
            # 额外卷积（现在是Identity）
            extra = extra_conv(smoothed)
            
            # 可配置的注意力机制
            if i in self.attention_layers and attention_idx < len(self.attention_blocks):
                b, c, h, w = extra.shape
                extra_flat = extra.view(b, c, h*w).permute(0, 2, 1)  # [B, H*W, C]
                attended, _ = self.attention_blocks[attention_idx](extra_flat, extra_flat, extra_flat)
                attended = attended.permute(0, 2, 1).view(b, c, h, w)  # [B, C, H, W]
                fpn_features.append(attended)
                attention_idx += 1
            else:
                # 不使用注意力的层直接输出
                fpn_features.append(extra)
        
        return fpn_features

class DeepVisualProcessor(nn.Module):
    """简化版视觉特征处理器"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 简化的卷积网络
        self.conv_layers = nn.Sequential(
            # 单层处理
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        # 残差连接
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_layers(x)
        return out + residual

class ImageEncoder(nn.Module):
    """简化版图像编码器 - 支持多种ResNet backbone + 简化FPN + 简化处理器"""
    def __init__(self, input_channels=3, output_dim=256, use_fpn=True, backbone="resnet50", pretrained=True):  # 从256减少到192
        super().__init__()
        self.use_fpn = use_fpn
        self.backbone = backbone
        self.pretrained = pretrained
        
        # 根据配置选择backbone
        if backbone.lower() == "resnet18":
            if pretrained:
                self.backbone_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone_model = models.resnet18(weights=None)
            self.backbone_channels = [64, 128, 256, 512]  # ResNet18的通道数
        elif backbone.lower() == "resnet34":
            if pretrained:
                self.backbone_model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone_model = models.resnet34(weights=None)
            self.backbone_channels = [64, 128, 256, 512]  # ResNet34的通道数
        elif backbone.lower() == "resnet50":
            if pretrained:
                self.backbone_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone_model = models.resnet50(weights=None)
            self.backbone_channels = [256, 512, 1024, 2048]  # ResNet50的通道数
        elif backbone.lower() == "resnet101":
            if pretrained:
                self.backbone_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.backbone_model = models.resnet101(weights=None)
            self.backbone_channels = [256, 512, 1024, 2048]  # ResNet101的通道数
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Supported: resnet18, resnet34, resnet50, resnet101")
        
        # 如果输入通道数不是3，需要正确适配预训练权重
        if input_channels != 3:
            # 保存原始预训练的conv1权重 [64, 3, 7, 7]
            pretrained_conv1_weight = self.backbone_model.conv1.weight.data.clone()
            
            # 创建新的conv1层
            self.backbone_model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            # 初始化新的conv1权重
            with torch.no_grad():
                if input_channels == 6:  # RGBXYZ情况
                    # RGB通道：直接复制预训练权重
                    self.backbone_model.conv1.weight[:, :3, :, :] = pretrained_conv1_weight
                    # XYZ通道：使用预训练权重的平均值初始化
                    mean_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
                    self.backbone_model.conv1.weight[:, 3:, :, :] = mean_weight.repeat(1, 3, 1, 1)
                else:
                    # 其他情况：使用预训练权重的平均值
                    mean_weight = pretrained_conv1_weight.mean(dim=1, keepdim=True)
                    self.backbone_model.conv1.weight.data = mean_weight.repeat(1, input_channels, 1, 1)
        
        # 简化FPN模块
        if use_fpn:
            # 根据backbone选择FPN的输入通道数
            if backbone.lower() in ["resnet18", "resnet34"]:
                fpn_in_channels = 512  # ResNet18/34的最后一层通道数
            else:  # resnet50, resnet101
                fpn_in_channels = 2048  # ResNet50/101的最后一层通道数
            self.fpn = EnhancedFPN(in_channels=fpn_in_channels, out_channels=32)  # 从64减少到32
            self.feature_dim = 32
        else:
            # 根据backbone选择特征维度
            if backbone.lower() in ["resnet18", "resnet34"]:
                self.feature_dim = 512
            else:  # resnet50, resnet101
                self.feature_dim = 2048
        
        # 简化深层处理器
        self.deep_processor = DeepVisualProcessor(self.feature_dim, output_dim)
        
        # 空间特征投影层
        self.spatial_projection = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        
        print(f"Enhanced ImageEncoder: {backbone.upper()} + {'Ultra-Lightweight FPN (32ch)' if use_fpn else 'No FPN'} + SimpleProcessor -> {self.feature_dim} -> {output_dim} (spatial features)")
        
    def forward(self, x, return_2d_features=False):
        # 获取ResNet的卷积特征图
        x = self.backbone_model.conv1(x)
        x = self.backbone_model.bn1(x)
        x = self.backbone_model.relu(x)
        x = self.backbone_model.maxpool(x)
        
        # 提取不同层的特征
        layer1_feat = self.backbone_model.layer1(x)      # [B, C1, H/4, W/4]
        layer2_feat = self.backbone_model.layer2(layer1_feat)  # [B, C2, H/8, W/8]
        layer3_feat = self.backbone_model.layer3(layer2_feat)  # [B, C3, H/16, W/16]
        layer4_feat = self.backbone_model.layer4(layer3_feat)  # [B, C4, H/32, W/32]
        
        if self.use_fpn:
            # 使用简化FPN处理多尺度特征
            fpn_features = self.fpn([layer1_feat, layer2_feat, layer3_feat, layer4_feat])
            
            # 选择最细粒度的特征（最高分辨率）
            output = fpn_features[0]  # [B, 64, H/4, W/4]
        else:
            # 不使用FPN，直接使用最后一层特征
            output = layer4_feat  # [B, 512, H/32, W/32]
        
        # 简化深层处理
        output = self.deep_processor(output)  # [B, output_dim, H, W]
        
        # 投影到目标维度
        output = self.spatial_projection(output)  # [B, output_dim, H, W]
        
        if return_2d_features:
            return output  # 返回2D特征图 [B, output_dim, H, W]
        
        # 将空间特征展平为序列 [batch_size, H*W, output_dim]
        batch_size, channels, height, width = output.shape
        output = output.permute(0, 2, 3, 1).contiguous()  # [B, H, W, channels]
        output = output.view(batch_size, height * width, channels)  # [B, H*W, channels]
        
        return output

@dataclass
class IncrementalState:
    """增量生成状态 - 参考PrimitiveAnything实现"""
    current_sequence: torch.Tensor  # [B, current_len, embed_dim]
    image_embed: torch.Tensor      # [B, H*W, image_dim]  
    image_cond: torch.Tensor       # [B, image_cond_dim]
    stopped_samples: torch.Tensor  # [B] 布尔值，标记哪些样本已停止
    current_step: int              # 当前步数
    
    # 多级KV缓存用于真正的增量解码（参考PrimitiveAnything）
    decoder_cache: Optional[object] = None  # Transformer decoder的cache
    gateloop_cache: Optional[List] = None   # 门控循环块的cache
    
    # 生成结果跟踪
    generated_boxes: Optional[Dict[str, List]] = None
    
    def __post_init__(self):
        if self.gateloop_cache is None:
            self.gateloop_cache = []
        if self.generated_boxes is None:
            # 这里会在initialize_incremental_generation中正确设置
            pass

class PrimitiveTransformer3D(nn.Module):
    """3D基本体变换器 - 支持RGBXYZ输入和3D箱子生成"""
    def __init__(
        self,
        *,
        # 离散化参数 - 3D坐标 + 旋转
        num_discrete_x = 128,
        num_discrete_y = 128,
        num_discrete_z = 128,  # 新增z坐标
        num_discrete_w = 64,
        num_discrete_h = 64,
        num_discrete_l = 64,  # 新增length维度
        num_discrete_roll = 64,    # 新增roll旋转
        num_discrete_pitch = 64,   # 新增pitch旋转
        num_discrete_yaw = 64,     # 新增yaw旋转
        
        # 连续范围 - 3D坐标 + 旋转
        continuous_range_x = [0.5, 2.5],
        continuous_range_y = [-2, 2],
        continuous_range_z = [-1.5, 1.5],  # 新增z范围
        continuous_range_w = [0.3, 0.7],
        continuous_range_h = [0.3, 0.7],
        continuous_range_l = [0.3, 0.7],  # 新增length范围
        continuous_range_roll = [-1.5708, 1.5708],    # 新增roll范围 (-90° to +90°)
        continuous_range_pitch = [-1.5708, 1.5708],   # 新增pitch范围 (-90° to +90°)
        continuous_range_yaw = [-1.5708, 1.5708],     # 新增yaw范围 (-90° to +90°)
        
        # 嵌入维度 - 3D + 旋转
        dim_x_embed = 64,
        dim_y_embed = 64,
        dim_z_embed = 64,  # 新增z嵌入
        dim_w_embed = 32,
        dim_h_embed = 32,
        dim_l_embed = 32,  # 新增length嵌入
        dim_roll_embed = 32,   # 新增roll嵌入
        dim_pitch_embed = 32,  # 新增pitch嵌入
        dim_yaw_embed = 32,    # 新增yaw嵌入
        
        # 模型参数
        dim = 512,
        max_primitive_len = 10,
        attn_depth = 6,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_dropout = 0.0,  # 注意力dropout
        ff_dropout = 0.0,    # 前馈dropout
        
        # 图像编码器 - 支持6通道RGBXYZ输入
        image_encoder_dim = 512,
        use_fpn = True,
        backbone = "resnet50",
        pretrained = True,
        
        # 其他参数
        shape_cond_with_cat = False,
        condition_on_image = True,
        gateloop_depth = 2,
        gateloop_use_heinsen = False,
        
        pad_id = -1,
    ):
        super().__init__()
        
        # 3D离散化参数
        self.num_discrete_x = num_discrete_x
        self.num_discrete_y = num_discrete_y
        self.num_discrete_z = num_discrete_z  # 新增
        self.num_discrete_w = num_discrete_w
        self.num_discrete_h = num_discrete_h
        self.num_discrete_l = num_discrete_l  # 新增
        self.num_discrete_roll = num_discrete_roll    # 新增
        self.num_discrete_pitch = num_discrete_pitch   # 新增
        self.num_discrete_yaw = num_discrete_yaw       # 新增
        
        # 3D连续范围
        self.continuous_range_x = continuous_range_x
        self.continuous_range_y = continuous_range_y
        self.continuous_range_z = continuous_range_z  # 新增
        self.continuous_range_w = continuous_range_w
        self.continuous_range_h = continuous_range_h
        self.continuous_range_l = continuous_range_l  # 新增
        self.continuous_range_roll = continuous_range_roll    # 新增
        self.continuous_range_pitch = continuous_range_pitch   # 新增
        self.continuous_range_yaw = continuous_range_yaw       # 新增
        
        # 其他参数
        self.shape_cond_with_cat = shape_cond_with_cat
        self.condition_on_image = condition_on_image
        self.gateloop_depth = gateloop_depth
        self.gateloop_use_heinsen = gateloop_use_heinsen
        self.pad_id = pad_id
        
        # 图像条件投影层
        if shape_cond_with_cat:
            self.image_cond_proj = nn.Linear(image_encoder_dim, dim)
        else:
            self.image_cond_proj = None
        
        # 图像条件化层
        if condition_on_image:
            self.image_film_cond = FiLM(dim, dim)
            self.image_cond_proj_film = nn.Linear(image_encoder_dim, self.image_film_cond.to_gamma.in_features)
        else:
            self.image_film_cond = None
            self.image_cond_proj_film = None
        
        # 门控循环块
        if gateloop_depth > 0:
            self.gateloop_block = GateLoopBlock(dim, depth=gateloop_depth, use_heinsen=gateloop_use_heinsen)
        else:
            self.gateloop_block = None
        
        # 图像编码器 - 修改为6通道RGBXYZ输入
        self.image_encoder = ImageEncoder(
            input_channels=6,  # 修改：RGB(3) + XYZ(3) = 6通道
            output_dim=image_encoder_dim,
            use_fpn=use_fpn,
            backbone=backbone,
            pretrained=pretrained
        )
        
        # 3D嵌入层
        self.x_embed = nn.Embedding(num_discrete_x, dim_x_embed)
        self.y_embed = nn.Embedding(num_discrete_y, dim_y_embed)
        self.z_embed = nn.Embedding(num_discrete_z, dim_z_embed)  # 新增
        self.w_embed = nn.Embedding(num_discrete_w, dim_w_embed)
        self.h_embed = nn.Embedding(num_discrete_h, dim_h_embed)
        self.l_embed = nn.Embedding(num_discrete_l, dim_l_embed)  # 新增
        self.roll_embed = nn.Embedding(num_discrete_roll, dim_roll_embed)    # 新增
        self.pitch_embed = nn.Embedding(num_discrete_pitch, dim_pitch_embed)  # 新增
        self.yaw_embed = nn.Embedding(num_discrete_yaw, dim_yaw_embed)        # 新增
        
        # 投影层 - 更新总维度（包含旋转属性）
        total_embed_dim = (dim_x_embed + dim_y_embed + dim_z_embed + 
                          dim_w_embed + dim_h_embed + dim_l_embed +
                          dim_roll_embed + dim_pitch_embed + dim_yaw_embed)
        self.project_in = nn.Linear(total_embed_dim, dim)
        
        # 连续值到embedding的转换层（用于属性间依赖）
        self.continuous_to_x_embed = nn.Linear(1, dim_x_embed)
        self.continuous_to_y_embed = nn.Linear(1, dim_y_embed)
        self.continuous_to_z_embed = nn.Linear(1, dim_z_embed)
        self.continuous_to_w_embed = nn.Linear(1, dim_w_embed)
        self.continuous_to_h_embed = nn.Linear(1, dim_h_embed)
        self.continuous_to_l_embed = nn.Linear(1, dim_l_embed)
        self.continuous_to_roll_embed = nn.Linear(1, dim_roll_embed)    # 新增
        self.continuous_to_pitch_embed = nn.Linear(1, dim_pitch_embed)  # 新增
        self.continuous_to_yaw_embed = nn.Linear(1, dim_yaw_embed)        # 新增
        
        # 解码器
        self.decoder = Decoder(
            dim=dim,
            depth=attn_depth,
            heads=attn_heads,
            attn_dim_head=attn_dim_head,
            attn_flash=True,
            attn_dropout=attn_dropout,  # 使用注意力dropout
            ff_dropout=ff_dropout,      # 使用前馈dropout
            cross_attend=True,
            cross_attn_dim_context=image_encoder_dim,
        )
        
        # 3D预测头 - 添加z坐标和length
        self.to_x_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_x),
        )

        self.to_y_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_y),
        )
        
        # 新增z坐标预测头
        self.to_z_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_z),
        )

        self.to_w_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_w),
        )

        self.to_h_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_h),
        )
        
        # 新增length预测头
        self.to_l_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_l),
        )
        
        # 3D Delta预测头
        self.to_x_delta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        self.to_y_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 新增z delta预测头
        self.to_z_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        self.to_w_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        self.to_h_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 新增length delta预测头
        self.to_l_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 新增旋转属性预测头
        self.to_roll_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_roll),
        )
        
        self.to_pitch_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed + dim_roll_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_pitch),
        )
        
        self.to_yaw_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed + dim_roll_embed + dim_pitch_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, num_discrete_yaw),
        )
        
        # 新增旋转属性delta预测头
        self.to_roll_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        self.to_pitch_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed + dim_roll_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        self.to_yaw_delta = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed + dim_roll_embed + dim_pitch_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # EOS预测网络 - 更新输入维度（包含旋转属性）
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim + dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed + dim_roll_embed + dim_pitch_embed + dim_yaw_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 特殊token
        self.sos_token = nn.Parameter(torch.randn(1, dim))
        self.pad_id = pad_id
        self.max_seq_len = max_primitive_len
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_primitive_len + 1, dim))
        
        print(f"3D PrimitiveTransformer: RGBXYZ(6ch) Input + 3D Box Generation (x,y,z,w,h,l)")
    
    def continuous_from_discrete(self, discrete_values, num_bins, value_range):
        """将离散值转换为连续值"""
        min_val, max_val = value_range
        return min_val + (discrete_values.float() / (num_bins - 1)) * (max_val - min_val)
    
    def get_continuous_embed(self, attr_name, continuous_value):
        """从连续值获取embedding"""
        # 将连续值reshape为[B, 1]或[B, seq_len, 1]
        if continuous_value.dim() == 1:
            continuous_value = continuous_value.unsqueeze(-1)
        elif continuous_value.dim() == 2:
            continuous_value = continuous_value.unsqueeze(-1)
        
        if attr_name == 'x':
            return self.continuous_to_x_embed(continuous_value)
        elif attr_name == 'y':
            return self.continuous_to_y_embed(continuous_value)
        elif attr_name == 'z':
            return self.continuous_to_z_embed(continuous_value)
        elif attr_name == 'w':
            return self.continuous_to_w_embed(continuous_value)
        elif attr_name == 'h':
            return self.continuous_to_h_embed(continuous_value)
        elif attr_name == 'l':
            return self.continuous_to_l_embed(continuous_value)
        elif attr_name == 'roll':
            return self.continuous_to_roll_embed(continuous_value)
        elif attr_name == 'pitch':
            return self.continuous_to_pitch_embed(continuous_value)
        elif attr_name == 'yaw':
            return self.continuous_to_yaw_embed(continuous_value)
        else:
            raise ValueError(f"Unknown attribute: {attr_name}")
    
    def predict_attribute_with_continuous_embed(self, step_embed, attr_name, prev_embeds=None, use_gumbel=None, temperature=1.0):
        """预测属性并返回连续值和embedding - 支持可微分采样"""
        # 构建输入
        if prev_embeds is None:
            input_embed = step_embed
        else:
            input_embed = torch.cat([step_embed] + prev_embeds, dim=-1)
        
        # 获取预测头和参数
        if attr_name == 'x':
            logits_head = self.to_x_logits
            delta_head = self.to_x_delta
            num_bins = self.num_discrete_x
            value_range = self.continuous_range_x
        elif attr_name == 'y':
            logits_head = self.to_y_logits
            delta_head = self.to_y_delta
            num_bins = self.num_discrete_y
            value_range = self.continuous_range_y
        elif attr_name == 'z':
            logits_head = self.to_z_logits
            delta_head = self.to_z_delta
            num_bins = self.num_discrete_z
            value_range = self.continuous_range_z
        elif attr_name == 'w':
            logits_head = self.to_w_logits
            delta_head = self.to_w_delta
            num_bins = self.num_discrete_w
            value_range = self.continuous_range_w
        elif attr_name == 'h':
            logits_head = self.to_h_logits
            delta_head = self.to_h_delta
            num_bins = self.num_discrete_h
            value_range = self.continuous_range_h
        elif attr_name == 'l':
            logits_head = self.to_l_logits
            delta_head = self.to_l_delta
            num_bins = self.num_discrete_l
            value_range = self.continuous_range_l
        elif attr_name == 'roll':
            logits_head = self.to_roll_logits
            delta_head = self.to_roll_delta
            num_bins = self.num_discrete_roll
            value_range = self.continuous_range_roll
        elif attr_name == 'pitch':
            logits_head = self.to_pitch_logits
            delta_head = self.to_pitch_delta
            num_bins = self.num_discrete_pitch
            value_range = self.continuous_range_pitch
        elif attr_name == 'yaw':
            logits_head = self.to_yaw_logits
            delta_head = self.to_yaw_delta
            num_bins = self.num_discrete_yaw
            value_range = self.continuous_range_yaw
        else:
            raise ValueError(f"Unknown attribute: {attr_name}")
        
        # 预测
        logits = logits_head(input_embed)
        delta = torch.tanh(delta_head(input_embed).squeeze(-1)) * 0.5
        
        # 决定使用哪种采样方式
        if use_gumbel is None:
            use_gumbel = self.training  # 训练时使用Gumbel Softmax，推理时使用argmax
        
        if use_gumbel:
            # 使用Gumbel Softmax进行可微分采样
            continuous_base = self._differentiable_discrete_to_continuous(
                logits, num_bins, value_range, temperature
            )
            # 用于返回的离散预测（不参与梯度传播）
            discrete_pred = torch.argmax(logits, dim=-1)
        else:
            # 推理时使用argmax（不需要梯度）
            discrete_pred = torch.argmax(logits, dim=-1)
            continuous_base = self.continuous_from_discrete(discrete_pred, num_bins, value_range)
        
        # 加上delta修正 - 🔧 修复：delta应该按bin_width缩放
        if use_gumbel:
            # Gumbel Softmax情况下，需要计算等效的bin_width
            min_val, max_val = value_range
            bin_width = (max_val - min_val) / (num_bins - 1)
            continuous_value = continuous_base + delta * bin_width
        else:
            # argmax情况下，同样使用bin_width缩放
            min_val, max_val = value_range
            bin_width = (max_val - min_val) / (num_bins - 1)
            continuous_value = continuous_base + delta * bin_width
        
        # 确保数据类型一致性（针对混合精度训练）
        if continuous_value.dtype != delta.dtype:
            continuous_value = continuous_value.to(dtype=delta.dtype)
        
        # 获取embedding
        embed = self.get_continuous_embed(attr_name, continuous_value)
        
        return logits, delta, continuous_value, embed
    
    def _differentiable_discrete_to_continuous(self, logits, num_bins, value_range, temperature=1.0):
        """使用Gumbel Softmax进行可微分的离散到连续转换 - 内存优化版本"""
        # 获取输入的数据类型和设备
        input_dtype = logits.dtype
        input_device = logits.device
        
        # 创建连续值的离散网格（复用缓存）
        min_val, max_val = value_range
        cache_key = (num_bins, min_val, max_val, input_device, input_dtype)
        
        # 简单的缓存机制避免重复创建
        if not hasattr(self, '_discrete_values_cache'):
            self._discrete_values_cache = {}
        
        if cache_key not in self._discrete_values_cache:
            self._discrete_values_cache[cache_key] = torch.linspace(
                min_val, max_val, num_bins, device=input_device, dtype=input_dtype
            )
        discrete_values = self._discrete_values_cache[cache_key]
        
        # 使用更内存友好的计算方式
        # 避免创建大的中间tensor
        if temperature <= 0.1:
            # 低温度时直接用argmax近似（节省内存）
            indices = torch.argmax(logits, dim=-1)
            continuous_value = discrete_values[indices]
        else:
            # 正常Gumbel Softmax - 确保数据类型一致
            # 在混合精度下，logits可能是Half类型
            if input_dtype == torch.half:
                # 对于Half精度，需要特殊处理
                with torch.cuda.amp.autocast(enabled=False):
                    # 转换为float32进行计算
                    logits_fp32 = logits.float()
                    discrete_values_fp32 = discrete_values.float()
                    
                    gumbel_weights = F.gumbel_softmax(logits_fp32, tau=temperature, hard=False, dim=-1)
                    continuous_value = torch.mv(gumbel_weights, discrete_values_fp32)
                    
                    # 转换回原始精度
                    continuous_value = continuous_value.to(dtype=input_dtype)
            else:
                # 正常精度计算
                gumbel_weights = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
                continuous_value = torch.mv(gumbel_weights, discrete_values)
        
        return continuous_value
    
    def discretize(self, value, num_discrete, continuous_range):
        """将连续值离散化"""
        min_val, max_val = continuous_range
        discrete = ((value - min_val) / (max_val - min_val) * (num_discrete - 1)).clamp(0, num_discrete - 1).long()
        return discrete
    
    def undiscretize(self, discrete, num_discrete, continuous_range):
        """将离散值转换回连续值"""
        min_val, max_val = continuous_range
        continuous = discrete.float() / (num_discrete - 1) * (max_val - min_val) + min_val
        return continuous
    
    def encode_primitive(self, x, y, z, w, h, l, roll, pitch, yaw, primitive_mask):
        """编码3D基本体参数"""
        # 检查是否有有效的框
        if x.numel() == 0 or y.numel() == 0 or z.numel() == 0 or w.numel() == 0 or h.numel() == 0 or l.numel() == 0:
            batch_size = x.shape[0] if x.numel() > 0 else 1
            dim = self.project_in.out_features
            empty_embed = torch.zeros(batch_size, 0, dim, device=x.device)
            empty_discrete = (torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device))
            return empty_embed, empty_discrete
        
        # 3D离散化
        discrete_x = self.discretize(x, self.num_discrete_x, self.continuous_range_x)
        discrete_y = self.discretize(y, self.num_discrete_y, self.continuous_range_y)
        discrete_z = self.discretize(z, self.num_discrete_z, self.continuous_range_z)
        discrete_w = self.discretize(w, self.num_discrete_w, self.continuous_range_w)
        discrete_h = self.discretize(h, self.num_discrete_h, self.continuous_range_h)
        discrete_l = self.discretize(l, self.num_discrete_l, self.continuous_range_l)
        discrete_roll = self.discretize(roll, self.num_discrete_roll, self.continuous_range_roll)
        discrete_pitch = self.discretize(pitch, self.num_discrete_pitch, self.continuous_range_pitch)
        discrete_yaw = self.discretize(yaw, self.num_discrete_yaw, self.continuous_range_yaw)
        
        # 3D嵌入
        x_embed = self.x_embed(discrete_x)
        y_embed = self.y_embed(discrete_y)
        z_embed = self.z_embed(discrete_z)
        w_embed = self.w_embed(discrete_w)
        h_embed = self.h_embed(discrete_h)
        l_embed = self.l_embed(discrete_l)
        roll_embed = self.roll_embed(discrete_roll)
        pitch_embed = self.pitch_embed(discrete_pitch)
        yaw_embed = self.yaw_embed(discrete_yaw)
        
        # 组合3D特征（包含旋转）
        primitive_embed, _ = pack([x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed, pitch_embed, yaw_embed], 'b np *')
        primitive_embed = self.project_in(primitive_embed)
        
        # 使用primitive_mask将无效位置的embedding设置为0
        primitive_embed = primitive_embed.masked_fill(~primitive_mask.unsqueeze(-1), 0.)
        
        return primitive_embed, (discrete_x, discrete_y, discrete_z, discrete_w, discrete_h, discrete_l, discrete_roll, discrete_pitch, discrete_yaw)
    
    def forward(
        self,
        *,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        w: Tensor,
        h: Tensor,
        l: Tensor,
        roll: Tensor,    # 新增
        pitch: Tensor,   # 新增
        yaw: Tensor,     # 新增
        image: Tensor,  # 现在是RGBXYZ，6通道
    ):
        """3D前向传播"""
        # 创建3D mask（包含旋转属性）
        primitive_mask = (x != self.pad_id) & (y != self.pad_id) & (z != self.pad_id) & (w != self.pad_id) & (h != self.pad_id) & (l != self.pad_id) & (roll != self.pad_id) & (pitch != self.pad_id) & (yaw != self.pad_id)
        
        # 编码3D基本体（包含旋转）
        codes, discrete_coords = self.encode_primitive(x, y, z, w, h, l, roll, pitch, yaw, primitive_mask)

        # 编码RGBXYZ图像
        image_embed = self.image_encoder(image)  # [batch_size, H*W, image_encoder_dim]
        
        # 添加位置编码
        batch_size, seq_len, _ = codes.shape
        device = codes.device
        
        # 为图像特征添加2D位置编码
        H = W = int(np.sqrt(image_embed.shape[1]))
        pos_embed_2d = build_2d_sine_positional_encoding(H, W, image_embed.shape[-1])
        pos_embed_2d = pos_embed_2d.flatten(0, 1).unsqueeze(0).to(image_embed.device)
        image_embed = image_embed + pos_embed_2d
        
        # 构建输入序列
        history = codes
        sos = repeat(self.sos_token, 'n d -> b n d', b=batch_size)
        
        primitive_codes, packed_sos_shape = pack([sos, history], 'b * d')
        seq_len = primitive_codes.shape[1]
        pos_embed = self.pos_embed[:, :seq_len, :]
        primitive_codes = primitive_codes + pos_embed
        
        # 图像条件化处理
        if self.condition_on_image and self.image_film_cond is not None:
            pooled_image_embed = image_embed.mean(dim=1)
            image_cond = self.image_cond_proj_film(pooled_image_embed)
            primitive_codes = self.image_film_cond(primitive_codes, image_cond)
        
        # 门控循环块处理
        if self.gateloop_block is not None:
            primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes)
        
        # 变换器解码 - 禁用gradient checkpointing以避免与Scheduled Sampling冲突
        attended_codes = self.decoder(
            primitive_codes,
            context=image_embed,
        )

        return attended_codes
    
    def forward_with_predictions(
        self,
        *,
        x: Tensor,
        y: Tensor,
        z: Tensor,
        w: Tensor,
        h: Tensor,
        l: Tensor,
        roll: Tensor,    # 新增
        pitch: Tensor,   # 新增
        yaw: Tensor,     # 新增
        image: Tensor
    ):
        """带预测输出的前向传播，用于训练"""
        # 先调用标准前向传播获取attended_codes
        attended_codes = self.forward(
            x=x, y=y, z=z, w=w, h=h, l=l, roll=roll, pitch=pitch, yaw=yaw, image=image
        )
        
        # attended_codes shape: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = attended_codes.shape
        
        # 为每个序列位置计算预测
        all_logits = {f'{attr}_logits': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
        all_deltas = {f'{attr}_delta': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
        all_continuous = {f'{attr}_continuous': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']}
        eos_logits_list = []
        
        for t in range(seq_len):
            step_embed = attended_codes[:, t, :]  # [batch_size, model_dim]
            
            # 累积的embed用于后续属性预测
            x_embed = y_embed = z_embed = w_embed = h_embed = l_embed = roll_embed = pitch_embed = yaw_embed = None
            
            # 预测x坐标 - 使用连续值embedding
            x_logits, x_delta, x_continuous, x_embed = self.predict_attribute_with_continuous_embed(step_embed, 'x', prev_embeds=None, use_gumbel=None, temperature=1.0)
            
            # 预测y坐标 - 使用连续值embedding
            y_logits, y_delta, y_continuous, y_embed = self.predict_attribute_with_continuous_embed(step_embed, 'y', prev_embeds=[x_embed], use_gumbel=None, temperature=1.0)
            
            # 预测z坐标 - 使用连续值embedding
            z_logits, z_delta, z_continuous, z_embed = self.predict_attribute_with_continuous_embed(step_embed, 'z', prev_embeds=[x_embed, y_embed], use_gumbel=None, temperature=1.0)
            
            # 预测w - 使用连续值embedding
            w_logits, w_delta, w_continuous, w_embed = self.predict_attribute_with_continuous_embed(step_embed, 'w', prev_embeds=[x_embed, y_embed, z_embed], use_gumbel=None, temperature=1.0)
            
            # 预测h - 使用连续值embedding
            h_logits, h_delta, h_continuous, h_embed = self.predict_attribute_with_continuous_embed(step_embed, 'h', prev_embeds=[x_embed, y_embed, z_embed, w_embed], use_gumbel=None, temperature=1.0)
            
            # 预测l - 使用连续值embedding
            l_logits, l_delta, l_continuous, l_embed = self.predict_attribute_with_continuous_embed(step_embed, 'l', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed], use_gumbel=None, temperature=1.0)
            
            # 预测roll - 使用连续值embedding
            roll_logits, roll_delta, roll_continuous, roll_embed = self.predict_attribute_with_continuous_embed(step_embed, 'roll', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed], use_gumbel=None, temperature=1.0)
            
            # 预测pitch - 使用连续值embedding
            pitch_logits, pitch_delta, pitch_continuous, pitch_embed = self.predict_attribute_with_continuous_embed(step_embed, 'pitch', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed], use_gumbel=None, temperature=1.0)
            
            # 预测yaw - 使用连续值embedding
            yaw_logits, yaw_delta, yaw_continuous, yaw_embed = self.predict_attribute_with_continuous_embed(step_embed, 'yaw', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed, pitch_embed], use_gumbel=None, temperature=1.0)
            
            # 预测EOS - 传入所有属性的嵌入
            combined_embeds = torch.cat([step_embed, x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed, pitch_embed, yaw_embed], dim=-1)
            eos_logit = self.to_eos_logits(combined_embeds).squeeze(-1)
            
            # 收集结果
            all_logits['x_logits'].append(x_logits)
            all_logits['y_logits'].append(y_logits)
            all_logits['z_logits'].append(z_logits)
            all_logits['w_logits'].append(w_logits)
            all_logits['h_logits'].append(h_logits)
            all_logits['l_logits'].append(l_logits)
            all_logits['roll_logits'].append(roll_logits)
            all_logits['pitch_logits'].append(pitch_logits)
            all_logits['yaw_logits'].append(yaw_logits)
            
            all_deltas['x_delta'].append(x_delta)
            all_deltas['y_delta'].append(y_delta)
            all_deltas['z_delta'].append(z_delta)
            all_deltas['w_delta'].append(w_delta)
            all_deltas['h_delta'].append(h_delta)
            all_deltas['l_delta'].append(l_delta)
            all_deltas['roll_delta'].append(roll_delta)
            all_deltas['pitch_delta'].append(pitch_delta)
            all_deltas['yaw_delta'].append(yaw_delta)
            
            all_continuous['x_continuous'].append(x_continuous)
            all_continuous['y_continuous'].append(y_continuous)
            all_continuous['z_continuous'].append(z_continuous)
            all_continuous['w_continuous'].append(w_continuous)
            all_continuous['h_continuous'].append(h_continuous)
            all_continuous['l_continuous'].append(l_continuous)
            all_continuous['roll_continuous'].append(roll_continuous)
            all_continuous['pitch_continuous'].append(pitch_continuous)
            all_continuous['yaw_continuous'].append(yaw_continuous)
            
            eos_logits_list.append(eos_logit)
        
        # 组装最终输出
        logits_dict = {}
        delta_dict = {}
        continuous_dict = {}
        
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
            logits_dict[f'{attr}_logits'] = torch.stack(all_logits[f'{attr}_logits'], dim=1)
            delta_dict[f'{attr}_delta'] = torch.stack(all_deltas[f'{attr}_delta'], dim=1)
            continuous_dict[f'{attr}_continuous'] = torch.stack(all_continuous[f'{attr}_continuous'], dim=1)
        
        eos_logits = torch.stack(eos_logits_list, dim=1)
        
        return {
            'logits_dict': logits_dict,
            'delta_dict': delta_dict,
            'continuous_dict': continuous_dict,
            'eos_logits': eos_logits
        }
    
    @eval_decorator
    @torch.no_grad()
    def generate(
        self,
        image: Tensor,  # RGBXYZ 6通道输入
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5,
        debug: bool = False
    ):
        """3D autoregressive生成"""
        max_seq_len = max_seq_len or self.max_seq_len
        batch_size = image.shape[0]
        device = image.device
        
        # 编码RGBXYZ图像
        image_embed = self.image_encoder(image)
        
        # 添加2D位置编码
        H = W = int(np.sqrt(image_embed.shape[1]))
        if H * W == image_embed.shape[1]:
            pos_embed_2d = build_2d_sine_positional_encoding(H, W, image_embed.shape[-1])
            pos_embed_2d = pos_embed_2d.flatten(0, 1).unsqueeze(0).to(image_embed.device)
            image_embed = image_embed + pos_embed_2d
        
        # 为每个样本独立跟踪3D生成结果
        generated_results = {
            'x': [[] for _ in range(batch_size)],
            'y': [[] for _ in range(batch_size)],
            'z': [[] for _ in range(batch_size)],  # 新增z坐标
            'w': [[] for _ in range(batch_size)],
            'h': [[] for _ in range(batch_size)],
            'l': [[] for _ in range(batch_size)]   # 新增length
        }
        
        # 跟踪每个样本是否已经停止生成
        stopped_samples = torch.zeros(batch_size, dtype=torch.bool, device=image.device)
        
        # 初始序列：只有SOS token
        current_sequence = repeat(self.sos_token, 'n d -> b n d', b=batch_size)
        
        for step in range(max_seq_len):
            # 如果所有样本都停止了，提前结束
            if torch.all(stopped_samples):
                break
            
            primitive_codes = current_sequence
            seq_len = primitive_codes.shape[1]
            pos_embed = self.pos_embed[:, :seq_len, :]
            primitive_codes = primitive_codes + pos_embed
            
            # 图像条件化处理
            if self.condition_on_image and self.image_film_cond is not None:
                pooled_image_embed = image_embed.mean(dim=1)
                image_cond = self.image_cond_proj_film(pooled_image_embed)
                primitive_codes = self.image_film_cond(primitive_codes, image_cond)
            
            # 门控循环块处理
            if self.gateloop_block is not None:
                primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes)
            
            # 通过decoder获取attended codes
            attended_codes = self.decoder(
                primitive_codes,
                context=image_embed,
            )
            
            # 用最后一个位置预测下一个token
            next_embed = attended_codes[:, -1]
            
            # 预测3D坐标和尺寸 - 按顺序：x, y, z, w, h, l
            # 预测x坐标 - 使用连续值embedding
            x_logits = self.to_x_logits(next_embed)
            x_delta = torch.tanh(self.to_x_delta(next_embed).squeeze(-1)) * 0.5
            if temperature == 0:
                next_x_discrete = x_logits.argmax(dim=-1)
            else:
                x_probs = F.softmax(x_logits / temperature, dim=-1)
                next_x_discrete = torch.multinomial(x_probs, 1).squeeze(-1)
            
            # 计算x的连续值用于后续预测
            x_continuous_base = self.continuous_from_discrete(next_x_discrete, self.num_discrete_x, self.continuous_range_x)
            x_continuous = x_continuous_base + x_delta
            x_embed = self.get_continuous_embed('x', x_continuous)
            
            # 预测y坐标 - 使用连续值embedding
            y_input = torch.cat([next_embed, x_embed], dim=-1)
            y_logits = self.to_y_logits(y_input)
            y_delta = torch.tanh(self.to_y_delta(y_input).squeeze(-1)) * 0.5
            
            if temperature == 0:
                next_y_discrete = y_logits.argmax(dim=-1)
            else:
                y_probs = F.softmax(y_logits / temperature, dim=-1)
                next_y_discrete = torch.multinomial(y_probs, 1).squeeze(-1)
            
            # 计算y的连续值用于后续预测
            y_continuous_base = self.continuous_from_discrete(next_y_discrete, self.num_discrete_y, self.continuous_range_y)
            y_continuous = y_continuous_base + y_delta
            y_embed = self.get_continuous_embed('y', y_continuous)
            
            # 预测z坐标 - 使用连续值embedding
            z_input = torch.cat([next_embed, x_embed, y_embed], dim=-1)
            z_logits = self.to_z_logits(z_input)
            z_delta = torch.tanh(self.to_z_delta(z_input).squeeze(-1)) * 0.5
            
            if temperature == 0:
                next_z_discrete = z_logits.argmax(dim=-1)
            else:
                z_probs = F.softmax(z_logits / temperature, dim=-1)
                next_z_discrete = torch.multinomial(z_probs, 1).squeeze(-1)
            
            # 计算z的连续值用于后续预测
            z_continuous_base = self.continuous_from_discrete(next_z_discrete, self.num_discrete_z, self.continuous_range_z)
            z_continuous = z_continuous_base + z_delta
            z_embed = self.get_continuous_embed('z', z_continuous)
            
            # 预测w（宽度）- 使用连续值embedding
            w_input = torch.cat([next_embed, x_embed, y_embed, z_embed], dim=-1)
            w_logits = self.to_w_logits(w_input)
            w_delta = torch.tanh(self.to_w_delta(w_input).squeeze(-1)) * 0.5
            
            if temperature == 0:
                next_w_discrete = w_logits.argmax(dim=-1)
            else:
                w_probs = F.softmax(w_logits / temperature, dim=-1)
                next_w_discrete = torch.multinomial(w_probs, 1).squeeze(-1)
            
            # 计算w的连续值用于后续预测
            w_continuous_base = self.continuous_from_discrete(next_w_discrete, self.num_discrete_w, self.continuous_range_w)
            w_continuous = w_continuous_base + w_delta
            w_embed = self.get_continuous_embed('w', w_continuous)
            
            # 预测h（高度）- 使用连续值embedding
            h_input = torch.cat([next_embed, x_embed, y_embed, z_embed, w_embed], dim=-1)
            h_logits = self.to_h_logits(h_input)
            h_delta = torch.tanh(self.to_h_delta(h_input).squeeze(-1)) * 0.5
            
            if temperature == 0:
                next_h_discrete = h_logits.argmax(dim=-1)
            else:
                h_probs = F.softmax(h_logits / temperature, dim=-1)
                next_h_discrete = torch.multinomial(h_probs, 1).squeeze(-1)
            
            # 计算h的连续值用于后续预测
            h_continuous_base = self.continuous_from_discrete(next_h_discrete, self.num_discrete_h, self.continuous_range_h)
            h_continuous = h_continuous_base + h_delta
            h_embed = self.get_continuous_embed('h', h_continuous)
            
            # 预测l（长度）- 使用连续值embedding
            l_input = torch.cat([next_embed, x_embed, y_embed, z_embed, w_embed, h_embed], dim=-1)
            l_logits = self.to_l_logits(l_input)
            l_delta = torch.tanh(self.to_l_delta(l_input).squeeze(-1)) * 0.5
            
            if temperature == 0:
                next_l_discrete = l_logits.argmax(dim=-1)
            else:
                l_probs = F.softmax(l_logits / temperature, dim=-1)
                next_l_discrete = torch.multinomial(l_probs, 1).squeeze(-1)
            
            # 计算l的连续值
            l_continuous_base = self.continuous_from_discrete(next_l_discrete, self.num_discrete_l, self.continuous_range_l)
            l_continuous = l_continuous_base + l_delta
            
            # 使用已经计算好的连续值（包含delta）
            x_center_pred = x_continuous
            y_center_pred = y_continuous
            z_center_pred = z_continuous
            w_center_pred = w_continuous
            h_center_pred = h_continuous
            l_center_pred = l_continuous

            # 连续值已经在上面计算好了（包含delta），直接使用并应用范围限制
            x_continuous = x_center_pred.clamp(self.continuous_range_x[0], self.continuous_range_x[1])
            y_continuous = y_center_pred.clamp(self.continuous_range_y[0], self.continuous_range_y[1])
            z_continuous = z_center_pred.clamp(self.continuous_range_z[0], self.continuous_range_z[1])
            w_continuous = w_center_pred.clamp(self.continuous_range_w[0], self.continuous_range_w[1])
            h_continuous = h_center_pred.clamp(self.continuous_range_h[0], self.continuous_range_h[1])
            l_continuous = l_center_pred.clamp(self.continuous_range_l[0], self.continuous_range_l[1])
            
            # 只为未停止的样本保存3D结果
            for i in range(batch_size):
                if not stopped_samples[i]:
                    generated_results['x'][i].append(x_continuous[i])
                    generated_results['y'][i].append(y_continuous[i])
                    generated_results['z'][i].append(z_continuous[i])
                    generated_results['w'][i].append(w_continuous[i])
                    generated_results['h'][i].append(h_continuous[i])
                    generated_results['l'][i].append(l_continuous[i])
            
            # 预测EOS
            eos_logits = self.to_eos_logits(next_embed).squeeze(-1)
            eos_prob = torch.sigmoid(eos_logits)
            new_stopped = eos_prob > eos_threshold
            stopped_samples = stopped_samples | new_stopped
            if debug:
                print(f"Step {step}: EOS probs = {eos_prob.tolist()}, stopped(next) = {stopped_samples.tolist()}")
            
            # 编码3D预测结果并添加到序列
            pred_embed, _ = self.encode_primitive(
                x_continuous.unsqueeze(0), y_continuous.unsqueeze(0), z_continuous.unsqueeze(0),
                w_continuous.unsqueeze(0), h_continuous.unsqueeze(0), l_continuous.unsqueeze(0),
                torch.ones_like(x_continuous, dtype=torch.bool).unsqueeze(0)
            )

            pred_embed = pred_embed.transpose(0, 1)
            current_sequence = torch.cat([current_sequence, pred_embed], dim=1)
        
        # 将3D结果转换为张量格式
        max_len = max(len(generated_results['x'][i]) for i in range(batch_size))
        
        if max_len == 0:
            return None
        
        # 创建3D结果张量
        result = {
            'x': torch.zeros(batch_size, max_len, device=device),
            'y': torch.zeros(batch_size, max_len, device=device),
            'z': torch.zeros(batch_size, max_len, device=device),
            'w': torch.zeros(batch_size, max_len, device=device),
            'h': torch.zeros(batch_size, max_len, device=device),
            'l': torch.zeros(batch_size, max_len, device=device),
        }
        
        for i in range(batch_size):
            seq_len = len(generated_results['x'][i])
            if seq_len > 0:
                result['x'][i, :seq_len] = torch.stack(generated_results['x'][i])
                result['y'][i, :seq_len] = torch.stack(generated_results['y'][i])
                result['z'][i, :seq_len] = torch.stack(generated_results['z'][i])
                result['w'][i, :seq_len] = torch.stack(generated_results['w'][i])
                result['h'][i, :seq_len] = torch.stack(generated_results['h'][i])
                result['l'][i, :seq_len] = torch.stack(generated_results['l'][i])
        
        return result
    
    # ======================== 增量推理相关代码 ========================
    
    def initialize_incremental_generation(
        self,
        image: Tensor,
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5
    ) -> IncrementalState:
        """
        初始化增量生成状态
        
        Args:
            image: [B, 6, H, W] RGBXYZ输入图像
            max_seq_len: 最大序列长度
            temperature: 采样温度
            eos_threshold: EOS阈值
            
        Returns:
            state: 初始化的增量状态
        """
        batch_size = image.shape[0]
        max_seq_len = max_seq_len or self.max_seq_len
        device = image.device
        
        # 1. 编码图像（只计算一次）
        with torch.no_grad():
            image_embed = self.image_encoder(image)
            
            # 添加2D位置编码
            H = W = int(np.sqrt(image_embed.shape[1]))
            if H * W == image_embed.shape[1]:
                pos_embed_2d = build_2d_sine_positional_encoding(H, W, image_embed.shape[-1])
                pos_embed_2d = pos_embed_2d.flatten(0, 1).unsqueeze(0).to(image_embed.device)
                image_embed = image_embed + pos_embed_2d
            
            # 准备图像条件化
            image_cond = None
            if self.condition_on_image and self.image_film_cond is not None:
                pooled_image_embed = image_embed.mean(dim=1)
                image_cond = self.image_cond_proj_film(pooled_image_embed)
        
        # 2. 初始化序列状态
        current_sequence = repeat(self.sos_token, 'n d -> b n d', b=batch_size)
        
        # 3. 初始化生成结果跟踪
        generated_boxes = {
            'x': [[] for _ in range(batch_size)],
            'y': [[] for _ in range(batch_size)],
            'z': [[] for _ in range(batch_size)],
            'w': [[] for _ in range(batch_size)],
            'h': [[] for _ in range(batch_size)],
            'l': [[] for _ in range(batch_size)],
            'roll': [[] for _ in range(batch_size)],
            'pitch': [[] for _ in range(batch_size)],
            'yaw': [[] for _ in range(batch_size)],
        }
        
        stopped_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 4. 创建状态对象
        state = IncrementalState(
            current_sequence=current_sequence,
            image_embed=image_embed,
            image_cond=image_cond,
            stopped_samples=stopped_samples,
            current_step=0,
            decoder_cache=None,
            gateloop_cache=[],
            generated_boxes=generated_boxes
        )
        
        return state
    
    def generate_next_box_incremental(
        self, 
        state: IncrementalState, 
        temperature: float = 1.0,
        eos_threshold: float = 0.5
    ) -> Tuple[Optional[Dict], bool]:
        """
        增量生成下一个box，参考PrimitiveAnything的实现使用真正的KV缓存
        
        Args:
            state: 增量生成状态（包含多级缓存）
            temperature: 采样温度
            eos_threshold: EOS阈值
            
        Returns:
            box_prediction: 预测的box属性字典，如果停止则为None
            all_stopped: 是否所有样本都停止生成
        """
        if torch.all(state.stopped_samples):
            return None, True
        
        batch_size = state.current_sequence.shape[0]
        device = state.current_sequence.device
        current_len = state.current_sequence.shape[1]
        
        # 参考PrimitiveAnything的forward方法结构
        if state.current_step == 0:
            # 第一步：完整前向传播，初始化所有缓存
            primitive_codes = state.current_sequence  # [B, current_len, dim]
            
            # 添加位置编码
            pos_embed = self.pos_embed[:, :current_len, :]
            primitive_codes = primitive_codes + pos_embed
            
            # 图像条件化
            if state.image_cond is not None:
                primitive_codes = self.image_film_cond(primitive_codes, state.image_cond)
            
            # 门控循环块（如果存在）
            if self.gateloop_block is not None:
                primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes, cache=None)
                state.gateloop_cache = gateloop_cache if gateloop_cache is not None else []
            
            # Transformer解码（初始化decoder缓存）
            attended_codes, decoder_cache = self.decoder(
                primitive_codes,
                context=state.image_embed,
                cache=None,  # 第一次调用，无缓存
                return_hiddens=True  # 返回中间状态用于缓存
            )
            
            # 保存decoder缓存
            state.decoder_cache = decoder_cache
            
        else:
            # 后续步骤：只处理新添加的token，使用缓存（真正的增量！）
            new_token = state.current_sequence[:, -1:, :]  # [B, 1, dim] - 只有最新的token
            
            # 🔧 修复：检查位置编码边界，防止索引超出范围
            pos_index = current_len - 1
            if pos_index >= self.pos_embed.shape[1]:
                print(f"⚠️  Position index {pos_index} exceeds pos_embed size {self.pos_embed.shape[1]}")
                print(f"   Using last available position {self.pos_embed.shape[1] - 1}")
                pos_index = self.pos_embed.shape[1] - 1
            
            # 添加位置编码（只对新token）
            pos_embed = self.pos_embed[:, pos_index:pos_index+1, :]
            primitive_codes = new_token + pos_embed
            
            # 图像条件化（只对新token）
            if state.image_cond is not None:
                primitive_codes = self.image_film_cond(primitive_codes, state.image_cond)
            
            # 门控循环块增量计算
            if self.gateloop_block is not None:
                primitive_codes, new_gateloop_cache = self.gateloop_block(
                    primitive_codes, 
                    cache=state.gateloop_cache
                )
                state.gateloop_cache = new_gateloop_cache if new_gateloop_cache is not None else state.gateloop_cache
            
            # 真正的增量Transformer解码！
            attended_codes, new_decoder_cache = self.decoder(
                primitive_codes,  # 只有新token [B, 1, dim]
                context=state.image_embed,
                cache=state.decoder_cache,  # 使用之前的decoder缓存
                return_hiddens=True
            )
            
            # 更新decoder缓存
            state.decoder_cache = new_decoder_cache
        
        # 预测下一个token（只需要最后一个位置）
        # 🔧 修复：添加安全检查，防止attended_codes为空
        if attended_codes.shape[1] == 0:
            print(f"❌ Error: attended_codes shape is {attended_codes.shape}")
            print(f"   current_step: {state.current_step}")
            print(f"   current_sequence shape: {state.current_sequence.shape}")
            print(f"   primitive_codes shape: {primitive_codes.shape}")
            print(f"   image_embed shape: {state.image_embed.shape}")
            if state.current_step == 0:
                print("   This is step 0 (initialization)")
            else:
                print(f"   This is step {state.current_step} (incremental)")
            raise RuntimeError("attended_codes is empty - this shouldn't happen")
        
        next_embed = attended_codes[:, -1, :]
        
        # 按顺序预测各个属性
        box_prediction = {}
        
        # 🔧 修复：直接使用predict_attribute_with_continuous_embed的输出（与训练逻辑一致）
        # 预测x坐标
        x_logits, x_delta, x_continuous, x_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'x', prev_embeds=None, use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用x_continuous，与训练逻辑一致
        box_prediction['x'] = x_continuous
        
        # 预测y坐标  
        y_logits, y_delta, y_continuous, y_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'y', prev_embeds=[x_embed], use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用y_continuous，与训练逻辑一致
        box_prediction['y'] = y_continuous
        
        # 预测z坐标
        z_logits, z_delta, z_continuous, z_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'z', prev_embeds=[x_embed, y_embed], use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用z_continuous，与训练逻辑一致
        box_prediction['z'] = z_continuous
        
        # 预测w（宽度）
        w_logits, w_delta, w_continuous, w_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'w', prev_embeds=[x_embed, y_embed, z_embed], use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用w_continuous，与训练逻辑一致
        box_prediction['w'] = w_continuous
        
        # 预测h（高度）
        h_logits, h_delta, h_continuous, h_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'h', prev_embeds=[x_embed, y_embed, z_embed, w_embed], use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用h_continuous，与训练逻辑一致
        box_prediction['h'] = h_continuous
        
        # 预测l（长度）
        l_logits, l_delta, l_continuous, l_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'l', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed], use_gumbel=False, temperature=temperature
        )
        # ✅ 直接使用l_continuous，与训练逻辑一致
        box_prediction['l'] = l_continuous
        
        # 预测roll（绕x轴旋转）
        roll_logits, roll_delta, roll_continuous, roll_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'roll', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed], use_gumbel=False, temperature=temperature
        )
        box_prediction['roll'] = roll_continuous
        
        # 预测pitch（绕y轴旋转）
        pitch_logits, pitch_delta, pitch_continuous, pitch_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'pitch', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed], use_gumbel=False, temperature=temperature
        )
        box_prediction['pitch'] = pitch_continuous
        
        # 预测yaw（绕z轴旋转）
        yaw_logits, yaw_delta, yaw_continuous, yaw_embed = self.predict_attribute_with_continuous_embed(
            next_embed, 'yaw', prev_embeds=[x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed, pitch_embed], use_gumbel=False, temperature=temperature
        )
        box_prediction['yaw'] = yaw_continuous
        
        # EOS预测 - 传入所有属性的嵌入
        combined_embeds = torch.cat([next_embed, x_embed, y_embed, z_embed, w_embed, h_embed, l_embed, roll_embed, pitch_embed, yaw_embed], dim=-1)
        eos_logits = self.to_eos_logits(combined_embeds).squeeze(-1)  # [B]
        eos_probs = torch.sigmoid(eos_logits)
        
        # 更新停止状态
        new_stops = eos_probs > eos_threshold
        state.stopped_samples = state.stopped_samples | new_stops
        
        # 保存生成结果（只为未停止的样本）
        for i in range(batch_size):
            if not state.stopped_samples[i]:
                for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
                    # 保存tensor而不是float，以便后续stack操作
                    state.generated_boxes[attr][i].append(box_prediction[attr][i:i+1])  # 保持tensor形状
        
        # 🔧 修复Bug: 更新current_sequence以便下一步使用
        # 构建下一步的输入embedding
        next_embeds = []
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
            continuous_val = box_prediction[attr]
            # 获取对应的离散化参数
            num_discrete = getattr(self, f'num_discrete_{attr}')
            continuous_range = getattr(self, f'continuous_range_{attr}')
            
            # 离散化连续值
            attr_discrete = self.discretize(continuous_val, num_discrete, continuous_range)
            attr_embed = getattr(self, f'{attr}_embed')(attr_discrete)
            next_embeds.append(attr_embed)
        
        # 组合所有属性的embedding
        combined_embed = torch.cat(next_embeds, dim=-1)  # [B, total_embed_dim]
        projected_embed = self.project_in(combined_embed).unsqueeze(1)  # [B, 1, model_dim]
        
        # 更新当前序列（这是关键的修复！）
        state.current_sequence = torch.cat([state.current_sequence, projected_embed], dim=1)
        
        # 更新步数
        state.current_step += 1
        
        # 检查是否所有样本都停止
        all_stopped = torch.all(state.stopped_samples)
        
        return box_prediction, all_stopped
    
    def _sample_discrete(self, logits: Tensor, temperature: float) -> Tensor:
        """离散采样"""
        if temperature == 0:
            return logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            return torch.multinomial(probs, 1).squeeze(-1)
    
    def _compute_continuous_value_from_discrete_delta(self, discrete: Tensor, delta: Tensor, attr: str) -> Tensor:
        """从离散值和delta计算连续值"""
        # 获取属性配置
        if attr == 'x':
            num_bins, value_range = self.num_discrete_x, self.continuous_range_x
        elif attr == 'y':
            num_bins, value_range = self.num_discrete_y, self.continuous_range_y
        elif attr == 'z':
            num_bins, value_range = self.num_discrete_z, self.continuous_range_z
        elif attr == 'w':
            num_bins, value_range = self.num_discrete_w, self.continuous_range_w
        elif attr == 'h':
            num_bins, value_range = self.num_discrete_h, self.continuous_range_h
        elif attr == 'l':
            num_bins, value_range = self.num_discrete_l, self.continuous_range_l
        else:
            raise ValueError(f"Unknown attribute: {attr}")
        
        # 计算连续值
        continuous_base = self.continuous_from_discrete(discrete, num_bins, value_range)
        continuous_value = continuous_base + delta
        
        # 应用范围限制
        continuous_value = continuous_value.clamp(value_range[0], value_range[1])
        
        return continuous_value
    
    @eval_decorator
    @torch.no_grad()
    def generate_incremental(
        self,
        image: Tensor,
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5,
        return_state: bool = False
    ) -> Dict[str, Tensor]:
        """
        完整的增量生成流程
        
        Args:
            image: [B, 6, H, W] RGBXYZ输入图像
            max_seq_len: 最大序列长度
            temperature: 采样温度
            eos_threshold: EOS阈值
            return_state: 是否返回最终状态
            
        Returns:
            results: 生成的完整序列
        """
        batch_size = image.shape[0]
        max_seq_len = max_seq_len or self.max_seq_len
        device = image.device
        
        # 初始化生成状态
        state = self.initialize_incremental_generation(image, max_seq_len, temperature, eos_threshold)
        
        # 逐步生成
        for step in range(max_seq_len):
            # 🔧 修复：检查序列长度，防止超过位置编码范围
            current_seq_len = state.current_sequence.shape[1]
            if current_seq_len >= self.pos_embed.shape[1]:
                print(f"⚠️  Sequence length {current_seq_len} would exceed pos_embed size {self.pos_embed.shape[1]}")
                print(f"   Stopping generation early at step {step}")
                break
                
            box_prediction, all_stopped = self.generate_next_box_incremental(state, temperature, eos_threshold)
            
            if all_stopped or box_prediction is None:
                break
        
        # 转换结果为张量格式
        results = self._convert_incremental_results_to_tensors(state.generated_boxes, batch_size, device)
        
        if return_state:
            return results, state
        else:
            return results
    
    def _convert_incremental_results_to_tensors(self, generated_boxes: Dict, batch_size: int, device: torch.device) -> Dict[str, Tensor]:
        """将增量生成结果转换为张量格式"""
        max_len = max(len(generated_boxes['x'][i]) for i in range(batch_size))
        
        if max_len == 0:
            return None
        
        result = {}
        for attr in ['x', 'y', 'z', 'w', 'h', 'l', 'roll', 'pitch', 'yaw']:
            result[attr] = torch.zeros(batch_size, max_len, device=device)
            
            for i in range(batch_size):
                seq_len = len(generated_boxes[attr][i])
                if seq_len > 0:
                    # 连接tensor列表，每个元素是[1]形状的tensor
                    result[attr][i, :seq_len] = torch.cat(generated_boxes[attr][i], dim=0)
        
        return result 