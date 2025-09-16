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
        # 离散化参数 - 3个属性
        num_discrete_position = 64,  # 位置属性 (x, y, z)
        num_discrete_rotation = 64,  # 旋转属性 (roll, pitch, yaw)
        num_discrete_size = 64,     # 尺寸属性 (w, h, l)
        
        # 连续范围 - 3个属性
        continuous_range_position = [[0.5, 2.5], [-2, 2], [-1.5, 1.5]],  # 位置属性 (x, y, z)
        continuous_range_rotation = [[-1.0472, 1.0472], [-1.0472, 1.0472], [-1.0472, 1.0472]],  # 旋转属性 (roll, pitch, yaw) [-π, π] (弧度)
        continuous_range_size = [[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]],  # 尺寸属性 (l, w, h)
        
        # 嵌入维度 - 3个属性，每个属性3维
        dim_position_embed = 64,  # 位置属性embedding维度 (64*3)
        dim_rotation_embed = 64,  # 旋转属性embedding维度 (64*3)
        dim_size_embed = 64,      # 尺寸属性embedding维度 (32*3)
        
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
        
        # 存储3个属性的参数
        self.num_discrete_position = num_discrete_position  # 64
        self.num_discrete_rotation = num_discrete_rotation  # 64
        self.num_discrete_size = num_discrete_size          # 64
        
        self.continuous_range_position = continuous_range_position  # [[x_range], [y_range], [z_range]]
        self.continuous_range_rotation = continuous_range_rotation  # [[roll_range], [pitch_range], [yaw_range]]
        self.continuous_range_size = continuous_range_size          # [[w_range], [h_range], [l_range]]
        
        
        # 其他参数
        self.shape_cond_with_cat = shape_cond_with_cat
        self.condition_on_image = condition_on_image
        self.gateloop_depth = gateloop_depth
        self.gateloop_use_heinsen = gateloop_use_heinsen
        
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
        
        # 3个属性的嵌入层 - 每个属性都是3维向量
        # 位置属性: 3维向量 (x, y, z) - 需要容纳组合索引
        position_vocab_size = self.num_discrete_position ** 3
        self.position_embed = nn.Embedding(position_vocab_size, dim_position_embed)
        
        # 角度属性: 3维向量 (roll, pitch, yaw) - 需要容纳组合索引
        rotation_vocab_size = self.num_discrete_rotation ** 3
        self.rotation_embed = nn.Embedding(rotation_vocab_size, dim_rotation_embed)
        
        # 尺寸属性: 3维向量 (w, h, l) - 需要容纳组合索引
        size_vocab_size = self.num_discrete_size ** 3
        self.size_embed = nn.Embedding(size_vocab_size, dim_size_embed)
        
        # 投影层 - 3个属性的总维度
        total_embed_dim = dim_position_embed + dim_rotation_embed + dim_size_embed
        self.project_in = nn.Linear(total_embed_dim, dim)
        
        # 分组连续值到embedding的转换层（用于属性间依赖）
        # 位置组：一次性输出3维embedding
        self.continuous_to_position_embed = nn.Linear(3, dim_position_embed)
        
        # 角度组：一次性输出3维embedding
        self.continuous_to_rotation_embed = nn.Linear(3, dim_rotation_embed)
        
        # 尺寸组：一次性输出3维embedding
        self.continuous_to_size_embed = nn.Linear(3, dim_size_embed)
        
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
        
        # 位置组预测头 - 一次性输出3维位置 (x, y, z)
        position_total_bins = self.num_discrete_position * 3  # 3个位置维度
        self.to_position_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, position_total_bins),
        )
        
        # 位置组Delta预测头
        self.to_position_delta = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),  # 3维位置delta
        )
        # 旋转组预测头 - 把位置作为输入，一次性输出3维旋转 (roll, pitch, yaw)
        rotation_total_bins = self.num_discrete_rotation * 3  # 3个角度维度
        self.to_rotation_logits = nn.Sequential(
            nn.Linear(dim + dim_position_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, rotation_total_bins),
        )
        
        # 旋转组Delta预测头
        self.to_rotation_delta = nn.Sequential(
            nn.Linear(dim + dim_position_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),  # 3维旋转delta
        )
        
        # 尺寸组预测头 - 把位置+旋转作为输入，一次性输出3维尺寸 (w, h, l)
        size_total_bins = self.num_discrete_size * 3  # 3个尺寸维度
        self.to_size_logits = nn.Sequential(
            nn.Linear(dim + dim_position_embed + dim_rotation_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, size_total_bins),
        )
        
        # 尺寸组Delta预测头
        self.to_size_delta = nn.Sequential(
            nn.Linear(dim + dim_position_embed + dim_rotation_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 3),  # 3维尺寸delta
        )
        
        # EOS预测网络 - 使用所有3个属性的embedding
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim + dim_position_embed + dim_rotation_embed + dim_size_embed, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 特殊token
        self.sos_token = nn.Parameter(torch.randn(1, dim))
        self.pad_id = pad_id
        self.max_seq_len = max_primitive_len
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, max_primitive_len + 1, dim))
        
        print(f"3D PrimitiveTransformer: RGBXYZ(6ch) Input + 3D Box Generation (x,y,z,w,h,l,roll,pitch,yaw)")
    
    def continuous_from_discrete(self, discrete_values, num_bins, value_range):
        """将离散值转换为连续值"""
        min_val, max_val = value_range
        return min_val + (discrete_values.float() / (num_bins - 1)) * (max_val - min_val)
    
    def get_continuous_embed(self, attr_name, continuous_value):
        """从连续值获取embedding - 支持3个属性"""
        # continuous_value应该是[B, 3]或[B, seq_len, 3]的形状
        if continuous_value.dim() == 1:
            continuous_value = continuous_value.unsqueeze(-1)  # [B] -> [B, 1]
        elif continuous_value.dim() == 2:
            if continuous_value.shape[-1] == 3:
                pass  # [B, 3] 正确
            else:
                continuous_value = continuous_value.unsqueeze(-1)  # [B, seq_len] -> [B, seq_len, 1]
        elif continuous_value.dim() == 3:
            if continuous_value.shape[-1] == 3:
                pass  # [B, seq_len, 3] 正确
            else:
                continuous_value = continuous_value.unsqueeze(-1)  # [B, seq_len, 1] -> [B, seq_len, 1]
        
        if attr_name == 'position':
            return self.continuous_to_position_embed(continuous_value)
        elif attr_name == 'rotation':
            return self.continuous_to_rotation_embed(continuous_value)
        elif attr_name == 'size':
            return self.continuous_to_size_embed(continuous_value)
        else:
            raise ValueError(f"Unknown attribute: {attr_name}. Expected 'position', 'rotation', or 'size'")
    
    def predict_3d_vector_with_continuous_embed(self, step_embed, vector_type, prev_embeds=None, use_gumbel=None, temperature=1.0):
        """预测3D向量（位置/旋转/尺寸）并返回连续值和embedding - 3属性版本"""
        # 构建输入
        if prev_embeds is None:
            input_embed = step_embed
        else:
            input_embed = torch.cat([step_embed] + prev_embeds, dim=-1)
        
        if vector_type == 'position':
            # 位置属性: 一次性预测3维向量 (x, y, z)
            logits_head = self.to_position_logits
            delta_head = self.to_position_delta
            num_bins_list = self.num_discrete_position
            value_ranges = self.continuous_range_position
            
        elif vector_type == 'rotation':
            # 旋转属性: 一次性预测3维向量 (roll, pitch, yaw)
            logits_head = self.to_rotation_logits
            delta_head = self.to_rotation_delta
            num_bins_list = self.num_discrete_rotation
            value_ranges = self.continuous_range_rotation
            
        elif vector_type == 'size':
            # 尺寸属性: 一次性预测3维向量 (w, h, l)
            logits_head = self.to_size_logits
            delta_head = self.to_size_delta
            num_bins_list = self.num_discrete_size
            value_ranges = self.continuous_range_size
            
        else:
            raise ValueError(f"Unknown vector type: {vector_type}")
        
        # 一次性预测所有logits
        all_logits = logits_head(input_embed)  # [B, sum(num_bins)]
        all_deltas = delta_head(input_embed)   # [B, 3]
        
        # 决定使用哪种采样方式
        if use_gumbel is None:
            use_gumbel = self.training
        
        # 处理每个维度
        continuous_values = []
        discrete_values = []
        
        start_idx = 0
        for i in range(3):  # 3个维度
            num_bins = num_bins_list  # 现在所有维度使用相同的离散化参数
            value_range = value_ranges[i]
            
            # 提取当前维度的logits和delta
            dim_logits = all_logits[:, start_idx:start_idx + num_bins]
            dim_delta = torch.tanh(all_deltas[:, i]) * 0.5
            
            if use_gumbel:
                # 使用优化的Gumbel Softmax实现（训练时）
                continuous_value = self._differentiable_discrete_to_continuous(dim_logits, num_bins, value_range, temperature)
                # 将连续值转换回离散索引（用于后续处理）
                discrete = self.discretize(continuous_value, num_bins, value_range)
            else:
                # 确定性采样（推理时）
                discrete = torch.argmax(dim_logits, dim=-1)
            
            # 计算连续值
            continuous_base = self.continuous_from_discrete(discrete, num_bins, value_range)
            continuous_value = continuous_base + dim_delta
            continuous_value = continuous_value.clamp(value_range[0], value_range[1])
            
            continuous_values.append(continuous_value)
            discrete_values.append(discrete)
            
            start_idx += num_bins
        
        # 组合成3维向量
        continuous_vector = torch.stack(continuous_values, dim=-1)  # [B, 3]
        discrete_vector = torch.stack(discrete_values, dim=-1)      # [B, 3]
        
        # 获取embedding
        attr_embed = self.get_continuous_embed(vector_type, continuous_vector)
        
        return {
            'logits': all_logits,
            'deltas': all_deltas,
            'continuous': continuous_vector,
            'discrete': discrete_vector,
            'embed': attr_embed
        }
    
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
    
    def encode_primitive(self, position, rotation, size, primitive_mask):
        """编码3D基本体参数（包含旋转）- 使用3属性结构"""
        # 检查是否有有效的框
        if position.numel() == 0 or rotation.numel() == 0 or size.numel() == 0:
            batch_size = position.shape[0] if position.numel() > 0 else 1
            dim = self.project_in.out_features
            empty_embed = torch.zeros(batch_size, 0, dim, device=position.device)
            empty_discrete = (torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device),
                            torch.zeros(batch_size, 0, dtype=torch.long, device=position.device))
            return empty_embed, empty_discrete
        
        # 分离各个属性
        x, y, z = position[:, :, 0], position[:, :, 1], position[:, :, 2]
        roll, pitch, yaw = rotation[:, :, 0], rotation[:, :, 1], rotation[:, :, 2]
        l, w, h = size[:, :, 0], size[:, :, 1], size[:, :, 2]
        
        # 3D离散化（包含旋转）- 使用3属性结构
        # 位置属性
        discrete_x = self.discretize(x, self.num_discrete_position, self.continuous_range_position[0])
        discrete_y = self.discretize(y, self.num_discrete_position, self.continuous_range_position[1])
        discrete_z = self.discretize(z, self.num_discrete_position, self.continuous_range_position[2])
        
        # 旋转属性
        discrete_roll = self.discretize(roll, self.num_discrete_rotation, self.continuous_range_rotation[0])
        discrete_pitch = self.discretize(pitch, self.num_discrete_rotation, self.continuous_range_rotation[1])
        discrete_yaw = self.discretize(yaw, self.num_discrete_rotation, self.continuous_range_rotation[2])
        
        # 尺寸属性
        discrete_w = self.discretize(w, self.num_discrete_size, self.continuous_range_size[0])
        discrete_h = self.discretize(h, self.num_discrete_size, self.continuous_range_size[1])
        discrete_l = self.discretize(l, self.num_discrete_size, self.continuous_range_size[2])
        
        # 3D嵌入（包含旋转）- 使用3属性结构
        # 位置embedding - 处理pad_id
        pos_discrete = discrete_x + discrete_y * self.num_discrete_position + discrete_z * (self.num_discrete_position ** 2)
        pos_discrete = pos_discrete.masked_fill(~primitive_mask, 0)  # 将pad位置设为0
        pos_embed = self.position_embed(pos_discrete)
        
        # 旋转embedding - 处理pad_id
        rot_discrete = discrete_roll + discrete_pitch * self.num_discrete_rotation + discrete_yaw * (self.num_discrete_rotation ** 2)
        rot_discrete = rot_discrete.masked_fill(~primitive_mask, 0)  # 将pad位置设为0
        rot_embed = self.rotation_embed(rot_discrete)
        
        # 尺寸embedding - 处理pad_id
        size_discrete = discrete_w + discrete_h * self.num_discrete_size + discrete_l * (self.num_discrete_size ** 2)
        size_discrete = size_discrete.masked_fill(~primitive_mask, 0)  # 将pad位置设为0
        size_embed = self.size_embed(size_discrete)
        
        # 组合3D特征（包含旋转）
        primitive_embed, _ = pack([pos_embed, rot_embed, size_embed], 'b np *')
        primitive_embed = self.project_in(primitive_embed)
        
        # 使用primitive_mask将无效位置的embedding设置为0
        primitive_embed = primitive_embed.masked_fill(~primitive_mask.unsqueeze(-1), 0.)
        
        return primitive_embed, (discrete_x, discrete_y, discrete_z, discrete_w, discrete_h, discrete_l, discrete_roll, discrete_pitch, discrete_yaw)
    
    def forward(
        self,
        *,
        position: Tensor,  # [B, seq_len, 3] - (x, y, z)
        rotation: Tensor,   # [B, seq_len, 3] - (roll, pitch, yaw)
        size: Tensor,       # [B, seq_len, 3] - (l, w, h)
        image: Tensor,      # [B, 6, H, W] - RGBXYZ
    ):
        """3D前向传播 - 使用3属性结构"""
        # 分离各个属性
        x, y, z = position[:, :, 0], position[:, :, 1], position[:, :, 2]
        roll, pitch, yaw = rotation[:, :, 0], rotation[:, :, 1], rotation[:, :, 2]
        l, w, h = size[:, :, 0], size[:, :, 1], size[:, :, 2]
        
        # 创建3D mask（包含旋转）
        primitive_mask = (x != self.pad_id) & (y != self.pad_id) & (z != self.pad_id) & (w != self.pad_id) & (h != self.pad_id) & (l != self.pad_id) & (roll != self.pad_id) & (pitch != self.pad_id) & (yaw != self.pad_id)
        
        # 编码3D基本体（包含旋转）
        codes, discrete_coords = self.encode_primitive(position, rotation, size, primitive_mask)

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
        position: Tensor,  # [B, seq_len, 3] - (x, y, z)
        rotation: Tensor,   # [B, seq_len, 3] - (roll, pitch, yaw)
        size: Tensor,       # [B, seq_len, 3] - (l, w, h)
        image: Tensor,      # [B, 6, H, W] - RGBXYZ
    ):
        """带预测输出的前向传播，用于训练"""
        # 先调用标准前向传播获取attended_codes
        attended_codes = self.forward(
            position=position, rotation=rotation, size=size, image=image
        )
        
        # attended_codes shape: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = attended_codes.shape
        
        # 为每个序列位置计算预测 - 使用3D向量预测
        all_logits = {'position_logits': [], 'rotation_logits': [], 'size_logits': []}
        all_deltas = {'position_delta': [], 'rotation_delta': [], 'size_delta': []}
        all_continuous = {'position_continuous': [], 'rotation_continuous': [], 'size_continuous': []}
        eos_logits_list = []
        
        for t in range(seq_len):
            step_embed = attended_codes[:, t, :]  # [batch_size, model_dim]
            
            # 按新顺序预测：位置(3D) → 旋转(3D) → 尺寸(3D)
            prev_embeds = []
            
            # 1. 预测位置向量 (x, y, z)
            pos_result = self.predict_3d_vector_with_continuous_embed(
                step_embed, 'position', prev_embeds=prev_embeds, use_gumbel=None, temperature=1.0
            )
            pos_logits = pos_result['logits']
            pos_deltas = pos_result['deltas']
            pos_continuous = pos_result['continuous']
            pos_embeds = [pos_result['embed']]
            prev_embeds.extend(pos_embeds)
            
            # 2. 预测旋转向量 (roll, pitch, yaw)
            rot_result = self.predict_3d_vector_with_continuous_embed(
                step_embed, 'rotation', prev_embeds=prev_embeds, use_gumbel=None, temperature=1.0
            )
            rot_logits = rot_result['logits']
            rot_deltas = rot_result['deltas']
            rot_continuous = rot_result['continuous']
            rot_embeds = [rot_result['embed']]
            prev_embeds.extend(rot_embeds)
            
            # 3. 预测尺寸向量 (w, h, l)
            size_result = self.predict_3d_vector_with_continuous_embed(
                step_embed, 'size', prev_embeds=prev_embeds, use_gumbel=None, temperature=1.0
            )
            size_logits = size_result['logits']
            size_deltas = size_result['deltas']
            size_continuous = size_result['continuous']
            size_embeds = [size_result['embed']]
            prev_embeds.extend(size_embeds)
            
            # 4. 预测EOS
            eos_logit = self.to_eos_logits(torch.cat([step_embed] + prev_embeds, dim=-1)).squeeze(-1)
            
            # 收集结果 - 使用3属性结构
            all_logits['position_logits'].append(pos_logits)
            all_deltas['position_delta'].append(pos_deltas)
            all_continuous['position_continuous'].append(pos_continuous)
            
            all_logits['rotation_logits'].append(rot_logits)
            all_deltas['rotation_delta'].append(rot_deltas)
            all_continuous['rotation_continuous'].append(rot_continuous)
            
            all_logits['size_logits'].append(size_logits)
            all_deltas['size_delta'].append(size_deltas)
            all_continuous['size_continuous'].append(size_continuous)
            
            eos_logits_list.append(eos_logit)
        
        # 组装最终输出 - 转换为9属性格式以匹配loss函数
        logits_dict = {}
        delta_dict = {}
        continuous_dict = {}
        
        # 位置属性: position -> x, y, z
        pos_logits = torch.stack(all_logits['position_logits'], dim=1)  # [B, seq_len, total_bins]
        pos_deltas = torch.stack(all_deltas['position_delta'], dim=1)  # [B, seq_len, 3]
        pos_continuous = torch.stack(all_continuous['position_continuous'], dim=1)  # [B, seq_len, 3]
        
        # 分离位置logits为x, y, z
        bins_per_dim = self.num_discrete_position
        logits_dict['x_logits'] = pos_logits[:, :, :bins_per_dim]
        logits_dict['y_logits'] = pos_logits[:, :, bins_per_dim:2*bins_per_dim]
        logits_dict['z_logits'] = pos_logits[:, :, 2*bins_per_dim:3*bins_per_dim]
        
        delta_dict['x_delta'] = pos_deltas[:, :, 0:1]
        delta_dict['y_delta'] = pos_deltas[:, :, 1:2]
        delta_dict['z_delta'] = pos_deltas[:, :, 2:3]
        
        continuous_dict['x_continuous'] = pos_continuous[:, :, 0:1]
        continuous_dict['y_continuous'] = pos_continuous[:, :, 1:2]
        continuous_dict['z_continuous'] = pos_continuous[:, :, 2:3]
        
        # 旋转属性: rotation -> roll, pitch, yaw
        rot_logits = torch.stack(all_logits['rotation_logits'], dim=1)  # [B, seq_len, total_bins]
        rot_deltas = torch.stack(all_deltas['rotation_delta'], dim=1)  # [B, seq_len, 3]
        rot_continuous = torch.stack(all_continuous['rotation_continuous'], dim=1)  # [B, seq_len, 3]
        
        # 分离旋转logits为roll, pitch, yaw
        bins_per_dim = self.num_discrete_rotation
        logits_dict['roll_logits'] = rot_logits[:, :, :bins_per_dim]
        logits_dict['pitch_logits'] = rot_logits[:, :, bins_per_dim:2*bins_per_dim]
        logits_dict['yaw_logits'] = rot_logits[:, :, 2*bins_per_dim:3*bins_per_dim]
        
        delta_dict['roll_delta'] = rot_deltas[:, :, 0:1]
        delta_dict['pitch_delta'] = rot_deltas[:, :, 1:2]
        delta_dict['yaw_delta'] = rot_deltas[:, :, 2:3]
        
        continuous_dict['roll_continuous'] = rot_continuous[:, :, 0:1]
        continuous_dict['pitch_continuous'] = rot_continuous[:, :, 1:2]
        continuous_dict['yaw_continuous'] = rot_continuous[:, :, 2:3]
        
        # 尺寸属性: size -> w, h, l
        size_logits = torch.stack(all_logits['size_logits'], dim=1)  # [B, seq_len, total_bins]
        size_deltas = torch.stack(all_deltas['size_delta'], dim=1)  # [B, seq_len, 3]
        size_continuous = torch.stack(all_continuous['size_continuous'], dim=1)  # [B, seq_len, 3]
        
        # 分离尺寸logits为w, h, l
        bins_per_dim = self.num_discrete_size
        logits_dict['w_logits'] = size_logits[:, :, :bins_per_dim]
        logits_dict['h_logits'] = size_logits[:, :, bins_per_dim:2*bins_per_dim]
        logits_dict['l_logits'] = size_logits[:, :, 2*bins_per_dim:3*bins_per_dim]
        
        delta_dict['w_delta'] = size_deltas[:, :, 0:1]
        delta_dict['h_delta'] = size_deltas[:, :, 1:2]
        delta_dict['l_delta'] = size_deltas[:, :, 2:3]
        
        continuous_dict['w_continuous'] = size_continuous[:, :, 0:1]
        continuous_dict['h_continuous'] = size_continuous[:, :, 1:2]
        continuous_dict['l_continuous'] = size_continuous[:, :, 2:3]
        
        eos_logits = torch.stack(eos_logits_list, dim=1)
        
        return {
            'logits_dict': logits_dict,
            'delta_dict': delta_dict,
            'continuous_dict': continuous_dict,
            'eos_logits': eos_logits
        }
    
    @eval_decorator
    @torch.no_grad()
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
            'yaw': [[] for _ in range(batch_size)]
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
        
        # 按新顺序预测：位置(3D) → 旋转(3D) → 尺寸(3D)
        box_prediction = {}
        prev_embeds = []
        
        # 1. 预测位置向量 (x, y, z)
        pos_result = self.predict_3d_vector_with_continuous_embed(
            next_embed, 'position', prev_embeds=prev_embeds, use_gumbel=False, temperature=temperature
        )
        pos_logits = pos_result['logits']
        pos_deltas = pos_result['deltas']
        pos_continuous = pos_result['continuous']  # [B, 3]
        pos_embeds = [pos_result['embed']]
        prev_embeds.extend(pos_embeds)
        
        # 将3维张量转换为字典格式
        box_prediction['x'] = pos_continuous[:, 0]  # [B]
        box_prediction['y'] = pos_continuous[:, 1]  # [B]
        box_prediction['z'] = pos_continuous[:, 2]  # [B]
        
        # 2. 预测旋转向量 (roll, pitch, yaw)
        rot_result = self.predict_3d_vector_with_continuous_embed(
            next_embed, 'rotation', prev_embeds=prev_embeds, use_gumbel=False, temperature=temperature
        )
        rot_logits = rot_result['logits']
        rot_deltas = rot_result['deltas']
        rot_continuous = rot_result['continuous']  # [B, 3]
        rot_embeds = [rot_result['embed']]
        prev_embeds.extend(rot_embeds)
        
        # 将3维张量转换为字典格式
        box_prediction['roll'] = rot_continuous[:, 0]  # [B]
        box_prediction['pitch'] = rot_continuous[:, 1]  # [B]
        box_prediction['yaw'] = rot_continuous[:, 2]  # [B]
        
        # 3. 预测尺寸向量 (w, h, l)
        size_result = self.predict_3d_vector_with_continuous_embed(
            next_embed, 'size', prev_embeds=prev_embeds, use_gumbel=False, temperature=temperature
        )
        size_logits = size_result['logits']
        size_deltas = size_result['deltas']
        size_continuous = size_result['continuous']  # [B, 3]
        size_embeds = [size_result['embed']]
        prev_embeds.extend(size_embeds)
        
        # 将3维张量转换为字典格式
        box_prediction['w'] = size_continuous[:, 0]  # [B]
        box_prediction['h'] = size_continuous[:, 1]  # [B]
        box_prediction['l'] = size_continuous[:, 2]  # [B]
        
        # 🔍 添加日志：检查尺寸预测结果
        for i in range(batch_size):
            if not state.stopped_samples[i]:
                w_val = size_continuous[i, 0].item()
                h_val = size_continuous[i, 1].item()
                l_val = size_continuous[i, 2].item()
                
                if w_val <= 0 or h_val <= 0 or l_val <= 0:
                    print(f"🚨 模型预测出无效尺寸 - Sample {i}:")
                    print(f"   尺寸预测: w={w_val:.6f}, h={h_val:.6f}, l={l_val:.6f}")
                    print(f"   位置预测: x={pos_continuous[i, 0].item():.6f}, y={pos_continuous[i, 1].item():.6f}, z={pos_continuous[i, 2].item():.6f}")
                    print(f"   旋转预测: roll={rot_continuous[i, 0].item():.6f}, pitch={rot_continuous[i, 1].item():.6f}, yaw={rot_continuous[i, 2].item():.6f}")
                    print(f"   尺寸logits: {size_logits[i].cpu().numpy()}")
                    print(f"   尺寸deltas: {size_deltas[i].cpu().numpy()}")
                    print(f"   尺寸continuous: {size_continuous[i].cpu().numpy()}")
        
        # EOS预测（更新输入维度）
        eos_logits = self.to_eos_logits(torch.cat([next_embed] + prev_embeds, dim=-1)).squeeze(-1)  # [B]
        eos_probs = torch.sigmoid(eos_logits)
        
        # 更新停止状态
        new_stops = eos_probs > eos_threshold
        state.stopped_samples = state.stopped_samples | new_stops
        
        # 保存生成结果（只为未停止的样本）
        for i in range(batch_size):
            if not state.stopped_samples[i]:
                # 保存位置属性 - pos_continuous是[B, 3]张量
                state.generated_boxes['x'][i].append(pos_continuous[i:i+1, 0:1])
                state.generated_boxes['y'][i].append(pos_continuous[i:i+1, 1:2])
                state.generated_boxes['z'][i].append(pos_continuous[i:i+1, 2:3])
                
                # 保存旋转属性 - rot_continuous是[B, 3]张量
                state.generated_boxes['roll'][i].append(rot_continuous[i:i+1, 0:1])
                state.generated_boxes['pitch'][i].append(rot_continuous[i:i+1, 1:2])
                state.generated_boxes['yaw'][i].append(rot_continuous[i:i+1, 2:3])
                
                # 保存尺寸属性 - size_continuous是[B, 3]张量
                state.generated_boxes['w'][i].append(size_continuous[i:i+1, 0:1])
                state.generated_boxes['h'][i].append(size_continuous[i:i+1, 1:2])
                state.generated_boxes['l'][i].append(size_continuous[i:i+1, 2:3])
            else:
                # 样本被停止，不保存预测值
                pass
        
        # 🔧 修复Bug: 更新current_sequence以便下一步使用
        # 构建下一步的输入embedding - 使用新的3属性结构
        next_embeds = []
        
        # 位置属性embedding - pos_continuous已经是[B, 3]张量
        pos_embed = self.get_continuous_embed('position', pos_continuous)
        next_embeds.append(pos_embed)
        
        # 旋转属性embedding - rot_continuous已经是[B, 3]张量
        rot_embed = self.get_continuous_embed('rotation', rot_continuous)
        next_embeds.append(rot_embed)
        
        # 尺寸属性embedding - size_continuous已经是[B, 3]张量
        size_embed = self.get_continuous_embed('size', size_continuous)
        next_embeds.append(size_embed)
        
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
        
        # 检查结果是否有效
        if results is None:
            return None
        
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
            # 🔧 修复：使用padding值而不是0来初始化
            result[attr] = torch.full((batch_size, max_len), -1.0, device=device)
            
            for i in range(batch_size):
                seq_len = len(generated_boxes[attr][i])
                if seq_len > 0:
                    # 连接tensor列表，每个元素是[1]形状的tensor，然后squeeze为[seq_len]
                    concatenated = torch.cat(generated_boxes[attr][i], dim=0)  # [seq_len, 1]
                    result[attr][i, :seq_len] = concatenated.squeeze(-1)  # [seq_len]
                else:
                    # seq_len为0，使用padding值-1.0（与数据加载器一致）
                    pass
        
        return result 