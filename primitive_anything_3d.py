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
        完全按照PrimitiveAnything的GateLoopBlock实现
        
        Args:
            x: 输入tensor [B, seq_len, dim]
            cache: 门控循环的缓存列表
            
        Returns:
            x: 输出tensor
            new_caches: 更新后的缓存列表
        """
        received_cache = cache is not None

        if x.numel() == 0:  # 空tensor检查
            # 返回正确长度的空缓存，而不是None
            expected_layers = len(self.gateloops)
            empty_caches = [None] * expected_layers
            return x, empty_caches

        if received_cache:
            # 如果有缓存，分离之前的序列和新token
            prev, x = x[:, :-1], x[:, -1:]

        cache = cache if cache is not None else []
        cache_iter = iter(cache)

        new_caches = []
        for gateloop in self.gateloops:
            layer_cache = next(cache_iter, None)
            out, new_cache = gateloop(x, cache=layer_cache, return_cache=True)
            new_caches.append(new_cache)
            x = x + out

        if received_cache:
            # 如果有缓存，将之前的序列与新处理的token连接
            x = torch.cat((prev, x), dim=-2)

        return x, new_caches


# 首先实现Michelangelo的核心组件
class FourierEmbedder(nn.Module):
    """Michelangelo的傅里叶位置编码器"""
    def __init__(self, num_freqs=8, logspace=True, input_dim=3, include_input=True, include_pi=True):
        super().__init__()
        
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32)
        
        if include_pi:
            frequencies *= torch.pi
            
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.out_dim = self.get_dims(input_dim)
    
    def get_dims(self, input_dim):
        temp = 1 if self.include_input or self.num_freqs == 0 else 0
        out_dim = input_dim * (self.num_freqs * 2 + temp)
        return out_dim
    
    def forward(self, x):
        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                embed = torch.cat([embed, x], dim=-1)
            return embed
        else:
            return x

class ResidualCrossAttentionBlock(nn.Module):
    """Michelangelo的残差交叉注意力块"""
    def __init__(self, device=None, dtype=None, width=768, heads=12, init_scale=0.25, 
                 qkv_bias=True, flash=False, n_data=None):
        super().__init__()
        
        self.width = width
        self.heads = heads
        
        # QKV投影
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        
        # 初始化
        nn.init.normal_(self.c_qkv.weight, std=init_scale)
        nn.init.normal_(self.c_proj.weight, std=init_scale)
        if qkv_bias:
            nn.init.constant_(self.c_qkv.bias, 0.0)
        if self.c_proj.bias is not None:
            nn.init.constant_(self.c_proj.bias, 0.0)
    
    def forward(self, query, data):
        # query: [B, num_latents, width]
        # data: [B, num_points, width]
        
        B, N_latents, _ = query.shape
        B, N_data, _ = data.shape
        
        # 计算QKV
        qkv = self.c_qkv(query)  # [B, num_latents, width*3]
        q, k, v = torch.split(qkv, self.width, dim=-1)
        
        # 从data中获取key和value
        data_kv = self.c_qkv(data)  # [B, num_data, width*3]
        _, k_data, v_data = torch.split(data_kv, self.width, dim=-1)
        
        # 交叉注意力
        scale = 1 / math.sqrt(math.sqrt(self.width // self.heads))
        
        # 重塑为多头格式
        q = q.view(B, N_latents, self.heads, -1)
        k_data = k_data.view(B, N_data, self.heads, -1)
        v_data = v_data.view(B, N_data, self.heads, -1)
        
        # 计算注意力权重
        attn_weights = torch.einsum("bthc,bshc->bhts", q * scale, k_data * scale)
        attn_weights = torch.softmax(attn_weights.float(), dim=-1).type(q.dtype)
        
        # 应用注意力
        out = torch.einsum("bhts,bshc->bthc", attn_weights, v_data).reshape(B, N_latents, -1)
        
        # 输出投影
        out = self.c_proj(out)
        
        return out

class Transformer(nn.Module):
    """Michelangelo的Transformer编码器"""
    def __init__(self, device=None, dtype=None, n_ctx=256, width=768, layers=8, 
                 heads=12, init_scale=0.25, qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.width = width
        self.layers = layers
        
        # 构建Transformer层
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
                init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint
            ) for _ in range(layers)
        ])
    
    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        return x

class ResidualAttentionBlock(nn.Module):
    """Michelangelo的残差注意力块"""
    def __init__(self, device=None, dtype=None, n_ctx=256, width=768, heads=12, 
                 init_scale=0.25, qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # 自注意力
        self.attn = MultiheadAttention(
            device=device, dtype=dtype, n_ctx=n_ctx, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash
        )
        
        # 前馈网络
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        
        # LayerNorm
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)
    
    def forward(self, x):
        # 自注意力 + 残差连接
        x = x + self.attn(self.ln_1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiheadAttention(nn.Module):
    """Michelangelo的多头注意力"""
    def __init__(self, device=None, dtype=None, n_ctx=256, width=768, heads=12, 
                 init_scale=0.25, qkv_bias=True, flash=False):
        super().__init__()
        
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        
        # 初始化
        nn.init.normal_(self.c_qkv.weight, std=init_scale)
        nn.init.normal_(self.c_proj.weight, std=init_scale)
        if qkv_bias:
            nn.init.constant_(self.c_qkv.bias, 0.0)
        if self.c_proj.bias is not None:
            nn.init.constant_(self.c_proj.bias, 0.0)
    
    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x
    
    def attention(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        weight = torch.einsum("bthc,bshc->bhts", q * scale, k * scale)
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)
        return out

class MLP(nn.Module):
    """Michelangelo的MLP"""
    def __init__(self, device=None, dtype=None, width=768, init_scale=0.25):
        super().__init__()
        
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        
        # 初始化
        nn.init.normal_(self.c_fc.weight, std=init_scale)
        nn.init.normal_(self.c_proj.weight, std=init_scale)
        nn.init.constant_(self.c_fc.bias, 0.0)
        nn.init.constant_(self.c_proj.bias, 0.0)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class CrossAttentionEncoder(nn.Module):
    """Michelangelo的交叉注意力编码器"""
    def __init__(self, device=None, dtype=None, num_latents=256, fourier_embedder=None,
                 point_feats=3, width=768, heads=12, layers=8, init_scale=0.25,
                 qkv_bias=True, flash=False, use_ln_post=False, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        
        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)
        
        # 傅里叶编码器和输入投影
        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        
        # 交叉注意力
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash
        )
        
        # 自注意力Transformer
        self.self_attn = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width, layers=layers,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, use_checkpoint=False
        )
        
        # LayerNorm
        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None
    
    def forward(self, pc, feats):
        """
        Args:
            pc: [B, N, 3] - XYZ坐标
            feats: [B, N, C] - RGB特征
        Returns:
            latents: [B, num_latents, width]
            pc: [B, N, 3]
        """
        bs = pc.shape[0]
        
        # 傅里叶位置编码
        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)
        
        # 交叉注意力
        query = self.query.unsqueeze(0).expand(bs, -1, -1)
        latents = self.cross_attn(query, data)
        
        # 自注意力
        latents = self.self_attn(latents)
        
        # LayerNorm
        if self.ln_post is not None:
            latents = self.ln_post(latents)
        
        return latents, pc



@dataclass
class IncrementalState:
    """增量生成状态 - 参考PrimitiveAnything实现"""
    current_sequence: torch.Tensor  # [B, current_len, embed_dim]
    point_cloud_embed: torch.Tensor      # [B, H*W, point_cloud_dim]  
    point_cloud_cond: torch.Tensor       # [B, point_cloud_cond_dim]
    stopped_samples: torch.Tensor  # [B] 布尔值，标记哪些样本已停止
    current_step: int              # 当前步数
    
    # 多级KV缓存用于真正的增量解码（参考PrimitiveAnything）
    decoder_cache: Optional[object] = None  # Transformer decoder的cache
    gateloop_cache: Optional[List] = None   # 门控循环块的cache
    
    # 生成结果跟踪
    generated_boxes: Optional[Dict[str, List]] = None
    
    def __post_init__(self):
        if self.gateloop_cache is None:
            # 注意：这里不能直接设置长度，因为gateloop_block可能还没有初始化
            # 实际的长度会在initialize_incremental_generation中设置
            self.gateloop_cache = []
        if self.generated_boxes is None:
            # 这里会在initialize_incremental_generation中正确设置
            pass

class PrimitiveTransformer3D(nn.Module):
    """3D基本体变换器 - 支持RGBXYZ输入和3D箱子生成"""
    def __init__(
        self,
        *,
        # 离散化参数 - 3D坐标
        num_discrete_x = 128,
        num_discrete_y = 128,
        num_discrete_z = 128,  # 新增z坐标
        num_discrete_w = 64,
        num_discrete_h = 64,
        num_discrete_l = 64,  # 新增length维度
        
        # 连续范围 - 3D坐标
        continuous_range_x = [0.5, 2.5],
        continuous_range_y = [-2, 2],
        continuous_range_z = [-1.5, 1.5],  # 新增z范围
        continuous_range_w = [0.3, 0.7],
        continuous_range_h = [0.3, 0.7],
        continuous_range_l = [0.3, 0.7],  # 新增length范围
        
        # 嵌入维度 - 3D
        dim_x_embed = 64,
        dim_y_embed = 64,
        dim_z_embed = 64,  # 新增z嵌入
        dim_w_embed = 32,
        dim_h_embed = 32,
        dim_l_embed = 32,  # 新增length嵌入
        
        # 模型参数
        dim = 512,
        max_primitive_len = 10,
        attn_depth = 6,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_dropout = 0.0,  # 注意力dropout
        ff_dropout = 0.0,    # 前馈dropout
        
        # 点云编码器 - 支持RGBXYZ输入
        point_cloud_encoder_dim = 512,
        use_point_cloud_encoder = True,
        point_cloud_encoder_config = {
            'num_latents': 256,
            'embed_dim': 64,
            'point_feats': 3,  # RGB特征
            'num_freqs': 8,
            'heads': 12,
            'width': 768,
            'num_encoder_layers': 8,
            'num_decoder_layers': 16,
            'pretrained': False  # 从头训练
        },
        
        # 其他参数
        shape_cond_with_cat = False,
        condition_on_point_cloud = True,
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
        
        # 3D连续范围
        self.continuous_range_x = continuous_range_x
        self.continuous_range_y = continuous_range_y
        self.continuous_range_z = continuous_range_z  # 新增
        self.continuous_range_w = continuous_range_w
        self.continuous_range_h = continuous_range_h
        self.continuous_range_l = continuous_range_l  # 新增
        
        # 其他参数
        self.shape_cond_with_cat = shape_cond_with_cat
        self.condition_on_point_cloud = condition_on_point_cloud
        self.gateloop_depth = gateloop_depth
        self.gateloop_use_heinsen = gateloop_use_heinsen
        
        # 点云条件投影层
        if shape_cond_with_cat:
            self.point_cloud_cond_proj = nn.Linear(point_cloud_encoder_dim, dim)
        else:
            self.point_cloud_cond_proj = None
        
        # 点云条件化层
        if condition_on_point_cloud:
            self.point_cloud_film_cond = FiLM(dim, dim)
            self.point_cloud_cond_proj_film = nn.Linear(point_cloud_encoder_dim, self.point_cloud_film_cond.to_gamma.in_features)
        else:
            self.point_cloud_film_cond = None
            self.point_cloud_cond_proj_film = None
        
        # 门控循环块
        if gateloop_depth > 0:
            self.gateloop_block = GateLoopBlock(dim, depth=gateloop_depth, use_heinsen=gateloop_use_heinsen)
        else:
            self.gateloop_block = None
        
        # 点云编码器 - 处理RGBXYZ数据
        if use_point_cloud_encoder:
            # 直接使用完整的Michelangelo架构
            from michelangelo_point_cloud_encoder import AdvancedPointCloudEncoder
            self.point_cloud_encoder = AdvancedPointCloudEncoder(
                output_dim=point_cloud_encoder_dim,
                **point_cloud_encoder_config
            )
            
            # 添加投影层 - 修复维度计算
            # pc_embed的实际维度 = width + embed_dim (如果有VAE)
            actual_pc_embed_dim = point_cloud_encoder_config.get('width', 256) + point_cloud_encoder_config.get('embed_dim', 32)
            actual_pc_head_dim = point_cloud_encoder_config.get('width', 256)
            
            print(f"Point Cloud Encoder Dimensions:")
            print(f"  width: {point_cloud_encoder_config.get('width', 256)}")
            print(f"  embed_dim: {point_cloud_encoder_config.get('embed_dim', 32)}")
            print(f"  actual_pc_embed_dim: {actual_pc_embed_dim}")
            print(f"  actual_pc_head_dim: {actual_pc_head_dim}")
            print(f"  point_cloud_encoder_dim: {point_cloud_encoder_dim}")
            
            self.to_cond_dim = nn.Linear(actual_pc_embed_dim, point_cloud_encoder_dim)
            self.to_cond_dim_head = nn.Linear(actual_pc_head_dim, point_cloud_encoder_dim)
            
            print("Using Advanced Michelangelo Point Cloud Encoder")
        else:
            raise ValueError("Point cloud encoder is required. Set use_point_cloud_encoder=True")
        
        # 3D嵌入层
        self.x_embed = nn.Embedding(num_discrete_x, dim_x_embed)
        self.y_embed = nn.Embedding(num_discrete_y, dim_y_embed)
        self.z_embed = nn.Embedding(num_discrete_z, dim_z_embed)  # 新增
        self.w_embed = nn.Embedding(num_discrete_w, dim_w_embed)
        self.h_embed = nn.Embedding(num_discrete_h, dim_h_embed)
        self.l_embed = nn.Embedding(num_discrete_l, dim_l_embed)  # 新增
        
        # 投影层 - 更新总维度
        total_embed_dim = dim_x_embed + dim_y_embed + dim_z_embed + dim_w_embed + dim_h_embed + dim_l_embed
        self.project_in = nn.Linear(total_embed_dim, dim)
        
        # 连续值到embedding的转换层（用于属性间依赖）
        self.continuous_to_x_embed = nn.Linear(1, dim_x_embed)
        self.continuous_to_y_embed = nn.Linear(1, dim_y_embed)
        self.continuous_to_z_embed = nn.Linear(1, dim_z_embed)
        self.continuous_to_w_embed = nn.Linear(1, dim_w_embed)
        self.continuous_to_h_embed = nn.Linear(1, dim_h_embed)
        self.continuous_to_l_embed = nn.Linear(1, dim_l_embed)
        
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
            cross_attn_dim_context=point_cloud_encoder_dim,
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
        
        # EOS预测网络
        self.to_eos_logits = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )
        
        # 特殊token
        self.sos_token = nn.Parameter(torch.randn(1, dim))
        self.pad_id = pad_id
        self.max_seq_len = max_primitive_len
        
        # 注意：原始PrimitiveAnything不使用位置编码
        
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
    
    def encode_primitive(self, x, y, z, w, h, l, primitive_mask):
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
                            torch.zeros(batch_size, 0, dtype=torch.long, device=x.device))
            return empty_embed, empty_discrete
        
        # 3D离散化
        discrete_x = self.discretize(x, self.num_discrete_x, self.continuous_range_x)
        discrete_y = self.discretize(y, self.num_discrete_y, self.continuous_range_y)
        discrete_z = self.discretize(z, self.num_discrete_z, self.continuous_range_z)  # 新增
        discrete_w = self.discretize(w, self.num_discrete_w, self.continuous_range_w)
        discrete_h = self.discretize(h, self.num_discrete_h, self.continuous_range_h)
        discrete_l = self.discretize(l, self.num_discrete_l, self.continuous_range_l)  # 新增
        
        # 3D嵌入
        x_embed = self.x_embed(discrete_x)
        y_embed = self.y_embed(discrete_y)
        z_embed = self.z_embed(discrete_z)  # 新增
        w_embed = self.w_embed(discrete_w)
        h_embed = self.h_embed(discrete_h)
        l_embed = self.l_embed(discrete_l)  # 新增
        
        # 组合3D特征
        primitive_embed, _ = pack([x_embed, y_embed, z_embed, w_embed, h_embed, l_embed], 'b np *')
        primitive_embed = self.project_in(primitive_embed)
        
        # 使用primitive_mask将无效位置的embedding设置为0
        primitive_embed = primitive_embed.masked_fill(~primitive_mask.unsqueeze(-1), 0.)
        
        return primitive_embed, (discrete_x, discrete_y, discrete_z, discrete_w, discrete_h, discrete_l)
    
    def forward(
        self,
        *,
        x: Tensor,
        y: Tensor,
        z: Tensor,  # 新增z坐标
        w: Tensor,
        h: Tensor,
        l: Tensor,  # 新增length
        point_clouds: List[Tensor],  # 变长点云数据列表
    ):
        """3D前向传播 - 完全适配点云处理"""
        # 创建3D mask
        primitive_mask = (x != self.pad_id) & (y != self.pad_id) & (z != self.pad_id) & (w != self.pad_id) & (h != self.pad_id) & (l != self.pad_id)
        
        # 编码3D基本体
        codes, discrete_coords = self.encode_primitive(x, y, z, w, h, l, primitive_mask)
        
        # 调试信息：检查是否有有效的box
        if codes.shape[1] == 0:
            print(f"Warning: No valid boxes found in batch. codes.shape: {codes.shape}")
            print(f"  primitive_mask sum: {primitive_mask.sum()}")
            print(f"  x non-pad count: {(x != self.pad_id).sum()}")
            print(f"  y non-pad count: {(y != self.pad_id).sum()}")
            print(f"  z non-pad count: {(z != self.pad_id).sum()}")
            print(f"  w non-pad count: {(w != self.pad_id).sum()}")
            print(f"  h non-pad count: {(h != self.pad_id).sum()}")
            print(f"  l non-pad count: {(l != self.pad_id).sum()}")

        # 使用点云编码器处理变长点云数据 - 完全按照原始PrimitiveAnything实现
        pc_head, pc_embed = self.point_cloud_encoder(point_clouds)  # pc_head: [B, 1, width], pc_embed: [B, num_latents-1, width+embed_dim]
        
        # 按照原始实现拼接pc_head和pc_embed
        pc_head_proj = self.to_cond_dim_head(pc_head)  # [B, 1, point_cloud_encoder_dim]
        pc_embed_proj = self.to_cond_dim(pc_embed)  # [B, num_latents-1, point_cloud_encoder_dim]
        pc_embed = torch.cat([pc_head_proj, pc_embed_proj], dim=-2)  # [B, num_latents, point_cloud_encoder_dim]
        
        # 构建输入序列
        batch_size, seq_len, _ = codes.shape  # codes: [B, seq_len, dim]
        device = codes.device
        
        history = codes  # [B, seq_len, dim]
        sos = repeat(self.sos_token, 'n d -> b n d', b=batch_size)  # [B, 1, dim]
        
        primitive_codes, packed_sos_shape = pack([sos, history], 'b * d')  # [B, seq_len+1, dim]
        
        # 点云条件化处理 - 使用全局特征
        if self.condition_on_point_cloud and self.point_cloud_film_cond is not None:
            # 使用pc_head的全局特征进行条件化（类似原始实现）
            pooled_pc_embed = pc_embed.mean(dim=1)  # [B, point_cloud_encoder_dim]
            point_cloud_cond = self.point_cloud_cond_proj_film(pooled_pc_embed)  # [B, dim]
            primitive_codes = self.point_cloud_film_cond(primitive_codes, point_cloud_cond)  # [B, seq_len+1, dim]
        
        # 门控循环块处理
        if self.gateloop_block is not None:
            primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes, cache=None)  # primitive_codes: [B, seq_len+1, dim]
        
        # 使用点云特征作为context进行交叉注意力
        attended_codes = self.decoder(
            primitive_codes,  # [B, seq_len+1, dim]
            context=pc_embed,  # [B, num_latents, point_cloud_encoder_dim]
        )  # attended_codes: [B, seq_len+1, dim]

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
        point_clouds: List[Tensor]
    ):
        """带预测输出的前向传播，用于训练 - 完全适配点云处理"""
        # 先调用标准前向传播获取attended_codes
        attended_codes = self.forward(
            x=x, y=y, z=z, w=w, h=h, l=l, point_clouds=point_clouds
        )
        
        # attended_codes shape: [batch_size, seq_len, model_dim]
        batch_size, seq_len, _ = attended_codes.shape
        
        # 为每个序列位置计算预测
        all_logits = {f'{attr}_logits': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        all_deltas = {f'{attr}_delta': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        all_continuous = {f'{attr}_continuous': [] for attr in ['x', 'y', 'z', 'w', 'h', 'l']}
        eos_logits_list = []
        
        for t in range(seq_len):
            step_embed = attended_codes[:, t, :]  # [batch_size, model_dim]
            
            # 累积的embed用于后续属性预测
            x_embed = y_embed = z_embed = w_embed = h_embed = None
            
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
            
            # 预测EOS
            eos_logit = self.to_eos_logits(step_embed).squeeze(-1)
            
            # 收集结果
            all_logits['x_logits'].append(x_logits)
            all_logits['y_logits'].append(y_logits)
            all_logits['z_logits'].append(z_logits)
            all_logits['w_logits'].append(w_logits)
            all_logits['h_logits'].append(h_logits)
            all_logits['l_logits'].append(l_logits)
            
            all_deltas['x_delta'].append(x_delta)
            all_deltas['y_delta'].append(y_delta)
            all_deltas['z_delta'].append(z_delta)
            all_deltas['w_delta'].append(w_delta)
            all_deltas['h_delta'].append(h_delta)
            all_deltas['l_delta'].append(l_delta)
            
            all_continuous['x_continuous'].append(x_continuous)
            all_continuous['y_continuous'].append(y_continuous)
            all_continuous['z_continuous'].append(z_continuous)
            all_continuous['w_continuous'].append(w_continuous)
            all_continuous['h_continuous'].append(h_continuous)
            all_continuous['l_continuous'].append(l_continuous)
            
            eos_logits_list.append(eos_logit)
        
        # 组装最终输出
        logits_dict = {}
        delta_dict = {}
        continuous_dict = {}
        
        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
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
        point_clouds: List[Tensor],  # 变长点云数据列表
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5,
        debug: bool = False
    ):
        """3D autoregressive生成 - 完全适配点云处理"""
        max_seq_len = max_seq_len or self.max_seq_len
        batch_size = len(point_clouds)
        device = point_clouds[0].device
        
        # 使用点云编码器处理变长点云数据 - 完全按照原始PrimitiveAnything实现
        pc_head, pc_embed = self.point_cloud_encoder(point_clouds)  # pc_head: [B, 1, width], pc_embed: [B, num_latents-1, width+embed_dim]
        
        # 按照原始实现拼接pc_head和pc_embed
        pc_head_proj = self.to_cond_dim_head(pc_head)  # [B, 1, point_cloud_encoder_dim]
        pc_embed_proj = self.to_cond_dim(pc_embed)  # [B, num_latents-1, point_cloud_encoder_dim]
        pc_embed = torch.cat([pc_head_proj, pc_embed_proj], dim=-2)  # [B, num_latents, point_cloud_encoder_dim]
        
        # 为每个样本独立跟踪3D生成结果
        generated_results = {
            'x': [[] for _ in range(batch_size)],  # 每个样本的x坐标列表
            'y': [[] for _ in range(batch_size)],  # 每个样本的y坐标列表
            'z': [[] for _ in range(batch_size)],  # 每个样本的z坐标列表
            'w': [[] for _ in range(batch_size)],  # 每个样本的宽度列表
            'h': [[] for _ in range(batch_size)],  # 每个样本的高度列表
            'l': [[] for _ in range(batch_size)]   # 每个样本的长度列表
        }
        
        # 跟踪每个样本是否已经停止生成
        stopped_samples = torch.zeros(batch_size, dtype=torch.bool, device=rgbxyz.device)  # [B]
        
        # 初始序列：只有SOS token
        current_sequence = repeat(self.sos_token, 'n d -> b n d', b=batch_size)  # [B, 1, dim]
        
        for step in range(max_seq_len):
            # 如果所有样本都停止了，提前结束
            if torch.all(stopped_samples):
                break
            
            primitive_codes = current_sequence  # [B, seq_len, dim]
            
            # 点云条件化处理
            if self.condition_on_point_cloud and self.point_cloud_film_cond is not None:
                pooled_pc_embed = pc_embed.mean(dim=1)  # [B, point_cloud_encoder_dim]
                point_cloud_cond = self.point_cloud_cond_proj_film(pooled_pc_embed)  # [B, dim]
                primitive_codes = self.point_cloud_film_cond(primitive_codes, point_cloud_cond)  # [B, seq_len, dim]
            
            # 门控循环块处理
            if self.gateloop_block is not None:
                primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes, cache=None)  # primitive_codes: [B, seq_len, dim]
            
            # 通过decoder获取attended codes
            attended_codes = self.decoder(
                primitive_codes,  # [B, seq_len, dim]
                context=pc_embed,  # [B, num_latents, point_cloud_encoder_dim]
            )  # attended_codes: [B, seq_len, dim]
            
            # 用最后一个位置预测下一个token
            next_embed = attended_codes[:, -1]  # [B, dim]
            
            # 预测3D坐标和尺寸 - 按顺序：x, y, z, w, h, l
            # 预测x坐标 - 使用连续值embedding
            x_logits = self.to_x_logits(next_embed)  # [B, num_discrete_x]
            x_delta = torch.tanh(self.to_x_delta(next_embed).squeeze(-1)) * 0.5  # [B]
            if temperature == 0:
                next_x_discrete = x_logits.argmax(dim=-1)  # [B]
            else:
                x_probs = F.softmax(x_logits / temperature, dim=-1)  # [B, num_discrete_x]
                next_x_discrete = torch.multinomial(x_probs, 1).squeeze(-1)  # [B]
            
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
        point_clouds: List[Tensor],
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5
    ) -> IncrementalState:
        """
        初始化增量生成状态 - 完全适配点云处理
        
        Args:
            point_clouds: List[Tensor] 变长点云数据列表
            max_seq_len: 最大序列长度
            temperature: 采样温度
            eos_threshold: EOS阈值
            
        Returns:
            state: 初始化的增量状态
        """
        batch_size = len(point_clouds)
        max_seq_len = max_seq_len or self.max_seq_len
        device = point_clouds[0].device
        
        # 1. 使用点云编码器处理变长点云数据（只计算一次）
        with torch.no_grad():
            pc_head, pc_embed = self.point_cloud_encoder(point_clouds)  # pc_head: [B, 1, width], pc_embed: [B, num_latents-1, width+embed_dim]
            
            # 按照原始实现拼接pc_head和pc_embed
            pc_head_proj = self.to_cond_dim_head(pc_head)  # [B, 1, point_cloud_encoder_dim]
            pc_embed_proj = self.to_cond_dim(pc_embed)  # [B, num_latents-1, point_cloud_encoder_dim]
            pc_embed = torch.cat([pc_head_proj, pc_embed_proj], dim=-2)  # [B, num_latents, point_cloud_encoder_dim]
            
            # 准备点云条件化
            point_cloud_cond = None
            if self.condition_on_point_cloud and self.point_cloud_film_cond is not None:
                pooled_pc_embed = pc_embed.mean(dim=1)  # [B, point_cloud_encoder_dim]
                point_cloud_cond = self.point_cloud_cond_proj_film(pooled_pc_embed)  # [B, dim]
        
        # 2. 初始化序列状态
        current_sequence = repeat(self.sos_token, 'n d -> b n d', b=batch_size)  # [B, 1, dim]
        
        # 3. 初始化生成结果跟踪
        generated_boxes = {
            'x': [[] for _ in range(batch_size)],  # 每个样本的x坐标列表
            'y': [[] for _ in range(batch_size)],  # 每个样本的y坐标列表
            'z': [[] for _ in range(batch_size)],  # 每个样本的z坐标列表
            'w': [[] for _ in range(batch_size)],  # 每个样本的宽度列表
            'h': [[] for _ in range(batch_size)],  # 每个样本的高度列表
            'l': [[] for _ in range(batch_size)]   # 每个样本的长度列表
        }
        
        stopped_samples = torch.zeros(batch_size, dtype=torch.bool, device=device)  # [B]
        
        # 4. 创建状态对象
        # 初始化正确长度的gateloop缓存
        if self.gateloop_block is not None:
            expected_layers = len(self.gateloop_block.gateloops)
            gateloop_cache = [None] * expected_layers
        else:
            gateloop_cache = []
        
        state = IncrementalState(
            current_sequence=current_sequence,
            point_cloud_embed=pc_embed,  # 使用拼接后的点云特征
            point_cloud_cond=point_cloud_cond,    # 使用点云条件
            stopped_samples=stopped_samples,
            current_step=0,
            decoder_cache=None,
            gateloop_cache=gateloop_cache,
            generated_boxes=generated_boxes
        )
        
        return state
    
    def _validate_cache_state(self, state: IncrementalState) -> bool:
        """
        验证缓存状态的有效性
        
        Args:
            state: 增量生成状态
            
        Returns:
            bool: 缓存状态是否有效
        """
        # 检查gateloop缓存
        if self.gateloop_block is not None:
            expected_layers = len(self.gateloop_block.gateloops)
            if state.gateloop_cache is not None and len(state.gateloop_cache) != expected_layers:
                return False
        
        # 检查decoder缓存
        if state.decoder_cache is None and state.current_step > 0:
            return False
            
        return True

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
        
        batch_size = state.current_sequence.shape[0]  # B
        device = state.current_sequence.device
        current_len = state.current_sequence.shape[1]  # seq_len
        
        # 验证缓存状态
        if not self._validate_cache_state(state):
            pass  # 缓存状态验证失败，但继续执行
        
        # 参考PrimitiveAnything的forward方法结构
        if state.current_step == 0:
            # 第一步：完整前向传播，初始化所有缓存
            primitive_codes = state.current_sequence  # [B, current_len, dim]
            
            # 点云条件化
            if state.point_cloud_cond is not None:
                primitive_codes = self.point_cloud_film_cond(primitive_codes, state.point_cloud_cond)  # [B, current_len, dim]
            
            # 门控循环块（如果存在）
            if self.gateloop_block is not None:
                primitive_codes, gateloop_cache = self.gateloop_block(primitive_codes, cache=None)  # primitive_codes: [B, current_len, dim]
                # 安全检查：确保gateloop缓存有效
                if gateloop_cache is not None:
                    state.gateloop_cache = gateloop_cache
                else:
                    # 初始化正确长度的缓存
                    expected_layers = len(self.gateloop_block.gateloops)
                    state.gateloop_cache = [None] * expected_layers
                    pass  # gateloop_block返回None缓存，已初始化空缓存
            
            # Transformer解码（初始化decoder缓存）
            attended_codes, decoder_cache = self.decoder(
                primitive_codes,  # [B, current_len, dim]
                context=state.point_cloud_embed,  # [B, num_latents, point_cloud_encoder_dim]
                cache=None,  # 第一次调用，无缓存
                return_hiddens=True  # 返回中间状态用于缓存
            )  # attended_codes: [B, current_len, dim]
            
            # 保存decoder缓存 - 添加安全检查
            if decoder_cache is not None:
                state.decoder_cache = decoder_cache
            else:
                pass  # decoder返回None缓存，已处理
                state.decoder_cache = None
            
        else:
            # 后续步骤：只处理新添加的token，使用缓存（真正的增量！）
            primitive_codes = state.current_sequence[:, -1:, :]  # [B, 1, dim] - 只有最新的token
            
            # 点云条件化（只对新token）
            if state.point_cloud_cond is not None:
                primitive_codes = self.point_cloud_film_cond(primitive_codes, state.point_cloud_cond)
            
            # 门控循环块增量计算
            if self.gateloop_block is not None:
                primitive_codes, new_gateloop_cache = self.gateloop_block(
                    primitive_codes, 
                    cache=state.gateloop_cache
                )
                # 安全检查：确保gateloop缓存更新有效
                if new_gateloop_cache is not None:
                    state.gateloop_cache = new_gateloop_cache
                else:
                        pass  # gateloop_block返回None缓存，保持之前的缓存
                    # 保持之前的缓存不变
            
            # 真正的增量Transformer解码！
            attended_codes, new_decoder_cache = self.decoder(
                primitive_codes,  # 只有新token [B, 1, dim]
                context=state.point_cloud_embed,  # 使用点云特征
                cache=state.decoder_cache,  # 使用之前的decoder缓存
                return_hiddens=True
            )
            
            # 更新decoder缓存 - 添加安全检查
            if new_decoder_cache is not None:
                state.decoder_cache = new_decoder_cache
            else:
                    pass  # decoder返回None缓存，保持之前的缓存
                # 保持之前的缓存不变
        
        # 预测下一个token（只需要最后一个位置）
        # 🔧 修复：添加安全检查，防止attended_codes为空
        if attended_codes.shape[1] == 0:
            print(f"❌ Error: attended_codes shape is {attended_codes.shape}")
            print(f"   current_step: {state.current_step}")
            print(f"   current_sequence shape: {state.current_sequence.shape}")
            print(f"   primitive_codes shape: {primitive_codes.shape}")
            print(f"   point_cloud_embed shape: {state.point_cloud_embed.shape}")
            if state.current_step == 0:
                print("   This is step 0 (initialization)")
            else:
                print(f"   This is step {state.current_step} (incremental)")
            raise RuntimeError("attended_codes is empty - this shouldn't happen")
        
        # 安全检查：确保attended_codes的batch size正确
        if attended_codes.shape[0] != batch_size:
            print(f"❌ Error: attended_codes batch size mismatch: {attended_codes.shape[0]} vs {batch_size}")
            return None, True
        
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
        
        # EOS预测
        eos_logits = self.to_eos_logits(next_embed).squeeze(-1)  # [B]
        eos_probs = torch.sigmoid(eos_logits)
        
        # 更新停止状态
        new_stops = eos_probs > eos_threshold
        state.stopped_samples = state.stopped_samples | new_stops
        
        # 保存生成结果（只为未停止的样本）
        for i in range(batch_size):
            if not state.stopped_samples[i]:
                for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
                    # 保存tensor而不是float，以便后续stack操作
                    state.generated_boxes[attr][i].append(box_prediction[attr][i:i+1])  # 保持tensor形状
        
        # 🔧 修复Bug: 更新current_sequence以便下一步使用
        # 构建下一步的输入embedding
        next_embeds = []
        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
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
        point_clouds: List[Tensor],
        max_seq_len: Optional[int] = None,
        temperature: float = 1.0,
        eos_threshold: float = 0.5,
        return_state: bool = False
    ) -> Dict[str, Tensor]:
        """
        完整的增量生成流程 - 完全适配点云处理
        
        Args:
            point_clouds: List[Tensor] 变长点云数据列表
            max_seq_len: 最大序列长度
            temperature: 采样温度
            eos_threshold: EOS阈值
            return_state: 是否返回最终状态
            
        Returns:
            results: 生成的完整序列
        """
        batch_size = len(point_clouds)
        max_seq_len = max_seq_len or self.max_seq_len
        device = point_clouds[0].device
        
        # 初始化生成状态
        state = self.initialize_incremental_generation(point_clouds, max_seq_len, temperature, eos_threshold)
        
        # 逐步生成
        for step in range(max_seq_len):
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
        for attr in ['x', 'y', 'z', 'w', 'h', 'l']:
            result[attr] = torch.zeros(batch_size, max_len, device=device)
            
            for i in range(batch_size):
                seq_len = len(generated_boxes[attr][i])
                if seq_len > 0:
                    # 连接tensor列表，每个元素是[1]形状的tensor
                    result[attr][i, :seq_len] = torch.cat(generated_boxes[attr][i], dim=0)
        
        return result 