# -*- coding: utf-8 -*-
"""
完全按照Michelangelo架构实现的点云编码器
包含VAE、交叉注意力、Transformer等完整组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import repeat

class FourierEmbedder(nn.Module):
    """The sin/cosine positional embedding. Given an input tensor `x` of shape [n_batch, ..., c_dim], it converts
    each feature dimension of `x[..., i]` into:
        [
            sin(x[..., i]),
            sin(f_1*x[..., i]),
            sin(f_2*x[..., i]),
            ...
            sin(f_N * x[..., i]),
            cos(x[..., i]),
            cos(f_1*x[..., i]),
            cos(f_2*x[..., i]),
            ...
            cos(f_N * x[..., i]),
            x[..., i]     # only present if include_input is True.
        ], here f_i is the frequency.

    Args:
        num_freqs (int): the number of frequencies, default is 6;
        logspace (bool): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
            otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1)];
        input_dim (int): the input dimension, default is 3;
        include_input (bool): include the input tensor or not, default is True.
        include_pi (bool): include pi in frequencies, default is True.

    Attributes:
        frequencies (torch.Tensor): If logspace is True, then the frequency f_i is [..., 2^(i / num_freqs), ...],
                otherwise, the frequencies are linearly spaced between [1.0, 2^(num_freqs - 1);

        out_dim (int): the embedding size, if include_input is True, it is input_dim * (num_freqs * 2 + 1),
            otherwise, it is input_dim * num_freqs * 2.

    """

    def __init__(self,
                 num_freqs: int = 6,
                 logspace: bool = True,
                 input_dim: int = 3,
                 include_input: bool = True,
                 include_pi: bool = True) -> None:

        """The initialization"""

        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                num_freqs,
                dtype=torch.float32
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32
            )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward process.

        Args:
            x: tensor of shape [..., dim]

        Returns:
            embedding: an embedding of `x` of shape [..., dim * (num_freqs * 2 + temp)]
                where temp is 1 if include_input is True and 0 otherwise.
        """

        if self.num_freqs > 0:
            embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
            if self.include_input:
                return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
            else:
                return torch.cat((embed.sin(), embed.cos()), dim=-1)
        else:
            return x

def init_linear(linear, init_scale):
    """Initialize linear layer with scaled initialization"""
    nn.init.normal_(linear.weight, std=init_scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)

def checkpoint(func, inputs, params, flag):
    """Checkpoint function for gradient checkpointing"""
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs)
    else:
        return func(*inputs)

class MLP(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = data_width or width
        self.flash = flash

        self.c_attn = nn.Linear(width, width * 3, device=device, dtype=dtype, bias=qkv_bias)
        self.c_attn_data = nn.Linear(self.data_width, self.data_width * 2, device=device, dtype=dtype, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        
        init_linear(self.c_attn, init_scale)
        init_linear(self.c_attn_data, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        B, M, C = x.shape
        B, N, D = data.shape
        
        qkv = self.c_attn(x)
        q, k, v = qkv.reshape(B, M, 3, C).permute(2, 0, 1, 3).unbind(0)
        
        kv = self.c_attn_data(data)
        k_data, v_data = kv.reshape(B, N, 2, D).permute(2, 0, 1, 3).unbind(0)
        
        # Concatenate k and v
        k = torch.cat([k, k_data], dim=1)
        v = torch.cat([v, v_data], dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, M, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.heads, C // self.heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, M, C)
        
        return self.c_proj(out)

class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.flash = flash

        self.c_attn = nn.Linear(width, width * 3, device=device, dtype=dtype, bias=qkv_bias)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        
        init_linear(self.c_attn, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.reshape(B, T, 3, C).permute(2, 0, 1, 3).unbind(0)
        
        # Reshape for multi-head attention
        q = q.reshape(B, T, self.heads, C // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(B, T, self.heads, C // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(B, T, self.heads, C // self.heads).permute(0, 2, 1, 3)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, T, C)
        
        return self.c_proj(out)

class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_data: Optional[int] = None,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(data_width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        device: Optional[torch.device],
        dtype: Optional[torch.dtype],
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    flash=flash,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

class CrossAttentionEncoder(nn.Module):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)

class DiagonalGaussianDistribution(nn.Module):
    """Michelangelo的需要分布"""
    def __init__(self, parameters, feat_dim=-1):
        super().__init__()
        self.parameters = parameters
        self.feat_dim = feat_dim
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    
    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        eps = torch.randn_like(std)
        return self.mean + eps * std
    
    def mode(self):
        return self.mean

class CrossAttentionDecoder(nn.Module):
    """Michelangelo的交叉注意力解码器"""
    def __init__(self, device=None, dtype=None, num_latents=256, out_channels=1,
                 fourier_embedder=None, width=768, heads=12, init_scale=0.25,
                 qkv_bias=True, flash=False, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder
        
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)
        
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device, dtype=dtype, n_data=num_latents, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash
        )
        
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)
    
    def forward(self, queries, latents):
        queries = self.query_proj(self.fourier_embedder(queries))
        x = self.cross_attn_decoder(queries, latents)
        x = self.ln_post(x)
        x = self.output_proj(x)
        return x

class ShapeAsLatentPerceiver(nn.Module):
    """完整的Michelangelo ShapeAsLatentPerceiver"""
    def __init__(self, device=None, dtype=None, num_latents=256, point_feats=3,
                 embed_dim=64, num_freqs=8, include_pi=True, width=768, heads=12,
                 num_encoder_layers=8, num_decoder_layers=16, init_scale=0.25,
                 qkv_bias=True, flash=False, use_ln_post=True, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            num_latents=num_latents, point_feats=point_feats, width=width, heads=heads,
            layers=num_encoder_layers, init_scale=init_scale, qkv_bias=qkv_bias,
            flash=flash, use_ln_post=use_ln_post, use_checkpoint=use_checkpoint
        )
        
        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)
        
        self.transformer = Transformer(
            device=device, dtype=dtype, n_ctx=num_latents, width=width, layers=num_decoder_layers,
            heads=heads, init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint
        )
        
        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device, dtype=dtype, fourier_embedder=self.fourier_embedder,
            out_channels=1, num_latents=num_latents, width=width, heads=heads,
            init_scale=init_scale, qkv_bias=qkv_bias, flash=flash, use_checkpoint=use_checkpoint
        )
    
    def encode(self, pc, feats=None, sample_posterior=True):
        latents, center_pos = self.encoder(pc, feats)
        
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
            
            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()
        
        return latents, center_pos, posterior
    
    def decode(self, latents):
        if self.embed_dim > 0:
            latents = self.post_kl(latents)
        return self.transformer(latents)
    
    def query_geometry(self, queries, latents):
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

class AdvancedPointCloudEncoder(nn.Module):
    """高级点云编码器 - 完全按照Michelangelo架构"""
    def __init__(self, output_dim=512, num_latents=256, embed_dim=64, point_feats=3,
                 num_freqs=8, heads=12, width=768, num_encoder_layers=8,
                 num_decoder_layers=16, pretrained=False):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_latents = num_latents
        self.point_feats = point_feats
        
        # 构建完整的Michelangelo架构
        self.perceiver = ShapeAsLatentPerceiver(
            device=None, dtype=None, num_latents=num_latents, point_feats=point_feats,
            embed_dim=embed_dim, num_freqs=num_freqs, include_pi=True, width=width,
            heads=heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
            init_scale=0.25, qkv_bias=True, flash=False, use_ln_post=True, use_checkpoint=False
        )
        
        # 输出投影
        self.output_proj = nn.Linear(width, output_dim)
        
        self.dim_model_out = output_dim
        print(f"AdvancedPointCloudEncoder: Using full Michelangelo architecture -> {output_dim}")
    
    def forward(self, point_clouds):
        """
        处理变长点云数据
        支持同一个batch内不同sample的点云数量不同
        
        Args:
            point_clouds: List[Tensor] - 变长点云数据列表，每个元素是[N_i, 6]
            
        Returns:
            pc_head: [B, 1, width] - 全局特征（类似原始shape_head）
            pc_embed: [B, num_latents-1, width+embed_dim] - 局部特征序列（类似原始shape_embed）
        """
        batch_size = len(point_clouds)
        device = point_clouds[0].device
        
        # 存储每个sample的编码结果
        all_pc_heads = []
        all_pc_embeds = []
        
        # 逐个处理每个sample的点云
        for i, point_cloud in enumerate(point_clouds):
            # point_cloud: [N_i, 6] - 第i个sample的点云数据
            num_points = point_cloud.shape[0]
            
            # 分离RGB和XYZ
            rgb_feats = point_cloud[:, 0:3]  # [N_i, 3] - RGB特征
            xyz_coords = point_cloud[:, 3:6]  # [N_i, 3] - XYZ坐标
            
            # 添加batch维度
            rgb_feats = rgb_feats.unsqueeze(0)  # [1, N_i, 3]
            xyz_coords = xyz_coords.unsqueeze(0)  # [1, N_i, 3]
            
            # 使用Michelangelo编码器
            encoded_latents, _ = self.perceiver.encoder(xyz_coords, rgb_feats)  # [1, num_latents, width]
            
            # 分离全局特征和局部特征
            pc_head = encoded_latents[:, 0:1]  # [1, 1, width]
            pc_embed = encoded_latents[:, 1:]  # [1, num_latents-1, width]
            
            # 如果需要VAE处理
            if self.perceiver.embed_dim > 0:
                moments = self.perceiver.pre_kl(pc_embed)  # [1, num_latents-1, 2*embed_dim]
                posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
                shape_latents = posterior.sample()  # [1, num_latents-1, embed_dim]
                
                # 拼接编码特征和形状潜在向量
                pc_embed = torch.cat([pc_embed, shape_latents], dim=-1)  # [1, num_latents-1, width+embed_dim]
            
            all_pc_heads.append(pc_head)
            all_pc_embeds.append(pc_embed)
        
        # 拼接所有sample的结果
        pc_head = torch.cat(all_pc_heads, dim=0)  # [B, 1, width]
        pc_embed = torch.cat(all_pc_embeds, dim=0)  # [B, num_latents-1, width+embed_dim]
        
        return pc_head, pc_embed

# 使用示例
if __name__ == "__main__":
    # 创建高级点云编码器
    encoder = AdvancedPointCloudEncoder(
        output_dim=512,
        num_latents=256,
        embed_dim=64,
        point_feats=3,
        num_freqs=8,
        heads=12,
        width=768,
        num_encoder_layers=8,
        num_decoder_layers=16,
        pretrained=False
    )
    
    # 测试输入
    batch_size = 2
    height, width = 64, 64
    rgbxyz_input = torch.randn(batch_size, 6, height, width)
    
    # 前向传播
    output = encoder(rgbxyz_input)
    print(f"Input shape: {rgbxyz_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Advanced Point Cloud Encoder created successfully!")
