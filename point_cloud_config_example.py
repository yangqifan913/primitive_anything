# -*- coding: utf-8 -*-
"""
点云编码器配置示例
只保留完整版Michelangelo架构配置
"""

# 完整版Michelangelo点云编码器配置
MICHELANGELO_POINT_CLOUD_ENCODER_CONFIG = {
    # 基础参数
    'num_latents': 256,        # 潜在向量数量
    'embed_dim': 64,           # VAE嵌入维度
    'point_feats': 3,          # RGB特征维度
    'num_freqs': 8,            # 傅里叶频率数量
    'heads': 12,               # 注意力头数
    'width': 768,              # 模型宽度
    'num_encoder_layers': 8,   # 编码器层数
    'num_decoder_layers': 16,  # 解码器层数
    'pretrained': False        # 从头训练
}

# 模型配置
MODEL_CONFIG = {
    # 3D基本体参数
    'num_discrete_x': 128,
    'num_discrete_y': 128,
    'num_discrete_z': 128,
    'num_discrete_w': 64,
    'num_discrete_h': 64,
    'num_discrete_l': 64,
    
    # 连续范围
    'continuous_range_x': [0.5, 2.5],
    'continuous_range_y': [-2, 2],
    'continuous_range_z': [-1.5, 1.5],
    'continuous_range_w': [0.3, 0.7],
    'continuous_range_h': [0.3, 0.7],
    'continuous_range_l': [0.3, 0.7],
    
    # 嵌入维度
    'dim_x_embed': 64,
    'dim_y_embed': 64,
    'dim_z_embed': 64,
    'dim_w_embed': 32,
    'dim_h_embed': 32,
    'dim_l_embed': 32,
    
    # 模型参数
    'dim': 512,
    'max_primitive_len': 10,
    'attn_depth': 6,
    'attn_dim_head': 64,
    'attn_heads': 8,
    'attn_dropout': 0.0,
    'ff_dropout': 0.0,
    
    # 点云编码器配置
    'point_cloud_encoder_dim': 512,
    'use_point_cloud_encoder': True,
    'point_cloud_encoder_config': MICHELANGELO_POINT_CLOUD_ENCODER_CONFIG,
    
    # 其他参数
    'shape_cond_with_cat': False,
    'condition_on_point_cloud': True,
    'gateloop_depth': 2,
    'gateloop_use_heinsen': False,
    'pad_id': -1,
}

# 训练配置建议
TRAINING_RECOMMENDATIONS = {
    'learning_rate': 1e-4,           # 较低的学习率，因为从头训练
    'batch_size': 8,                 # 较小的batch size，因为点云编码器计算量大
    'warmup_epochs': 5,              # 预热轮数
    'weight_decay': 1e-4,            # 权重衰减
    'gradient_clip': 1.0,            # 梯度裁剪
    'scheduler': 'cosine',           # 余弦学习率调度
}

print("点云编码器配置已准备完成！")
print("\n=== 完整版Michelangelo编码器特点 ===")
print("1. 完整的Michelangelo ShapeAsLatentPerceiver架构")
print("2. 包含VAE编码器和解码器")
print("3. 支持几何查询功能")
print("4. 更强大的特征表示能力")
print("5. 完全从头训练，无需预训练模型")
print("6. 使用傅里叶位置编码处理XYZ坐标")
print("7. 使用交叉注意力机制处理RGB特征")
print("8. 包含Transformer自注意力编码")
print("\n使用方法：")
print("- 设置 use_point_cloud_encoder=True")
print("- 使用 MICHELANGELO_POINT_CLOUD_ENCODER_CONFIG 配置")
