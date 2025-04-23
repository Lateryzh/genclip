import torch
import torch.nn as nn

from .codec import Encoder, Decoder
from .quantizer import VectorQuantizer
from .discriminator import Discriminator

import torch.nn.functional as F

class VQGAN(nn.Module):

    def __init__(self, codebook_size, n_embed):
        super().__init__()

        self.encoder = Encoder(z_channels=n_embed)
        self.decoder = Decoder(z_channels=n_embed)
        self.quantizer = VectorQuantizer(codebook_size, n_embed)
        self.discriminator = Discriminator()

    def encode(self, x):
        z = self.encoder(x)  # map to latent space, [N, C, H, W]
        z_q, loss_q, indices = self.quantizer(z)  # quantize
        return z_q, loss_q, indices

    def decode(self, z):
        # reconstruct images
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x, stage: int):

        if stage == 0:
            # Stage 0: training E + G + Q
            z, loss_q, _ = self.encode(x)
            x_recon = self.decode(z)
            logits_fake = self.discriminator(x_recon)

            return x_recon, loss_q, logits_fake

        elif stage == 1:
            # Stage 1: training D
            with torch.no_grad():
                z, loss_q, _ = self.encode(x)
                x_recon = self.decode(z)

            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(x_recon.detach())
            return logits_real, logits_fake

        else:
            raise ValueError(f"Invalid stage: {stage}")

'''
为什么不能直接使用现有判别器做有监督？

信号类型不匹配：
原始判别器设计用于判断图像全局真伪（对抗训练），而关键点监督需要精确的局部坐标回归。二者的监督粒度不同：
特征层级差异：
判别器的深层特征偏向抽象语义，而关键点检测需要细粒度空间信息。二者特征金字塔不兼容
多任务冲突风险：
同一判别器同时承担真伪判断和关键点回归任务，易导致梯度冲突。
'''


#新增代码，新增内容为：
# 在模型定义中增加跨图像交互模块
class CrossImageAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        # 确保初始参数在正确设备
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    def forward(self, z1, z2):
        # 展平空间维度 [B, C, H, W] => [B, H*W, C]
        z1 = z1.flatten(2).permute(0, 2, 1)
        z2 = z2.flatten(2).permute(0, 2, 1)
        """成对潜在编码交互"""
        B, L, C = z1.shape
        q = self.q_proj(z1)  # [B, L, C]
        k, v = self.kv_proj(z2).chunk(2, dim=-1)  # [B, L, C] each

        attn = (q @ k.transpose(1, 2)) * (C  **  -0.5)
        attn = attn.softmax(dim=-1)
        return (attn @ v).reshape(B, L, C)

class FeatureFusion(nn.Module):
    """
    融合图像特征 z1_q 与关键点编码特征（FiLM 调制）
    假设 z1_q: [B, C, H, W]，kp_feat: [B, C]
    """
    def __init__(self, feat_channels):
        super().__init__()
        self.gamma = nn.Linear(feat_channels, feat_channels)
        self.beta = nn.Linear(feat_channels, feat_channels)

    def forward(self, z1_q, kp_feat):
        B, C, H, W = z1_q.shape
        gamma = self.gamma(kp_feat).view(B, C, 1, 1)
        beta = self.beta(kp_feat).view(B, C, 1, 1)
        return z1_q * (1 + gamma) + beta        # FiLM: scale + shift modulation

class VQGAN3(nn.Module):

    def __init__(self, codebook_size, n_embed,keypoint_dim):
        super().__init__()

        self.encoder = Encoder(z_channels=n_embed)
        self.decoder = Decoder(z_channels=n_embed)
        self.quantizer = VectorQuantizer(codebook_size, n_embed)
        self.discriminator = Discriminator()

        #新增代码，新增内容为：
        self.keypoint_dim = keypoint_dim
        self.kp_encoder = nn.Sequential(
            nn.Flatten(),                       # [B, N, 2] → [B, N*2]
            nn.Linear(keypoint_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # 新增关键点回归头
        # 建议添加正则化层防止过拟合
        self.kp_decoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(n_embed, 512),
            nn.BatchNorm1d(512),  # 新增BatchNorm
            nn.ReLU(),
            nn.Dropout(0.5),  # 新增Dropout
            nn.Linear(512, keypoint_dim),
            nn.Linear(keypoint_dim, 256)
        )

    def encode(self, x_pair):
        def encode(self, x):
            z = self.encoder(x)  # map to latent space, [N, C, H, W]
            z_q, loss_q, indices = self.quantizer(z)  # quantize
            return z_q, loss_q, indices

        # 我原本想这样改的，但是发现forward里面用到的是单路的encode，其实这里不用改
        # x_pair = x_pair.to(self.encoder.conv_in.weight.device)  # 对齐编码器设备
        # # 将输入拆分为两个图像 [B*2, C, H, W] → 两个[B, C, H, W]
        # x1, x2 = x_pair.chunk(2, dim=0)  # 按batch维度拆分
        # z1 = self.encoder(x1)  # map to latent space, [N, C, H, W]
        # z2 = self.encoder(x2)  # map to latent space, [N, C, H, W]
        # z_q1, loss_q1, indices1 = self.quantizer(z1)  # quantize
        # z_q2, loss_q2, indices2 = self.quantizer(z2)  # quantize
        # return z_q1, loss_q1, indices1,z_q2, loss_q2, indices2



    def decode(self, z):
        # reconstruct images
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x_pair, keypoints=None,):
        # 确保输入在CUDA
        x_pair = x_pair.to(self.encoder.conv_in.weight.device)  # 对齐编码器设备
        # 将输入拆分为两个图像 [B*2, C, H, W] → 两个[B, C, H, W]
        x1, x2 = x_pair.chunk(2, dim=0)  # 按batch维度拆分
        if keypoints is not None:
            keypoints = keypoints.to(x_pair.device)



        # 双路编码
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        kp_feat = self.kp_encoder(keypoints)

        z1_q, loss_q1, _ = self.quantizer(z1)
        fuse = FeatureFusion(256)
        z1_cond = fuse(z1_q, kp_feat)  # shape 与 decoder 输入一致

        x1_recon = self.decoder(z1_cond)

        # 只用x2作为x1的标签
        pair_loss = F.l1_loss(x1_recon, x2)

        # 原有单帧损失保持
        self_loss = 0.5 * (
                F.mse_loss(x1_recon, x1) +
                F.mse_loss(x2_recon, x2)
        )

        # # 关键点预测分支
        # kp_pred = self.kp_decoder(kp_feature)  # 融合特征

        # 计算关键点损失
        kp_loss = F.l1_loss(kp_pred, keypoints) if keypoints is not None else 0

        return {
            'recon': (x1_recon, x2_recon),
            'kp_pred': kp_pred,
            'total_loss': self_loss + 0.3 * pair_loss + kp_loss * args.kp_loss_weight,
            'quant_loss': loss_q1 + loss_q2
        }
class VQGAN4(nn.Module):

    def __init__(self, codebook_size, n_embed):
        super().__init__()

        self.encoder = Encoder(z_channels=n_embed)
        self.decoder = Decoder(z_channels=n_embed)
        self.quantizer = VectorQuantizer(codebook_size, n_embed)
        self.discriminator = Discriminator()



    def encode(self, x):
        z = self.encoder(x)  # map to latent space, [N, C, H, W]
        z_q, loss_q, indices = self.quantizer(z)  # quantize
        return z_q, loss_q, indices

    def decode(self, z):
        # reconstruct images
        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        # 保持原始处理流程
        z = self.encoder(x)
        z_q, indices, quant_loss = self.quantize(z)
        x_recon = self.decoder(z_q)
        return x_recon, indices, quant_loss