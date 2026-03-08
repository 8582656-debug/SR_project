"""
RLFN-SE: Residual Local Feature Network with SiLU + ECA improvements
改进点：
  1. RLFB中 ReLU → SiLU
  2. ESA输出后串联ECA轻量级通道注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ECA: Efficient Channel Attention（轻量级，仅1个1D卷积，无FC层）
# ---------------------------------------------------------------------------
class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, k_size, padding=k_size // 2, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                          # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)           # (B, 1, C)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ---------------------------------------------------------------------------
# ESA: Enhanced Spatial Attention（原版RLFN保留，不改动）
# ---------------------------------------------------------------------------
class ESA(nn.Module):
    def __init__(self, num_feat=52, conv=nn.Conv2d):
        super().__init__()
        f = num_feat // 4
        self.conv1    = conv(num_feat, f, 1)
        self.conv_f   = conv(f, f, 1)
        self.conv_max = conv(f, f, 3, padding=1)
        self.conv2    = conv(f, f, 3, stride=2, padding=0)
        self.conv3    = conv(f, f, 3, padding=1)
        self.conv3_   = conv(f, f, 3, padding=1)
        self.conv4    = conv(f, num_feat, 1)
        self.sigmoid  = nn.Sigmoid()
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_   = self.conv1(x)
        c1    = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_rng = self.relu(self.conv_max(v_max))
        c3    = self.relu(self.conv3(v_rng))
        c3    = self.conv3_(c3)
        c3    = F.interpolate(c3, (x.size(2), x.size(3)),
                              mode='bilinear', align_corners=False)
        cf    = self.conv_f(c1_)
        c4    = self.conv4(c3 + cf)
        m     = self.sigmoid(c4)
        return x * m


# ---------------------------------------------------------------------------
# RLFB: Residual Local Feature Block（改进版）
#   ★ 激活函数：ReLU → SiLU
#   ★ 注意力：ESA 后串联 ECA
# ---------------------------------------------------------------------------
class RLFB(nn.Module):
    def __init__(self, num_feat=52):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.act   = nn.SiLU(inplace=True)   # ★ 创新点一
        self.esa   = ESA(num_feat)
        self.eca   = ECA(num_feat)            # ★ 创新点二

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.conv3(out)
        out = self.eca(self.esa(out))         # ESA → ECA
        return out + x                        # 残差连接


# ---------------------------------------------------------------------------
# RLFN_Baseline: 原版 RLFN，用于对照组（ReLU，无ECA）
# ---------------------------------------------------------------------------
class RLFB_Baseline(nn.Module):
    def __init__(self, num_feat=52):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.act   = nn.ReLU(inplace=True)
        self.esa   = ESA(num_feat)

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        out = self.conv3(out)
        out = self.esa(out)
        return out + x


# ---------------------------------------------------------------------------
# 通用主干（Baseline 和 RLFN-SE 共用骨架）
# ---------------------------------------------------------------------------
class _RLFNBase(nn.Module):
    def __init__(self, block_cls, num_in_ch=3, num_out_ch=3,
                 num_feat=52, num_block=6, upscale=4):
        super().__init__()
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, padding=1)
        self.blocks   = nn.Sequential(*[block_cls(num_feat) for _ in range(num_block)])
        self.lr_conv  = nn.Conv2d(num_feat, num_feat, 3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_feat, num_out_ch * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale),
        )

    def forward(self, x):
        fea = self.fea_conv(x)
        out = self.blocks(fea)
        out = self.lr_conv(out) + fea   # 全局残差
        return self.upsample(out)


def RLFN_Baseline(**kwargs):
    return _RLFNBase(RLFB_Baseline, **kwargs)


def RLFN_SE(**kwargs):
    """改进版：SiLU + ECA"""
    return _RLFNBase(RLFB, **kwargs)


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = RLFN_SE()
    x = torch.randn(1, 3, 48, 48)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")   # 期望 (1, 3, 192, 192)

    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total / 1e3:.1f} K")
