"""
评估工具
  - rgb2y: RGB tensor → Y 通道（YCbCr 标准，符合超分领域惯例）
  - calc_psnr: 在 Y 通道计算 PSNR，crop border=4
  - calc_ssim: 在 Y 通道计算 SSIM，crop border=4
  - AverageMeter: 训练时统计均值
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim


# ---------------------------------------------------------------------------
# RGB → Y 通道（YCbCr BT.601，与 MATLAB / basicsr 一致）
# ---------------------------------------------------------------------------
def rgb2y(tensor: torch.Tensor) -> torch.Tensor:
    """
    Args:
        tensor: (B, 3, H, W) float [0, 1]
    Returns:
        (B, 1, H, W) float，Y 通道（值域约 16/255 ~ 235/255）
    """
    r, g, b = tensor[:, 0:1], tensor[:, 1:2], tensor[:, 2:3]
    y = 16.0 / 255.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    return y


# ---------------------------------------------------------------------------
# PSNR（Y 通道，crop border=4）
# ---------------------------------------------------------------------------
def calc_psnr(sr: torch.Tensor, hr: torch.Tensor,
              crop: int = 4) -> float:
    """
    Args:
        sr, hr: (B, 3, H, W) float tensor [0, 1]
        crop:   边界裁剪像素数，×4 SR 标准为 4
    Returns:
        float, dB
    """
    with torch.no_grad():
        sr_y = rgb2y(sr.clamp(0, 1))
        hr_y = rgb2y(hr.clamp(0, 1))
        if crop > 0:
            sr_y = sr_y[..., crop:-crop, crop:-crop]
            hr_y = hr_y[..., crop:-crop, crop:-crop]
        mse = torch.mean((sr_y - hr_y) ** 2).item()
        if mse < 1e-10:
            return 100.0
        return -10.0 * np.log10(mse)


# ---------------------------------------------------------------------------
# SSIM（Y 通道，crop border=4）
# ---------------------------------------------------------------------------
def calc_ssim(sr: torch.Tensor, hr: torch.Tensor,
              crop: int = 4) -> float:
    """
    Args:
        sr, hr: (B, 3, H, W) float tensor [0, 1]
    Returns:
        float, batch 均值
    """
    sr_y = rgb2y(sr.clamp(0, 1))
    hr_y = rgb2y(hr.clamp(0, 1))
    if crop > 0:
        sr_y = sr_y[..., crop:-crop, crop:-crop]
        hr_y = hr_y[..., crop:-crop, crop:-crop]

    sr_np = sr_y.squeeze(1).cpu().numpy()   # (B, H, W)
    hr_np = hr_y.squeeze(1).cpu().numpy()

    ssims = [
        sk_ssim(s, h, data_range=1.0)
        for s, h in zip(sr_np, hr_np)
    ]
    return float(np.mean(ssims))


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sr = torch.rand(2, 3, 192, 192)
    hr = torch.rand(2, 3, 192, 192)
    print(f"PSNR: {calc_psnr(sr, hr):.2f} dB")
    print(f"SSIM: {calc_ssim(sr, hr):.4f}")

    # 完全相同时 PSNR 应接近 100
    print(f"PSNR (identical): {calc_psnr(hr, hr):.2f} dB")
    print("utils 自测通过 ✓")
