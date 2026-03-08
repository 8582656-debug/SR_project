"""
损失函数
  - L1Loss（基础）
  - FrequencyLoss（频率域损失，创新点三）
  - TotalLoss = L1 + λ·FreqLoss
"""

import torch
import torch.nn as nn


class FrequencyLoss(nn.Module):
    """
    对SR和HR分别做2D FFT，在幅度谱上计算L1差异。
    直接约束高频分量（边缘、纹理），缓解L1的过平滑问题。

    ★ 注意：FFT输出为复数张量，必须先 torch.abs() 取幅值再计算L1，
      否则会报错或产生梯度异常。
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_fft = torch.fft.fft2(sr)
        hr_fft = torch.fft.fft2(hr)
        return self.l1(torch.abs(sr_fft), torch.abs(hr_fft))   # ★ 取幅值


class TotalLoss(nn.Module):
    """
    L_total = L1 + λ · L_freq

    Args:
        lam: 频率损失权重，默认0.1；不稳定时可尝试0.05或0.01
    """

    def __init__(self, lam: float = 0.1):
        super().__init__()
        self.l1   = nn.L1Loss()
        self.freq = FrequencyLoss()
        self.lam  = lam

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        l1_loss   = self.l1(sr, hr)
        freq_loss = self.freq(sr, hr)
        return l1_loss + self.lam * freq_loss


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sr = torch.randn(2, 3, 192, 192)
    hr = torch.randn(2, 3, 192, 192)

    criterion = TotalLoss(lam=0.1)
    loss = criterion(sr, hr)
    loss.backward()
    print(f"TotalLoss: {loss.item():.4f}  (backward OK)")
