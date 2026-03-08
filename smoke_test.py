"""
冒烟测试：不需要真实数据，验证所有模块能跑通
本地搭好环境后第一步就运行此脚本。

用法：
    python smoke_test.py

全部通过后再准备数据、跑训练。
"""

import os
import tempfile
import shutil

import numpy as np
import torch
from PIL import Image


def section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ---------------------------------------------------------------------------
# 1. 模型结构
# ---------------------------------------------------------------------------
section("1. 模型结构")
from model.rlfn import RLFN_Baseline, RLFN_SE

baseline = RLFN_Baseline()
rlfn_se  = RLFN_SE()

x = torch.randn(1, 3, 48, 48)
y_base = baseline(x)
y_se   = rlfn_se(x)

assert y_base.shape == (1, 3, 192, 192), f"Baseline 输出形状错误: {y_base.shape}"
assert y_se.shape   == (1, 3, 192, 192), f"RLFN-SE 输出形状错误: {y_se.shape}"

p_base = sum(p.numel() for p in baseline.parameters())
p_se   = sum(p.numel() for p in rlfn_se.parameters())
print(f"  Baseline 参数量: {p_base/1e3:.1f} K")
print(f"  RLFN-SE  参数量: {p_se/1e3:.1f} K")
print(f"  参数增量: +{(p_se-p_base)/1e3:.1f} K")
print("  ✓ 模型结构 OK")


# ---------------------------------------------------------------------------
# 2. 损失函数
# ---------------------------------------------------------------------------
section("2. 损失函数")
from model.loss import TotalLoss, FrequencyLoss

sr = torch.randn(2, 3, 192, 192, requires_grad=True)
hr = torch.randn(2, 3, 192, 192)

freq_loss  = FrequencyLoss()(sr, hr)
total_loss = TotalLoss(lam=0.1)(sr, hr)
total_loss.backward()

assert sr.grad is not None, "梯度未回传！"
print(f"  FrequencyLoss : {freq_loss.item():.4f}")
print(f"  TotalLoss     : {total_loss.item():.4f}")
print(f"  grad norm     : {sr.grad.norm().item():.4f}")
print("  ✓ 损失函数 + 反向传播 OK")


# ---------------------------------------------------------------------------
# 3. 数据集
# ---------------------------------------------------------------------------
section("3. 数据集")
from dataset import SRDataset, SRTestDataset
from torch.utils.data import DataLoader

tmp = tempfile.mkdtemp()
hr_dir = os.path.join(tmp, "HR"); os.makedirs(hr_dir)
lr_dir = os.path.join(tmp, "LR"); os.makedirs(lr_dir)
test_dir = os.path.join(tmp, "test")
os.makedirs(os.path.join(test_dir, "HR")); os.makedirs(os.path.join(test_dir, "LR"))

for i in range(4):
    Image.fromarray(np.random.randint(0,255,(256,256,3),dtype=np.uint8)).save(
        os.path.join(hr_dir, f"{i:04d}.png"))
    Image.fromarray(np.random.randint(0,255,(64,64,3),dtype=np.uint8)).save(
        os.path.join(lr_dir, f"{i:04d}.png"))
    Image.fromarray(np.random.randint(0,255,(192,192,3),dtype=np.uint8)).save(
        os.path.join(test_dir, "HR", f"{i:04d}.png"))
    Image.fromarray(np.random.randint(0,255,(48,48,3),dtype=np.uint8)).save(
        os.path.join(test_dir, "LR", f"{i:04d}.png"))

train_ds = SRDataset(hr_dir, lr_dir, patch_size=192, scale=4)
loader   = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
lr_t, hr_t = next(iter(loader))
assert lr_t.shape == (2, 3, 48, 48),  f"训练 LR 形状错误: {lr_t.shape}"
assert hr_t.shape == (2, 3, 192, 192), f"训练 HR 形状错误: {hr_t.shape}"
print(f"  训练集: LR {lr_t.shape}  HR {hr_t.shape}")

test_ds  = SRTestDataset(test_dir)
tl, th, tn = test_ds[0]
print(f"  测试集: LR {tl.shape}  HR {th.shape}  name={tn}")
print("  ✓ 数据集 OK")

shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# 4. PSNR / SSIM
# ---------------------------------------------------------------------------
section("4. PSNR / SSIM")
from utils import calc_psnr, calc_ssim

a = torch.rand(2, 3, 192, 192)
b = torch.rand(2, 3, 192, 192)

psnr_diff = calc_psnr(a, b)
ssim_diff = calc_ssim(a, b)
psnr_same = calc_psnr(a, a)

print(f"  PSNR (不同): {psnr_diff:.2f} dB")
print(f"  PSNR (相同): {psnr_same:.2f} dB  （期望接近 100）")
print(f"  SSIM (不同): {ssim_diff:.4f}")
assert psnr_same > 80, "PSNR 自相关应接近 100"
print("  ✓ PSNR / SSIM OK")


# ---------------------------------------------------------------------------
# 5. 端到端前向（模拟一个训练 step）
# ---------------------------------------------------------------------------
section("5. 端到端训练 Step")
model     = RLFN_SE()
criterion = TotalLoss(lam=0.1)
optim     = torch.optim.Adam(model.parameters(), lr=2e-4)

lr_in = torch.rand(2, 3, 48, 48)
hr_gt = torch.rand(2, 3, 192, 192)

model.train()
sr_out = model(lr_in)
loss   = criterion(sr_out, hr_gt)
optim.zero_grad()
loss.backward()
optim.step()

print(f"  输入 LR: {lr_in.shape}  输出 SR: {sr_out.shape}")
print(f"  Loss: {loss.item():.4f}")
print("  ✓ 端到端训练 Step OK")


# ---------------------------------------------------------------------------
# 6. 权重保存 & 加载
# ---------------------------------------------------------------------------
section("6. 权重保存 & 加载")
import tempfile
ckpt_path = os.path.join(tempfile.gettempdir(), "test_model.pth")
torch.save(model.state_dict(), ckpt_path)

model2 = RLFN_SE()
model2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
model2.eval()

# 用保存权重的模型重新推理（sr_out 是 optim.step() 前的结果，不能直接对比）
model.eval()
with torch.no_grad():
    out1 = model(lr_in)
    out2 = model2(lr_in)
diff = (out1 - out2).abs().max().item()
assert diff < 1e-5, f"加载权重后输出不一致！diff={diff}"
os.remove(ckpt_path)
print(f"  最大输出差异: {diff:.2e}  (< 1e-5)")
print("  ✓ 权重保存 & 加载 OK")


# ---------------------------------------------------------------------------
# 全部通过
# ---------------------------------------------------------------------------
print("\n" + "🎉 " * 10)
print("  全部冒烟测试通过！环境搭建完成，可以开始准备数据。")
print("🎉 " * 10)
