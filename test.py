"""
测试脚本：在 Set5 / Set14 上评估模型，输出 PSNR + SSIM 表格
用法：
    python test.py --exp baseline --ckpt checkpoints/baseline_best.pth
    python test.py --exp final    --ckpt checkpoints/final_s2_best.pth
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import SRTestDataset
from model import RLFN_Baseline, RLFN_SE
from utils import calc_psnr, calc_ssim, AverageMeter


# ---------------------------------------------------------------------------
# 根据实验名判断用哪个模型结构
# ---------------------------------------------------------------------------
def load_model(exp: str, ckpt_path: str, device: torch.device):
    if exp == "baseline":
        model = RLFN_Baseline(num_feat=cfg.NUM_FEAT,
                               num_block=cfg.NUM_BLOCK,
                               upscale=cfg.UPSCALE)
    else:
        model = RLFN_SE(num_feat=cfg.NUM_FEAT,
                        num_block=cfg.NUM_BLOCK,
                        upscale=cfg.UPSCALE)

    if ckpt_path and os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[test] 加载权重: {ckpt_path}")
    else:
        print(f"[test] ⚠ 未找到权重文件，使用随机初始化（仅用于调试）")

    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# 在单个测试集上评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, test_root: str, device: torch.device, set_name: str):
    dataset = SRTestDataset(test_root)
    loader  = DataLoader(dataset, batch_size=1,
                         shuffle=False, num_workers=0)

    psnr_meter = AverageMeter("PSNR")
    ssim_meter = AverageMeter("SSIM")

    print(f"\n  {set_name} ({len(dataset)} 张)")
    print(f"  {'Image':<20} {'PSNR':>8} {'SSIM':>8}")
    print(f"  {'-'*40}")

    for lr, hr, (name,) in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr).clamp(0, 1)

        psnr = calc_psnr(sr, hr, crop=cfg.CROP_BORDER)
        ssim = calc_ssim(sr, hr, crop=cfg.CROP_BORDER)

        psnr_meter.update(psnr)
        ssim_meter.update(ssim)
        print(f"  {name:<20} {psnr:>8.2f} {ssim:>8.4f}")

    print(f"  {'Average':<20} {psnr_meter.avg:>8.2f} {ssim_meter.avg:>8.4f}")
    return psnr_meter.avg, ssim_meter.avg


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",  type=str, default="baseline",
                        help="实验组名（用于确定模型结构）")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="权重路径（可选）")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] exp={args.exp}  device={device}")

    model = load_model(args.exp, args.ckpt, device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[test] 参数量: {total_params / 1e3:.1f} K")

    # 在所有测试集上评估
    results = {}
    for set_name, test_root in cfg.TEST_SETS.items():
        if not os.path.isdir(test_root):
            print(f"[test] ⚠ 测试集路径不存在，跳过：{test_root}")
            continue
        psnr, ssim = evaluate(model, test_root, device, set_name)
        results[set_name] = (psnr, ssim)

    # 汇总表格
    if results:
        print("\n" + "=" * 50)
        print(f"  {'Dataset':<12} {'PSNR (dB)':>10} {'SSIM':>10}")
        print("  " + "-" * 36)
        for name, (p, s) in results.items():
            print(f"  {name:<12} {p:>10.2f} {s:>10.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
