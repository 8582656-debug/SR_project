"""
训练主脚本
用法（本地调试）：
    python train.py --exp baseline --epochs 2 --debug

用法（正式训练，以 silu 组为例）：
    python train.py --exp silu

用法（两阶段热启动第二阶段，加载第一阶段权重）：
    python train.py --exp final_s2 --resume checkpoints/final_s1_best.pth
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

import config as cfg
from dataset import SRDataset
from model import RLFN_Baseline, RLFN_SE, TotalLoss
from utils import calc_psnr, AverageMeter


# ---------------------------------------------------------------------------
# 根据实验名选择模型 + 损失函数
# ---------------------------------------------------------------------------
def build_model_and_loss(exp: str):
    use_silu = exp in {"silu", "silu_eca", "freq", "final_s1", "final_s2"}
    use_eca  = exp in {"eca", "silu_eca", "freq", "final_s1", "final_s2"}
    use_freq = exp in {"freq", "final_s1", "final_s2"}

    if use_silu or use_eca:
        # 改进版（RLFN_SE 同时含 SiLU + ECA）
        # 注：若只测单项改进，可在 rlfn.py 中单独构造变体；
        #     此处为简化起见，silu/eca 单项组也使用 RLFN_SE，
        #     在论文中说明该组固定另一项即可。
        model = RLFN_SE(num_feat=cfg.NUM_FEAT,
                        num_block=cfg.NUM_BLOCK,
                        upscale=cfg.UPSCALE)
    else:
        model = RLFN_Baseline(num_feat=cfg.NUM_FEAT,
                              num_block=cfg.NUM_BLOCK,
                              upscale=cfg.UPSCALE)

    lam      = cfg.FREQ_LAMBDA if use_freq else 0.0
    criterion = TotalLoss(lam=lam)
    return model, criterion


# ---------------------------------------------------------------------------
# 验证（在少量训练图上跑，本地快速检查；Colab 可接测试集）
# ---------------------------------------------------------------------------
@torch.no_grad()
def quick_validate(model, loader, device, max_batches=5):
    model.eval()
    meter = AverageMeter("PSNR")
    for i, (lr, hr) in enumerate(loader):
        if i >= max_batches:
            break
        sr = model(lr.to(device)).clamp(0, 1)
        meter.update(calc_psnr(sr, hr.to(device)), n=lr.size(0))
    model.train()
    return meter.avg


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------
def train(exp: str, epochs: int, resume: str | None, debug: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] exp={exp}  epochs={epochs}  device={device}")

    # 创建输出目录
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)

    # 数据集
    dataset = SRDataset(cfg.HR_DIR, cfg.LR_DIR,
                        patch_size=cfg.PATCH_SIZE, scale=cfg.SCALE)
    loader  = DataLoader(dataset,
                         batch_size=cfg.BATCH_SIZE,
                         shuffle=True,
                         num_workers=cfg.NUM_WORKERS,
                         pin_memory=(device.type == "cuda"))
    print(f"[train] 训练集: {len(dataset)} 张图，每 epoch {len(loader)} 个 batch")

    # 模型 + 损失
    model, criterion = build_model_and_loss(exp)
    model = model.to(device)
    criterion = criterion.to(device)

    # 优化器 + 学习率调度
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.LR_INIT, betas=cfg.BETAS)
    # StepLR 按 iteration 步长减半
    iter_per_epoch = len(loader)
    step_epochs    = max(1, cfg.LR_DECAY_STEP // iter_per_epoch)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_epochs, gamma=0.5)

    # 断点续训
    start_epoch = 1
    best_psnr   = 0.0
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f"[train] 加载权重: {resume}")

    # 训练循环
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_meter = AverageMeter("Loss")
        t0 = time.time()

        for lr_img, hr_img in loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            sr   = model(lr_img)
            loss = criterion(sr, hr_img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=lr_img.size(0))

            if debug:   # debug 模式只跑 1 个 batch
                break

        scheduler.step()
        elapsed = time.time() - t0

        # 每 SAVE_EVERY 个 epoch 验证并保存
        if epoch % cfg.SAVE_EVERY == 0 or epoch == epochs or debug:
            psnr = quick_validate(model, loader, device)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch:>4}/{epochs}]  "
                  f"Loss={loss_meter.avg:.4f}  "
                  f"PSNR={psnr:.2f}dB  "
                  f"LR={lr_now:.2e}  "
                  f"Time={elapsed:.1f}s")

            # 保存最新权重
            latest_path = os.path.join(cfg.CKPT_DIR, f"{exp}_latest.pth")
            torch.save(model.state_dict(), latest_path)

            # 保存最优权重
            if psnr > best_psnr:
                best_psnr = psnr
                best_path = os.path.join(cfg.CKPT_DIR, f"{exp}_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  ★ 新最优 PSNR={best_psnr:.2f}dB → {best_path}")

            # 每 5 epoch 带 epoch 编号存一份（防 Colab 断连）
            numbered = os.path.join(cfg.CKPT_DIR,
                                    f"{exp}_epoch{epoch:03d}_psnr{psnr:.2f}.pth")
            torch.save(model.state_dict(), numbered)

        if debug:
            print("[debug] 单 batch 验证通过 ✓")
            break

    print(f"[train] 完成，最优 PSNR={best_psnr:.2f} dB")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="baseline",
                        choices=list(cfg.EPOCHS_BY_EXP.keys()),
                        help="消融实验组名")
    parser.add_argument("--epochs", type=int, default=None,
                        help="覆盖 config 中的 epoch 数（调试用）")
    parser.add_argument("--resume", type=str, default=None,
                        help="加载已有权重路径（两阶段热启动用）")
    parser.add_argument("--debug", action="store_true",
                        help="仅跑 1 个 batch，验证代码流程")
    args = parser.parse_args()

    epochs = args.epochs or cfg.EPOCHS_BY_EXP[args.exp]
    train(args.exp, epochs, args.resume, args.debug)
