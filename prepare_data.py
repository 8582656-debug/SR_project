"""
生成训练/测试 LR 图像
使用 PIL BICUBIC 插值（与 basicsr 标准一致，比 OpenCV 默认更接近 MATLAB）

用法：
    # 生成训练集 LR
    python prepare_data.py --hr data/train/HR --lr data/train/LR --scale 4

    # 生成测试集 LR（若官方测试集中只有 HR，需要自行生成 LR）
    python prepare_data.py --hr data/test/Set5/HR --lr data/test/Set5/LR --scale 4

注意：
  - 如果从 RLFN 官方仓库下载了完整测试集（已含配套 LR），则测试集无需运行此脚本。
  - 训练集 DIV2K 的 LR 必须用此脚本（或 basicsr 官方脚本）生成，
    禁止用 OpenCV cv2.resize 默认插值（INTER_LINEAR），
    否则 PSNR 可能偏低 0.2~0.4 dB，Baseline 无法对齐原文。
"""

import argparse
import os
from PIL import Image


def generate_lr(hr_dir: str, lr_dir: str, scale: int = 4):
    os.makedirs(lr_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp"}

    hr_files = sorted([
        f for f in os.listdir(hr_dir)
        if os.path.splitext(f)[1].lower() in exts
    ])
    if not hr_files:
        print(f"[prepare] ⚠ 未找到图像文件：{hr_dir}")
        return

    print(f"[prepare] HR 目录: {hr_dir}，共 {len(hr_files)} 张")
    print(f"[prepare] LR 目录: {lr_dir}，缩小倍数: ×{scale}")

    for fname in hr_files:
        hr_path = os.path.join(hr_dir, fname)
        lr_path = os.path.join(lr_dir, fname)

        hr = Image.open(hr_path).convert("RGB")
        w, h = hr.size
        lw, lh = w // scale, h // scale

        # PIL BICUBIC — 与 MATLAB imresize 最接近的 Python 实现
        lr = hr.resize((lw, lh), Image.BICUBIC)
        lr.save(lr_path)

    print(f"[prepare] 完成，已生成 {len(hr_files)} 张 LR 图像 → {lr_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr",    type=str, required=True, help="HR 图像目录")
    parser.add_argument("--lr",    type=str, required=True, help="LR 输出目录")
    parser.add_argument("--scale", type=int, default=4,     help="下采样倍数")
    args = parser.parse_args()

    generate_lr(args.hr, args.lr, args.scale)
