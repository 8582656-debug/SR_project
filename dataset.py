"""
数据集类
  - SRDataset: 训练集，随机裁剪 + 数据增强
  - SRTestDataset: 测试集，整图推理（Set5 / Set14）
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# 工具：PIL Image → float tensor [0,1]，CHW
# ---------------------------------------------------------------------------
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


# ---------------------------------------------------------------------------
# 训练集
# ---------------------------------------------------------------------------
class SRDataset(Dataset):
    """
    目录结构要求：
        hr_dir/  → 高分辨率图像（PNG/JPG），文件名与 lr_dir 一一对应
        lr_dir/  → 低分辨率图像（提前用标准 Bicubic 脚本生成）

    Args:
        patch_size: HR patch 尺寸，默认192（Colab T4 安全值）
        scale:      超分辨率倍数，默认4
    """

    def __init__(self, hr_dir: str, lr_dir: str,
                 patch_size: int = 192, scale: int = 4):
        super().__init__()
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        self.hr_paths = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        self.lr_dir    = lr_dir
        self.patch     = patch_size
        self.scale     = scale
        assert len(self.hr_paths) > 0, f"hr_dir 中没有找到图像：{hr_dir}"

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_path = self.hr_paths[idx]
        fname   = os.path.basename(hr_path)
        lr_path = os.path.join(self.lr_dir, fname)

        hr = Image.open(hr_path).convert("RGB")
        lr = Image.open(lr_path).convert("RGB")

        # ---- 随机裁剪（LR空间对齐）----
        lp = self.patch // self.scale          # LR patch size
        lw, lh = lr.size
        assert lw >= lp and lh >= lp, (
            f"图像太小（{lw}×{lh}），无法裁剪 patch={lp}。"
            f"请检查文件：{lr_path}"
        )
        x = random.randint(0, lw - lp)
        y = random.randint(0, lh - lp)

        lr = lr.crop((x, y, x + lp, y + lp))
        hr = hr.crop((x * self.scale, y * self.scale,
                      (x + lp) * self.scale, (y + lp) * self.scale))

        # ---- 数据增强：随机水平翻转 + 随机90度旋转 ----
        if random.random() > 0.5:
            lr = TF.hflip(lr)
            hr = TF.hflip(hr)
        k = random.randint(0, 3)
        lr = TF.rotate(lr, 90 * k)
        hr = TF.rotate(hr, 90 * k)

        return pil_to_tensor(lr), pil_to_tensor(hr)


# ---------------------------------------------------------------------------
# 测试集（整图，不裁剪）
# ---------------------------------------------------------------------------
class SRTestDataset(Dataset):
    """
    目录结构：
        root/
          HR/  → 高分辨率参考图
          LR/  → 低分辨率输入图
    文件名必须一一对应（从 RLFN 官方仓库下载的版本）。
    """

    def __init__(self, root: str):
        super().__init__()
        hr_dir = os.path.join(root, "HR")
        lr_dir = os.path.join(root, "LR")
        exts   = {".png", ".jpg", ".jpeg", ".bmp"}

        self.hr_paths = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        self.lr_paths = sorted([
            os.path.join(lr_dir, f)
            for f in os.listdir(lr_dir)
            if os.path.splitext(f)[1].lower() in exts
        ])
        assert len(self.hr_paths) == len(self.lr_paths), (
            f"HR({len(self.hr_paths)}) 与 LR({len(self.lr_paths)}) 数量不匹配！"
        )

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        lr = Image.open(self.lr_paths[idx]).convert("RGB")
        name = os.path.splitext(os.path.basename(self.hr_paths[idx]))[0]
        return pil_to_tensor(lr), pil_to_tensor(hr), name


# ---------------------------------------------------------------------------
# 快速自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile, shutil
    from PIL import Image as PILImage

    # 创建临时假数据
    tmp = tempfile.mkdtemp()
    hr_dir = os.path.join(tmp, "HR"); os.makedirs(hr_dir)
    lr_dir = os.path.join(tmp, "LR"); os.makedirs(lr_dir)

    for i in range(3):
        hr_img = PILImage.fromarray(
            np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8))
        lr_img = PILImage.fromarray(
            np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8))
        hr_img.save(os.path.join(hr_dir, f"{i:04d}.png"))
        lr_img.save(os.path.join(lr_dir, f"{i:04d}.png"))

    ds = SRDataset(hr_dir, lr_dir, patch_size=192)
    lr_t, hr_t = ds[0]
    print(f"SRDataset  LR: {lr_t.shape}  HR: {hr_t.shape}")

    shutil.rmtree(tmp)
    print("Dataset 自测通过 ✓")
