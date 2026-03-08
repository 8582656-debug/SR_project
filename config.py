"""
训练配置文件
本地调试 / Colab 训练均从这里改参数，train.py 不需要动。
"""

import os

# ============================================================
# 路径配置
# ============================================================
# 本地路径（Windows 风格，斜线也可以，Python 兼容）
LOCAL_HR_DIR    = r"data/train/HR"
LOCAL_LR_DIR    = r"data/train/LR"
LOCAL_CKPT_DIR  = r"checkpoints"
LOCAL_RESULT_DIR = r"results"

# Colab 路径（搬到 Colab 时只改这里）
COLAB_HR_DIR    = "/content/drive/MyDrive/SR_project/data/DIV2K_HR"
COLAB_LR_DIR    = "/content/drive/MyDrive/SR_project/data/DIV2K_LR_x4"
COLAB_CKPT_DIR  = "/content/drive/MyDrive/SR_project/checkpoints"
COLAB_RESULT_DIR = "/content/drive/MyDrive/SR_project/results"

# 当前使用哪套路径（本地调试改 "local"，Colab 改 "colab"）
ENV = "local"

if ENV == "local":
    HR_DIR     = LOCAL_HR_DIR
    LR_DIR     = LOCAL_LR_DIR
    CKPT_DIR   = LOCAL_CKPT_DIR
    RESULT_DIR = LOCAL_RESULT_DIR
else:
    HR_DIR     = COLAB_HR_DIR
    LR_DIR     = COLAB_LR_DIR
    CKPT_DIR   = COLAB_CKPT_DIR
    RESULT_DIR = COLAB_RESULT_DIR

# ============================================================
# 模型配置
# ============================================================
NUM_FEAT   = 52
NUM_BLOCK  = 6
UPSCALE    = 4

# ============================================================
# 训练超参数
# ============================================================
PATCH_SIZE   = 192       # HR patch 尺寸（Colab T4 安全值）
BATCH_SIZE   = 8         # Colab T4 安全值
NUM_WORKERS  = 0         # Windows 下必须为 0，Colab 可改 2

LR_INIT      = 2e-4
LR_DECAY_STEP = 200_000  # 每 N 次 iter 学习率减半
BETAS        = (0.9, 0.999)

FREQ_LAMBDA  = 0.1       # 频率损失权重，不稳定可改 0.05 / 0.01

# ============================================================
# 各消融组 Epoch 配置（方案更新版）
# ============================================================
EPOCHS_BY_EXP = {
    "baseline":    100,   # 必须跑满
    "silu":         50,
    "eca":          50,
    "silu_eca":     50,
    "freq":         50,
    "final_s1":    100,   # 两阶段热启动第一阶段
    "final_s2":    100,   # 两阶段热启动第二阶段
}

# ============================================================
# 验证 / 测试配置
# ============================================================
SAVE_EVERY   = 5         # 每 N epoch 保存一次 checkpoint
CROP_BORDER  = 4         # PSNR/SSIM 边界裁剪（×4 SR 标准）
SCALE        = 4

# 测试集路径
TEST_SETS = {
    "Set5":  r"data/test/Set5",
    "Set14": r"data/test/Set14",
}
