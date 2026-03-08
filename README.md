# RLFN-SE 本地开发指南

## 目录结构

```
SR_project/
├── model/
│   ├── __init__.py
│   ├── rlfn.py          # 网络结构（Baseline + RLFN-SE）
│   └── loss.py          # TotalLoss = L1 + λ·FreqLoss
├── data/
│   ├── train/
│   │   ├── HR/          # 训练集高分辨率图像
│   │   └── LR/          # 训练集低分辨率图像（用 prepare_data.py 生成）
│   └── test/
│       ├── Set5/
│       │   ├── HR/
│       │   └── LR/
│       └── Set14/
│           ├── HR/
│           └── LR/
├── checkpoints/         # 训练权重（自动生成）
├── results/             # 测试结果（自动生成）
├── config.py            # 所有参数配置
├── dataset.py           # 数据集类
├── utils.py             # PSNR / SSIM 工具
├── prepare_data.py      # 生成 LR 图像
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── gui.py               # PyQt5 GUI
├── smoke_test.py        # 冒烟测试（第一步跑这个）
└── requirements.txt
```

---

## 第一步：创建 venv 环境并安装依赖

```bash
# 在项目根目录下创建虚拟环境
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 安装 PyTorch（CPU 版，本地调试用）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 安装其他依赖
pip install -r requirements.txt
```

> 以后每次打开终端都要先激活：`.venv\Scripts\activate`
> VS Code 会自动检测到 `.venv`，在右下角选择该解释器即可。

---

## 第二步：验证环境（不需要任何数据）

```bash
python smoke_test.py
```

全部输出 ✓ 即可进入下一步。

---

## 第三步：准备少量测试数据（本地调试用）

从网上下载几张高清图片放到 `data/train/HR/`，然后：

```bash
# 生成训练集 LR
python prepare_data.py --hr data/train/HR --lr data/train/LR --scale 4

# 生成测试集 LR（如果 Set5/Set14 只有 HR 的话）
python prepare_data.py --hr data/test/Set5/HR  --lr data/test/Set5/LR  --scale 4
python prepare_data.py --hr data/test/Set14/HR --lr data/test/Set14/LR --scale 4
```

> ⚠ 正式训练时须用 basicsr 标准 Bicubic 脚本生成 LR（在 Colab 上做）

---

## 第四步：本地调试训练（debug 模式，1 个 batch）

```bash
# 验证训练流程跑通，不报错即成功
python train.py --exp baseline --debug

# 跑 2 个 epoch 看看 loss 下降趋势
python train.py --exp baseline --epochs 2
```

---

## 第五步：本地测试脚本

```bash
# 无权重（随机初始化，仅看流程）
python test.py --exp baseline

# 有权重
python test.py --exp baseline --ckpt checkpoints/baseline_best.pth
```

---

## 第六步：启动 GUI

```bash
python gui.py
```

操作顺序：
1. 点"加载 LR 图片"
2. （可选）点"加载 HR 参考"
3. 选择模型
4. 点"开始超分辨率重建"

---

## 搬到 Colab 训练

1. 打开 `config.py`，将 `ENV = "local"` 改为 `ENV = "colab"`
2. 将整个项目上传到 Google Drive
3. Colab 中挂载 Drive，进入项目目录，按消融实验表依次运行：

```bash
python train.py --exp baseline    # 100 epoch
python train.py --exp silu        # 50 epoch
python train.py --exp eca         # 50 epoch
python train.py --exp silu_eca    # 50 epoch
python train.py --exp freq        # 50 epoch
python train.py --exp final_s1    # 100 epoch（热启动第一阶段）
python train.py --exp final_s2 --resume checkpoints/final_s1_best.pth  # 100 epoch
```

---

## 消融实验对照表

| 实验组     | 模型    | 激活  | 注意力 | 损失      | 热启动 | Epoch |
|------------|---------|-------|--------|-----------|--------|-------|
| baseline   | RLFN    | ReLU  | 无     | L1        | 无     | 100   |
| silu       | RLFN-SE | SiLU  | 无*    | L1        | 无     | 50    |
| eca        | RLFN-SE | SiLU* | ECA    | L1        | 无     | 50    |
| silu_eca   | RLFN-SE | SiLU  | ECA    | L1        | 无     | 50    |
| freq       | RLFN-SE | SiLU  | ECA    | L1+Freq   | 无     | 50    |
| final      | RLFN-SE | SiLU  | ECA    | L1+Freq   | ✓      | 100×2 |

> *单项消融组在论文中说明另一改进项保持固定即可。
