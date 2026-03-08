"""
PyQt5 超分辨率演示 GUI
本地 CPU 运行，答辩现场演示用

用法：
    python gui.py

功能：
  - 加载 LR 图片（PNG / JPG）
  - 可选加载对应 HR 参考图（用于计算 PSNR / SSIM）
  - 模型选择：Baseline / RLFN-SE
  - 一键推理，左右对比展示
  - 数据面板：推理耗时、PSNR、SSIM
"""

import sys
import os
import time

import torch
import numpy as np
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QGroupBox, QSizePolicy,
    QMessageBox,
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt

import config as cfg
from model import RLFN_Baseline, RLFN_SE
from utils import calc_psnr, calc_ssim


# ---------------------------------------------------------------------------
# 辅助：numpy RGB (H,W,3) uint8 → QPixmap
# ---------------------------------------------------------------------------
def np_to_pixmap(arr: np.ndarray) -> QPixmap:
    h, w, c = arr.shape
    qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ---------------------------------------------------------------------------
# 辅助：PIL Image → float tensor (1,3,H,W) [0,1]
# ---------------------------------------------------------------------------
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0
    t   = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return t


# ---------------------------------------------------------------------------
# 辅助：float tensor (1,3,H,W) → numpy (H,W,3) uint8
# ---------------------------------------------------------------------------
def tensor_to_np(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------
class SRWindow(QMainWindow):
    MODELS = {
        "Baseline (RLFN)": ("baseline", RLFN_Baseline),
        "RLFN-SE (Ours)":  ("final_s2",  RLFN_SE),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RLFN-SE 图像超分辨率重建系统")
        self.setMinimumSize(1100, 680)

        self._lr_tensor = None
        self._hr_tensor = None
        self._model_cache = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        # ── 控制栏 ──
        ctrl_box = QGroupBox("控制面板")
        ctrl     = QHBoxLayout(ctrl_box)

        self.btn_load_lr = QPushButton("① 加载 LR 图片")
        self.btn_load_hr = QPushButton("② 加载 HR 参考（可选）")
        self.combo       = QComboBox()
        self.combo.addItems(list(self.MODELS.keys()))
        self.btn_run     = QPushButton("▶  开始超分辨率重建")
        self.btn_save    = QPushButton("💾  保存 SR 结果")

        self.btn_run.setFixedHeight(36)
        bold = QFont(); bold.setBold(True)
        self.btn_run.setFont(bold)

        for w in [self.btn_load_lr, self.btn_load_hr,
                  self.combo, self.btn_run, self.btn_save]:
            ctrl.addWidget(w)

        self.btn_load_lr.clicked.connect(self.load_lr)
        self.btn_load_hr.clicked.connect(self.load_hr)
        self.btn_run.clicked.connect(self.run_sr)
        self.btn_save.clicked.connect(self.save_result)
        root.addWidget(ctrl_box)

        # ── 图像展示区 ──
        img_box  = QGroupBox("图像对比")
        img_grid = QGridLayout(img_box)

        lbl_lr_title = QLabel("LR 输入"); lbl_lr_title.setAlignment(Qt.AlignCenter)
        lbl_sr_title = QLabel("SR 输出"); lbl_sr_title.setAlignment(Qt.AlignCenter)
        lbl_hr_title = QLabel("HR 参考"); lbl_hr_title.setAlignment(Qt.AlignCenter)
        for lbl in [lbl_lr_title, lbl_sr_title, lbl_hr_title]:
            lbl.setFont(bold)

        self.label_lr = self._make_img_label()
        self.label_sr = self._make_img_label()
        self.label_hr = self._make_img_label()

        img_grid.addWidget(lbl_lr_title, 0, 0)
        img_grid.addWidget(lbl_sr_title, 0, 1)
        img_grid.addWidget(lbl_hr_title, 0, 2)
        img_grid.addWidget(self.label_lr, 1, 0)
        img_grid.addWidget(self.label_sr, 1, 1)
        img_grid.addWidget(self.label_hr, 1, 2)
        root.addWidget(img_box)

        # ── 数据面板 ──
        info_box = QGroupBox("推理指标")
        info_lay = QHBoxLayout(info_box)

        self.lbl_time  = self._make_metric("推理时间", "--")
        self.lbl_psnr  = self._make_metric("PSNR", "--")
        self.lbl_ssim  = self._make_metric("SSIM", "--")
        self.lbl_size  = self._make_metric("图像尺寸", "--")

        for w in [self.lbl_time, self.lbl_psnr, self.lbl_ssim, self.lbl_size]:
            info_lay.addWidget(w)
        root.addWidget(info_box)

        # 状态栏
        self.statusBar().showMessage("就绪 — 请先加载 LR 图片")

        # 保存 SR 结果用
        self._sr_np = None

    # ------------------------------------------------------------------
    # 辅助：创建图像标签
    # ------------------------------------------------------------------
    def _make_img_label(self) -> QLabel:
        lbl = QLabel("（未加载）")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedSize(320, 320)
        lbl.setStyleSheet("border: 1px solid #aaa; background: #f5f5f5;")
        lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return lbl

    def _make_metric(self, title: str, val: str) -> QGroupBox:
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lbl = QLabel(val)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFont(QFont("Consolas", 14))
        lay.addWidget(lbl)
        box._lbl = lbl
        return box

    # ------------------------------------------------------------------
    # 加载 LR 图片
    # ------------------------------------------------------------------
    def load_lr(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 LR 图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = Image.open(path).convert("RGB")
        self._lr_tensor = pil_to_tensor(img)
        self._show(self.label_lr, np.array(img))
        w, h = img.size
        self.lbl_size._lbl.setText(f"{w}×{h}")
        self.statusBar().showMessage(f"已加载 LR: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # 加载 HR 参考（可选）
    # ------------------------------------------------------------------
    def load_hr(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 HR 参考图（可选）", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not path:
            return
        img = Image.open(path).convert("RGB")
        self._hr_tensor = pil_to_tensor(img)
        self._show(self.label_hr, np.array(img))
        self.statusBar().showMessage(f"已加载 HR: {os.path.basename(path)}")

    # ------------------------------------------------------------------
    # 超分辨率推理
    # ------------------------------------------------------------------
    def run_sr(self):
        if self._lr_tensor is None:
            QMessageBox.warning(self, "提示", "请先加载 LR 图片！")
            return

        model_name = self.combo.currentText()
        exp_name, model_cls = self.MODELS[model_name]

        # 尝试加载权重
        model = self._get_model(exp_name, model_cls)
        self.statusBar().showMessage(f"推理中…（{model_name}）")
        QApplication.processEvents()

        t0 = time.perf_counter()
        with torch.no_grad():
            sr = model(self._lr_tensor).clamp(0, 1)
        elapsed = time.perf_counter() - t0

        self._sr_np = tensor_to_np(sr)
        self._show(self.label_sr, self._sr_np)

        # 更新指标
        self.lbl_time._lbl.setText(f"{elapsed:.3f} s")

        if self._hr_tensor is not None:
            psnr = calc_psnr(sr, self._hr_tensor, crop=cfg.CROP_BORDER)
            ssim = calc_ssim(sr, self._hr_tensor, crop=cfg.CROP_BORDER)
            self.lbl_psnr._lbl.setText(f"{psnr:.2f} dB")
            self.lbl_ssim._lbl.setText(f"{ssim:.4f}")
        else:
            self.lbl_psnr._lbl.setText("N/A")
            self.lbl_ssim._lbl.setText("N/A")

        h, w = self._sr_np.shape[:2]
        self.lbl_size._lbl.setText(f"{w}×{h}")
        self.statusBar().showMessage(f"推理完成  耗时 {elapsed:.3f}s")

    # ------------------------------------------------------------------
    # 保存 SR 结果
    # ------------------------------------------------------------------
    def save_result(self):
        if self._sr_np is None:
            QMessageBox.warning(self, "提示", "请先执行超分辨率重建！")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 SR 结果", "sr_result.png",
            "PNG (*.png);;JPEG (*.jpg)")
        if path:
            Image.fromarray(self._sr_np).save(path)
            self.statusBar().showMessage(f"已保存: {path}")

    # ------------------------------------------------------------------
    # 辅助：展示图像到 QLabel
    # ------------------------------------------------------------------
    def _show(self, label: QLabel, rgb_np: np.ndarray):
        if rgb_np is None:
            return
        pixmap = np_to_pixmap(np.ascontiguousarray(rgb_np))
        label.setPixmap(pixmap.scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ------------------------------------------------------------------
    # 辅助：加载/缓存模型（避免重复加载）
    # ------------------------------------------------------------------
    def _get_model(self, exp_name: str, model_cls):
        if exp_name in self._model_cache:
            return self._model_cache[exp_name]

        model = model_cls(num_feat=cfg.NUM_FEAT,
                          num_block=cfg.NUM_BLOCK,
                          upscale=cfg.UPSCALE)

        # 尝试自动匹配权重
        ckpt_candidates = [
            os.path.join(cfg.CKPT_DIR, f"{exp_name}_best.pth"),
            os.path.join(cfg.CKPT_DIR, f"{exp_name}_latest.pth"),
        ]
        loaded = False
        for ckpt in ckpt_candidates:
            if os.path.isfile(ckpt):
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
                self.statusBar().showMessage(f"加载权重: {os.path.basename(ckpt)}")
                loaded = True
                break

        if not loaded:
            self.statusBar().showMessage(
                f"⚠ 未找到 {exp_name} 权重，使用随机初始化（结果仅供调试）")

        model.eval()
        self._model_cache[exp_name] = model
        return model


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = SRWindow()
    win.show()
    sys.exit(app.exec_())
