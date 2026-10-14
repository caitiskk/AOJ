# src/chiari/viz/__init__.py
"""出版级绘图。import 本包即应用期刊 rcParams 样式。"""
from chiari.viz import style  # 副作用：任何 viz 导入自动应用全局样式
from chiari.viz.style import COLOR_PALETTE, COLOR_MEAN, COLOR_SHADE
from chiari.viz.curves import plot_roc_curves, plot_pr_curves
from chiari.viz.confusion import plot_confusion_matrix_total

__all__ = ["plot_roc_curves", "plot_pr_curves", "plot_confusion_matrix_total",
           "COLOR_PALETTE", "COLOR_MEAN", "COLOR_SHADE"]