# -*- coding: utf-8 -*-
"""
全局配置文件：管理模型预测文件路径、颜色主题、绘图样式及默认超参数
"""
import matplotlib.pyplot as plt

# 默认模型名称及 5-Fold 预测结果路径映射
MODEL_FILES = {
    'Ours (MedicalGMU)': './experiments/gmu_asymmetric/predictions/5fold_results.pkl',
    'Concat':            './experiments/concat/predictions/5fold_results.pkl',
    'Gmu-standard':      './experiments/gmu_standard/predictions/5fold_results.pkl',
    'Dense-Net':          	'./experiments/densenet/predictions/5fold_results.pkl',
    'Image-only':        './experiments/image_only/predictions/5fold_results.pkl',
    'Text-only':        './experiments/text_only/predictions/5fold_results.pkl',
}

# 基准模型名称（用于配对显著性检验与曲线高亮）
TARGET_MODEL = 'Ours (MedicalGMU)'

# 多模型对比绘图主色板（粉红、蓝、绿、橙）
MODEL_COLORS = ['#E28491', '#4A90E2', '#2ECC71', '#F39C12']

# 混淆矩阵类别标签名称
CLASS_NAMES = ['Non-III', 'III Type']

# 默认评估超参数
DEFAULT_FIXED_THRESHOLD = 0.25
DEFAULT_BOOTSTRAP_ITERS = 2000
DEFAULT_RANDOM_SEED = 42


def set_journal_plot_style():
    """配置符合医学/计算机期刊规范的 Matplotlib 全局样式"""
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.linewidth'] = 1.0
