import matplotlib.pyplot as plt
import numpy as np

# ====== 全局期刊样式配置 ======
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']  # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['axes.linewidth'] = 1.0        # 边框粗细

# 定义学术配色
COLOR_PALETTE = ['#4A90E2', '#50E3C2', '#F5A623', '#E28491', '#9B51E0'] # 5折线颜色
COLOR_MEAN = '#1F2D3D'       # 均值线颜色 (深灰黑，比纯蓝更显高级)
COLOR_SHADE = '#90A4AE'      # 标准差阴影颜色
# ==============================