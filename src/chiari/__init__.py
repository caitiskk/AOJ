# src/chiari/__init__.py
"""Chiari I/III 型多模态分类：X-ray 影像 + 临床表格。"""

__version__ = "0.1.0"

# ⚠️ 刻意不放任何子包 import。
# 原因：from chiari.xxx import y 会先执行本文件——这里若 import torch/cv2，
# 只想用 load_config 的地方也得背上全部依赖；且链上任何一个 __init__
# 出错，整个包都 import 不了。