# src/chiari/interpret/__init__.py
"""可解释性：GradCAM 算法 + 渲染 + 批量生成。"""
from chiari.interpret.gradcam import ManualGradCAM
from chiari.interpret.overlays import (
    concatenate_images_with_gap,
    create_colorful_overlay,
    create_multi_threshold_colorful,
)
from chiari.interpret.generate import generate_cams_for_fold

__all__ = ["ManualGradCAM", "concatenate_images_with_gap",
           "create_colorful_overlay", "create_multi_threshold_colorful",
           "generate_cams_for_fold"]