# src/chiari/interpret/generate.py
"""按折批量生成 CAM 可视化结果。

"""
from chiari.data.transforms import IMAGENET_MEAN, IMAGENET_STD   # ← 与 transforms 共享常量
from chiari.interpret.gradcam import ManualGradCAM
from chiari.interpret.overlays import create_multi_threshold_colorful
from chiari.utils.io import save_cam_result

import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


logger = logging.getLogger(__name__)


def generate_cams_for_fold(model, val_loader, fold_idx, save_root, device,
                           thresholds=(0.05, 0.4, 0.6), alpha=0.5, gap=5,
                           colormap_style='turbo'):
    """对一折验证集逐样本生成 CAM 拼接图。
    """
    save_root = Path(save_root)
    cam_generator = ManualGradCAM(model, target_layer='resnet.layer4')   # 显式，不吃默认值

    results, joint_counter = [], defaultdict(int)
    mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

    for imgs, labels, cat_batch, cont_batch, _, joint_ids in val_loader:
        img_tensor = imgs.to(device)
        gt_label = labels[0].item()
        joint_id = joint_ids[0]
        cat_features = {k: v[0:1].to(device) for k, v in cat_batch.items()}
        cont_features = cont_batch[0:1].to(device)

        # no_grad 只包预测段——CAM 生成需要梯度，包进去会得到全零热力图且不报错
        with torch.no_grad():
            logits = model(img_tensor, cat_features, cont_features)
            probs = torch.softmax(logits, dim=1)[0]
            pred_label = int(torch.argmax(probs).item())
            pred_prob = probs[pred_label].item()

        try:
            cam, _, _ = cam_generator.generate(
                img_tensor, cat_features, cont_features, target_class=pred_label)

            # 反归一化原图
            # img_np = imgs[0].cpu().numpy().transpose(1, 2, 0)
            img_np = imgs[0].detach().cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np * std + mean, 0, 1)
            h, w = img_np.shape[:2]
            cam_resized = cv2.resize(cam, (w, h))

            combined_img = create_multi_threshold_colorful(
                img_np, cam_resized,
                thresholds=list(thresholds), alpha=alpha,
                gap=gap, colormap_style=colormap_style)

            joint_counter[joint_id] += 1
            img_idx = joint_counter[joint_id]

            save_path = save_cam_result(
                combined_img=combined_img, gt_label=gt_label,
                pred_label=pred_label, prob=pred_prob,
                sample_name=f'{joint_id}', fold_num=fold_idx,
                save_root=save_root, img_idx=img_idx)

            results.append({'joint_id': joint_id, 'gt': gt_label,
                            'pred': pred_label, 'prob': pred_prob,
                            'path': str(save_path)})
        except Exception:
            logger.exception("Failed for %s", joint_id)

    cam_generator.remove_hooks()
    logger.info("Fold %d complete: %d CAMs", fold_idx, len(results))
    return results


