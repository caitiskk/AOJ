# scripts/interpret.py
"""用法: python scripts/interpret.py --config configs/default.yaml [--folds 1 3 5]"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from chiari.config import load_config
from chiari.data.dataset import ChiariXrayDataset, collate_fn
from chiari.models.multimodal import MedicalGMUNet
from chiari.interpret.generate import generate_cams_for_fold



import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chiari.config import load_config
from chiari.data.dataset import ChiariXrayDataset, collate_fn
from chiari.models.multimodal import MedicalGMUNet
from chiari.interpret.generate import generate_cams_for_fold
from chiari.utils.checkpoint import ckpt_path, load_fold_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--folds", nargs='+', type=int, default=None, help="默认全部折")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_folds = cfg['training']['n_splits']
    folds = args.folds or range(1, n_folds + 1)

    save_root = Path(cfg['paths']['run_dir']) / 'cams'
    for fold in folds:
        ckpt_path = Path(cfg['paths']['save_dir']) / f'{fold}_medicalgmu_best.pt'

        val_ds = ChiariXrayDataset.from_checkpoint(ckpt_path)   # ← 复用前几轮写的类方法
        val_loader = DataLoader(val_ds, batch_size=1,           # ← 契约：必须为 1
                                shuffle=False, collate_fn=collate_fn, num_workers=2)
        model,_ = load_fold_model(ckpt_path, cfg['device'])

        interp_cfg = cfg.get('interpret', {})
        generate_cams_for_fold(model, val_loader, fold, save_root, cfg['device'],
                               thresholds=interp_cfg.get('thresholds', [0.05, 0.4, 0.6]),
                               alpha=interp_cfg.get('alpha', 0.5),
                               colormap_style=interp_cfg.get('colormap', 'turbo'))


if __name__ == "__main__":
    main()