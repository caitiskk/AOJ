# src/chiari/config.py
import os
from pathlib import Path

import torch
import yaml


def _find_repo_root(start: Path) -> Path:
    """向上找到含 src/ 的目录。这样 yaml 放 configs/ 或 configs/experiments/ 都能正确锚定。"""
    root = start.resolve()
    while root != root.parent and not (root / 'src').is_dir():
        root = root.parent
    return root


def adapt_config(raw_cfg: dict) -> dict:
    """
    自动兼容：识别传入的配置是旧版 Notebook 扁平 CONFIG 还是新版工程化嵌套 YAML，
    统一输出为标准的嵌套字典。
    """
    if not isinstance(raw_cfg, dict):
        raise TypeError(f"Config must be dict, got {type(raw_cfg)}")

    # 如果已经具有 'data' 和 'training'，说明是新版工程化嵌套配置，直接返回
    if 'data' in raw_cfg and 'training' in raw_cfg:
        return raw_cfg
    # print('DEBUG: GET HERE?')
    # 旧版 Notebook 扁平 CONFIG -> 映射转为新版工程化嵌套结构
    legacy = raw_cfg
    adapted = {
        'run': {
            'seed': legacy.get('seed', 42),
            'name': 'legacy_notebook',
        },
        'data': {
            'excel_path': legacy.get('excel_path1','data/raw/tabular_data.xlsx'),
            'pkl_path': legacy.get('pkl_pat1h', 'data/interim/clinical.pkl'),
            'image_root': legacy.get('image_root1','data/raw/image'),
            'resnet_weight': legacy.get('weights/resnet18.pth'),
        },
        'model': {
            'arch': legacy.get('arch', 'gmu_asymmetric'),
            'num_cont': legacy.get('num_cont', 8),
            'clin_out': legacy.get('clin_out', 16),
            'dropout': legacy.get('dropout', 0.4),
            'use_aux_task': legacy.get('use_aux_task', True),
        },
        'training': {
            'img_size': legacy.get('img_size', 224),
            'batch_size': legacy.get('batch_size', 8),
            'epochs': legacy.get('epochs', 100),
            'n_splits': legacy.get('n_splits', 5),
            'lr_backbone': legacy.get('lr_backbone', 1e-5),
            'lr_head': legacy.get('lr_head', 1e-3),
            'weight_decay': legacy.get('weight_decay', 1e-4),
            'aux_weight': legacy.get('aux_weight', 0.3),
            'pos_weight': legacy.get('pos_weight', 3),
            'patience': legacy.get('patience', 15),
        },
        'eval': {
            # 优先从 config 里取，取不到则用默认 0.25
            'threshold': legacy.get('threshold', 0.25),
        },
        '_is_legacy': True,
    }
    return adapted




def load_config(path, gpu=None):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    root = _find_repo_root(path.parent)

    # ── GPU / device──
    gpu = gpu if gpu is not None else cfg['run'].get('gpu')
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cfg['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 数据路径锚定仓库根（修复对 Path.cwd() 的依赖）──
    for k, v in cfg['data'].items():
        if isinstance(v, str) and not Path(v).is_absolute():
            cfg['data'][k] = str(root / v)

    # ── 展开 run 目录──
    run = cfg['run']
    exp_root = root / cfg['paths']['experiment_root']
    run_dir = exp_root / str(run['name'])
    if run.get('tag'):
        run_dir = run_dir / str(run['tag'])            # 加入tag后缀，避免不同实验混淆
    cfg['paths'] = {
        'run_dir':         str(run_dir),
        'save_dir':        str(run_dir / 'checkpoints'),    #  ← 训练中保存的模型
        'predictions_dir': str(run_dir / 'predictions'),
        'figures_dir':     str(run_dir / 'figures'),
        'tables_dir':      str(run_dir / 'tables'),
        'logs_dir':        str(run_dir / 'logs'),
    }
    for d in cfg['paths'].values():                     
        Path(d).mkdir(parents=True, exist_ok=True)
    return cfg