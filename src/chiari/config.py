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