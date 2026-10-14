# src/chiari/utils/logging_setup.py
"""日志初始化：终端 + 文件双输出。

"""
import logging
import sys
from pathlib import Path


def setup_logging(logs_dir, prefix="run", level=logging.INFO):
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()          # 防重复调用时 handler 叠加

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = logging.FileHandler(logs_dir / f"{prefix}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    return root