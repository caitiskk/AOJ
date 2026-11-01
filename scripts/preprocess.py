# scripts/preprocess.py
"""命令行入口。用法：
    python scripts/preprocess.py --config configs/default.yaml [--force]
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # scripts/ → 仓库根

from chiari.config import load_config                       # pip install -e . 后可直接 import
from chiari.data.preprocessing import preprocess_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--force", action="store_true", help="强制重新生成 pkl")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocess_and_save(
        excel_path=cfg["data"]["excel_path"],
        pkl_path=cfg["data"]["pkl_path"],
        force=args.force,
    )


if __name__ == "__main__":
    main()