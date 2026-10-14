# scripts/train.py
"""训练入口：数据准备 → 5折交叉验证 → 预测/指标落盘。

用法：
    python scripts/train.py                                          # 默认 configs/default.yaml
    python scripts/train.py --config configs/age_gmu_noaux.yaml --gpu 1
    python scripts/train.py --force-preprocess                      # 强制重建 clinical.pkl
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# ── 仓库根锚定：scripts/ 的上一级，与运行时 cwd 无关 ──
ROOT = Path(__file__).resolve().parents[1]


# 避免没建立本项目的依赖：未做 pip install -e . 时仍可运行
sys.path.insert(0, str(ROOT / "src"))

from chiari.config import load_config
from chiari.utils.logging_setup import setup_logging
from chiari.utils.seed import set_seed
from chiari.data.preprocessing import preprocess_and_save
from chiari.engine.trainer import cross_validation


def parse_args():
    parser = argparse.ArgumentParser(description="MedicalGMU 5-fold 训练")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"),
                        help="配置文件路径（默认 configs/default.yaml）")
    parser.add_argument("--gpu", default=None, type=int,
                        help="物理 GPU 编号，覆盖 yaml 中的 run.gpu")
    parser.add_argument("--force-preprocess", action="store_true",
                        help="强制重新生成 clinical.pkl（默认存在即跳过）")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 配置：路径锚定仓库根、展开 run 目录、设 CUDA_VISIBLE_DEVICES、构造 device
    cfg = load_config(args.config, gpu=args.gpu)

    # 2. 日志：终端 + outputs/experiments/<run>/logs/train.log 双输出
    setup_logging(cfg["paths"]["logs_dir"], prefix="train")

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("MedicalGMU (Embedding Clinical + X-ray) Training")
    logger.info("config  : %s", args.config)
    logger.info("run_dir : %s", cfg["paths"]["run_dir"])
    logger.info("device  : %s | CUDA_VISIBLE_DEVICES=%s",
                cfg["device"], os.environ.get("CUDA_VISIBLE_DEVICES"))
    logger.info("=" * 60)

    # 3. 全局种子（cross_validation 内每折还会重置一次，双保险）
    set_seed(cfg["run"]["seed"])

    # 4. 数据准备
    preprocess_and_save(
        excel_path=cfg["data"]["excel_path"],
        pkl_path=cfg["data"]["pkl_path"],
        force=args.force_preprocess,
    )

    # 5. 训练：
    #    切折 → 每折 Dataset/loader/模型/训练循环 → checkpoint → 预测落盘）
    fold_results = cross_validation(cfg)

    # 6. 总结
    logger.info("训练完成，共 %d 折", len(fold_results))
    for fold, r in fold_results.items():
        m = r["metrics"]
        logger.info("  Fold %d | AUC %.3f | Acc %.3f | Sen %.3f | Spe %.3f",
                    fold + 1, m["auc"], m["accuracy"], m["sensitivity"], m["specificity"])
    logger.info("下一步: python scripts/evaluate.py  --config %s", args.config)
    logger.info("     或: python scripts/interpret.py --config %s", args.config)


if __name__ == "__main__":
    main()