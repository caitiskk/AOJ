# -*- coding: utf-8 -*-
"""
主程序入口：支持通过命令行参数一键运行多模型 5-Fold 评估、显著性检验与科研图表导出
用法示例：
    python main.py --output_dir ./outputs --threshold 0.25 --ext svg
"""
import os
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
print(f'DEBUG：{ROOT}')
from compare.config import (
    MODEL_FILES,
    TARGET_MODEL,
    DEFAULT_FIXED_THRESHOLD,
    DEFAULT_BOOTSTRAP_ITERS,
    DEFAULT_RANDOM_SEED,
    set_journal_plot_style,
)
print(f'DEBUG：{MODEL_FILES}')
from compare.utils import (
    generate_comparison_tables,
    plot_roc_curves_multi,
    plot_pr_curves_multi,
    export_all_cms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="5-Fold 多模型交叉验证评估与统计显著性检验工具")
    parser.add_argument('--output_dir', type=str, default='./outputs', help='图表与统计表输出目录')
    parser.add_argument('--threshold', type=float, default=DEFAULT_FIXED_THRESHOLD, help='固定分类阈值 (默认: 0.25)')
    parser.add_argument('--bootstraps', type=int, default=DEFAULT_BOOTSTRAP_ITERS, help='Bootstrap 重采样次数 (默认: 2000)')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED, help='随机种子 (默认: 42)')
    parser.add_argument('--ext', type=str, default='svg', choices=['svg', 'png'], help='图片导出格式 (svg 或 png)')
    parser.add_argument('--show_percent', action='store_true', help='在单独导出的混淆矩阵中显示行百分比')
    return parser.parse_args()


def main():
    args = parse_args()
    # 让 os 相对路径（如 open、pkl 路径）统一以根目录为基准
    os.chdir(ROOT)
    os.makedirs(args.output_dir, exist_ok=True)
    set_journal_plot_style()

    # 检查模型文件是否存在
    missing = [f"{k}: {v}" for k, v in MODEL_FILES.items() if not os.path.exists(v)]
    if missing:
        print("[提示] 以下模型预测文件尚未就绪，请确认路径配置 (config.py):")
        for m in missing:
            print("  -", m)
        return

    # 1. 绘制多模型 5-Fold ROC 与 PR 曲线
    plot_roc_curves_multi(
        MODEL_FILES,
        save_path=os.path.join(args.output_dir, f'roc_multi.{args.ext}')
    )
    plot_pr_curves_multi(
        MODEL_FILES,
        save_path=os.path.join(args.output_dir, f'pr_multi.{args.ext}')
    )

    # 2. 生成统计评估报表并保存 CSV
    generate_comparison_tables(
        MODEL_FILES,
        target_model=TARGET_MODEL,
        fixed_threshold=args.threshold,
        n_bootstraps=args.bootstraps,
        seed=args.seed,
        save_dir=args.output_dir,
    )

    # # 3. 绘制 1x4 混淆矩阵对比图与各模型独立混淆矩阵图
    # plot_multi_cm(
    #     MODEL_FILES,
    #     show_percent=True,
    #     save_path=os.path.join(args.output_dir, f'cm_4models.{args.ext}')
    # )
    export_all_cms(
        MODEL_FILES,
        save_dir=os.path.join(args.output_dir, 'cm_figures'),
        file_ext=args.ext,
        show_title=False,
        show_percent=args.show_percent,
    )
    print(f"\n[完成] 所有图表与统计检验表格已输出至: {os.path.abspath(args.output_dir)}")


if __name__ == '__main__':
    main()
