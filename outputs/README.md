# 5-Fold 多模型交叉验证评估与统计检验工程

本工程由 Jupyter Notebook 实验脚本重构而来，消除了多单元格重复定义的函数与硬编码逻辑，采用模块化分层设计。

## 目录结构

```text
compare/
├── config.py               # 模型路径映射、配色方案、全局期刊绘图样式与默认超参数
├── main.py                 # 命令行主入口（一键生成曲线、显著性表、混淆矩阵）
├── requirements.txt        # 依赖环境配置
├── README.md               # 工程说明文档
└── utils/
    ├── __init__.py         # 工具包导出接口
    ├── data_loader.py      # PKL/Tensor 数据加载、样本按 (fold, joint_id) 对齐、Youden 阈值寻优
    ├── stat_tests.py       # DeLong 检验、配对 McNemar 检验、Paired Bootstrap 检验
    ├── evaluator.py        # 5-Fold 6大指标计算、均值±标准差汇总及 p 值报表导出
    └── plotter.py          # 期刊级多模型 ROC/PR 曲线及混淆矩阵绘制
```

## 快速运行

```bash
# 安装依赖
pip install -r requirements.txt

# 默认运行（输出 SVG 矢量图与 CSV 统计表至 ./outputs 目录）
python ./outputs/compare/main.py

# 自定义阈值与导出 PNG 格式
python ./outputs/compare/main.py --output_dir ./compares --threshold 0.25 --ext png --show_percent
```
