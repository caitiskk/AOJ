```chiari-multimodal/
├── README.md    requirements.txt
├── configs/
│   └── default.yaml（+ 消融变体若干）
├── data/
│   ├── raw/{tabular_data.xlsx, image/}
│   └── interim/clinical.pkl
├── weights/resnet18.pth      
├── src/chiari/
│   ├── config.py                      # yaml → cfg（路径锚定、run 目录、device）
│   ├── data/
│   │   ├── preprocessing.py           # preprocess_and_save
│   │   ├── transforms.py              # get_transforms + IMAGENET_MEAN/STD
│   │   └── dataset.py                 # ChiariXrayDataset + collate_fn + from_checkpoint
│   ├── models/
│   │   ├── clinical_encoder.py        # ClinicalEncoder
│   │   ├── fusion.py                  # MedicalGMUFusion
│   │   └── multimodal.py              # MedicalGMUNet + build_model
│   ├── engine/
│   │   ├── trainer.py                 # cross_validation + train_one_fold
│   │   └── evaluate.py                # evaluate_joint_level + summarize_folds
│   ├── viz/
│   │   ├── style.py  curves.py  confusion.py
│   ├── interpret/
│   │   ├── gradcam.py  overlays.py  generate.py
│   └── utils/
│       ├── seed.py  io.py  checkpoint.py
├── scripts/
│   ├── preprocess.py  train.py  evaluate.py  interpret.py   # 各 ≤30 行的薄壳
├── notebooks/
│     ├──AGMU/                              # Asymtric GMU的结果
│     ├──GMU/                               # Standard GMU的结果
│     ├──MedCAT/                            #Concate 的结果
│     ├──Resnet/                                  # Resnet18的结果
│    ├──  *.ipynb                           [Asymtric_GMU.ipynb, Concate.ipynb, Img_only.ipynb, Standard_GMU.ipynb]      # 存放这各个模型的jupyter notebook
└── outputs/
    ├── cmp/
    │   ├── roc_cmp.ipynb
    └── experiments/<run.name>[/<tag>]/
        ├── checkpoints/   # {1..5}_medicalgmu_best.pt（内嵌完整 cfg）
        ├── predictions/   # fold_predictions.pkl + 5fold_results.pkl
        ├── figures/       # roc_5fold.svg / pr_5fold.svg / cm_total.png
        ├── tables/        # metrics.csv（论文 Table 直接来源）
        ├── cams/
        └── logs/
```

ipynb 可以直接运行得到结果 # 暂时论文的结果又notebook提供 

## 运行
uv pip install -e .   # 运行前需要先建立依赖
# pip install -e . # 非uv环境
python scripts/train.py  --config configs/default.yaml [从Cat.yaml/GMU.yaml/Resnet18.yaml/default.yaml 中选择]

## Evaluate
python scripts/evaluate.py  --config configs/default.yaml [从Cat.yaml/GMU.yaml/Resnet18.yaml/default.yaml 中选择]

## 绘制热力图
python scripts/interpret.py  --config configs/default.yaml 
