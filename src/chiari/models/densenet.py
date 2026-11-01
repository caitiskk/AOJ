# src/chiari/models/densenet.py
from torchvision import models
import torch.nn as nn
import torch

class MedicalDenseNet(nn.Module):
    def __init__(self, num_cont=8, clin_out=16,pretrained_path=None):
        super().__init__()
        # 1. 图像骨干：使用 densenet121
        self.densenet = models.densenet121(weights=None)
        if pretrained_path:
          state_dict = torch.load(pretrained_path, map_location="cpu",weights_only=False)
          self.densenet.load_state_dict(state_dict, strict=False)
        num_ftrs = self.densenet.classifier.in_features  # 1024
        self.densenet.classifier = nn.Identity()         # 移除原本的分类头
        self.classifier = nn.Linear(num_ftrs, 2)

    def forward(self, img, cat_features, cont_features):
        img_feat = self.densenet(img)                                    # [B, 1024]
        logits = self.classifier(img_feat)                                 # [B, 2]
        return logits