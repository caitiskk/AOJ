# src/chiari/data/transforms.py
"""图像增强 pipeline。纯函数，不读 CONFIG。"""
from torchvision import transforms

# ImageNet 预训练 ResNet 对应的归一化参数
# GradCAM 反归一化也要用 → 提到模块级共享，防止两处不同步
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(phase='train', img_size=224):
	if phase == 'train':
		return transforms.Compose([
			transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.ColorJitter(brightness=0.3, contrast=0.3), # 强化骨骼边缘对比度
			transforms.Grayscale(num_output_channels=3),
			transforms.RandomRotation(degrees=5),
			transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
		])
	else:
		return transforms.Compose([
			transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
		])


