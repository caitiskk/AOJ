# src/chiari/models/__init__.py   
"""模型层。build_model 是全项目唯一的模型构造入口。"""
from chiari.models.clinical_encoder import ClinicalEncoder
from chiari.models.fusion import MedicalGMUFusion, StandardGMUFusion
from chiari.models.multimodal import MedicalGMUNet
from chiari.models.concat import MedicalConcatNet
from chiari.models.gated import MedicalGatedNet
from chiari.models.img_only import AOJBIResNET

ARCH_REGISTRY = {
    'image_only':   AOJBIResNET,     # 单模态基线
    'concat':       MedicalConcatNet, # 拼接基线
    'gmu_standard': MedicalGatedNet,  # 标准GMU对照
    'gmu_asymmetric':   MedicalGMUNet,    # ★主模型（非对称门控）
}

def build_model(cfg, pretrained=True):
    m = cfg['model']
    arch = m.get('arch', 'gmu')
    if arch not in ARCH_REGISTRY:
        raise ValueError(f"未知 arch: {arch!r}，可选 {list(ARCH_REGISTRY)}")
    elif arch in ('image_only'):
        return ARCH_REGISTRY[arch](
        pretrained_path=cfg['data']['resnet_weight'] if pretrained else None)
    return ARCH_REGISTRY[arch](
        num_cont=m['num_cont'], clin_out=m['clin_out'], dropout=m.get('dropout', 0.4),
        pretrained_path=cfg['data']['resnet_weight'] if pretrained else None)