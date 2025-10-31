#https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/tinynet_pytorch/tinynet.py

from data_loader import *

import timm
from timm.models._efficientnet_builder import decode_arch_def, round_channels
from timm.models.efficientnet import EfficientNet
from timm.layers import Swish

#It is a custom EfficientNet that is out of NAS
def TinyNet(depth_multiplier=0.5, channel_multiplier=1.0, depth_trunc=None, **kwargs):
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]

    block_args = decode_arch_def(arch_def, depth_multiplier, depth_trunc=depth_trunc)
    num_features = max(1280, round_channels(1280, channel_multiplier, 8, None))

    model = EfficientNet(
        block_args=block_args,
        num_features=num_features,
        stem_size=32,
        fix_stem=True,
        act_layer=Swish,
        **kwargs,  # e.g. num_classes, drop_rate, etc.
    )
    return model



# Test
if __name__ == "__main__":
    import timm

    # Example: Create and modify
    model = TinyNet(num_classes=10).to(device)

    
    count_params_and_shapes(model)
