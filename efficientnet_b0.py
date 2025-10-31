from data_loader import *

import copy, math
from functools import partial
from typing import Sequence, Callable, Optional, Union
import torch
from torch import nn, Tensor
from torchvision.ops import StochasticDepth
from torchvision.models.efficientnet import _MBConvConfig, MBConvConfig, MBConv, Conv2dNormActivation, SqueezeExcitation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

#Identical architecture with official model from torchvision.models import efficientnet_b0
class EfficientNet(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[_MBConvConfig],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        last_channel: Optional[int] = None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: list[nn.Module] = []
        # Stem
        first_out = inverted_residual_setting[0].input_channels
        layers.append(Conv2dNormActivation(3, first_out, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU))

        # MBConv stages
        total_blocks = sum(cfg.num_layers for cfg in inverted_residual_setting)
        block_id = 0
        for cfg in inverted_residual_setting:
            stage: list[nn.Module] = []
            for _ in range(cfg.num_layers):
                block_cfg = copy.copy(cfg)
                if stage:
                    block_cfg.input_channels = cfg.out_channels
                    block_cfg.stride = 1
                sd_prob = stochastic_depth_prob * block_id / total_blocks
                stage.append(cfg.block(block_cfg, sd_prob, norm_layer))
                block_id += 1
            layers.append(nn.Sequential(*stage))

        # Head
        last_in = inverted_residual_setting[-1].out_channels
        last_out = last_channel or 4 * last_in
        layers.append(Conv2dNormActivation(last_in, last_out, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))
        self.features = nn.Sequential(*layers)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_out, num_classes),
        )

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)
        return x

    forward = _forward_impl


#this is the original B0 model with about 4M params 
def EfficientNet_B0_Model(num_classes: int = 1000):
    cfg = [MBConvConfig(1, 3, 1, 32, 16, 1),
           MBConvConfig(6, 3, 2, 16, 24, 2),
           MBConvConfig(6, 5, 2, 24, 40, 2),
           MBConvConfig(6, 3, 2, 40, 80, 3),
           MBConvConfig(6, 5, 1, 80, 112, 3),
           MBConvConfig(6, 5, 2, 112, 192, 4),
           MBConvConfig(6, 3, 1, 192, 320, 1)]
    return EfficientNet(cfg, dropout=0.2, stochastic_depth_prob=0.2, num_classes=num_classes)

#2.3M params
def EfficientNet_B0_2M(num_classes: int = 1000):
    cfg = [MBConvConfig(1, 3, 1, 32, 16, 1),
           MBConvConfig(6, 3, 2, 16, 24, 2),
           MBConvConfig(6, 5, 2, 24, 40, 2),
           MBConvConfig(6, 3, 2, 40, 80, 3),
           MBConvConfig(6, 5, 1, 80, 192, 3),
           MBConvConfig(6, 3, 1, 192, 256, 1)]
    return EfficientNet(cfg, dropout=0.2, stochastic_depth_prob=0.2, num_classes=num_classes)

# ✅ Test
if __name__ == "__main__":
    from torchinfo import summary

    model = EfficientNet_B0_2M(num_classes=10).to(device)
    print("-------------------------My Efficient B0 Model----------------------")
    summary(model, input_size=(1, 3, 224, 224))
   
    count_params_and_shapes(model)
    #count_params(model)
    
