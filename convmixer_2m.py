import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import *

class ConvMixerLayer(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        
        self.depthwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        x = x + self.depthwise(x) #Residual after spatial mixing, before channel mixing
        x = self.pointwise(x)
        return x

class ConvMixerModel(nn.Module):
    def __init__(self, dim=256, depth=8, kernel_size=5, patch_size=4, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        self.blocks = nn.Sequential(*[
            ConvMixerLayer(dim, kernel_size) for _ in range(depth)
        ])
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    # Example usage
    model = ConvMixerModel(dim=512, depth=6, kernel_size=5, patch_size=5, n_classes=10)
    count_params_and_shapes(model)
