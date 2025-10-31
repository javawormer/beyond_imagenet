#https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py

from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# Core Building Blocks
# -------------------

class ConvBNAct(nn.Sequential):
    def __init__(self, in_c, out_c, k=3, s=1, p=1, act=nn.SiLU()):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            act
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_c, out_c, stride, exp_ratio):
        super().__init__()
        hidden_c = in_c * exp_ratio
        self.use_res = stride == 1 and in_c == out_c
        self.block = nn.Sequential(
            nn.Conv2d(in_c, hidden_c, 1, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(),
            
            nn.Conv2d(hidden_c, hidden_c, 3, stride, 1, groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(),
            
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, x):
        out = self.block(x)
        return out + x if self.use_res else out

class MobileViTBlock(nn.Module):
    def __init__(self, in_c, transformer_dim, ffn_dim, n_heads, n_blocks=2, patch_h=2, patch_w=2):
        super().__init__()
        self.local_conv = ConvBNAct(in_c, in_c, 3, 1, 1)
        self.reduce_conv = ConvBNAct(in_c, transformer_dim, 1, 1, 0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.expand_conv = ConvBNAct(transformer_dim, in_c, 1, 1, 0)

        self.patch_h, self.patch_w = patch_h, patch_w
        self.in_c = in_c

    def forward(self, x):
        y = self.local_conv(x)
        B, C, H, W = y.shape
        # unfold (B, transformer_dim, H*W) tokens
        z = self.reduce_conv(y)
        z = z.flatten(2).transpose(1, 2)
        z = self.transformer(z)
        z = z.transpose(1, 2).view(B, -1, H, W)
        z = self.expand_conv(z)
        return x + y + z

# -------------------
# MobileViT-Small Model
# -------------------

class MobileViT_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = ConvBNAct(3, 32, 3, 2, 1)  # 256→128

        self.layer1 = nn.Sequential(
            InvertedResidual(32, 64, 1, 4)
        )
        self.layer2 = nn.Sequential(
            InvertedResidual(64, 128, 2, 4),
            InvertedResidual(128, 128, 1, 4)
        )
        self.layer3 = nn.Sequential(
            InvertedResidual(128, 128, 2, 4),
            MobileViTBlock(128, transformer_dim=144, ffn_dim=288, n_heads=4),
        )
        self.layer4 = nn.Sequential(
            InvertedResidual(128, 256, 2, 4),
            MobileViTBlock(256, transformer_dim=192, ffn_dim=384, n_heads=4),
        )
        self.layer5 = nn.Sequential(
            InvertedResidual(256, 256, 2, 4),
            MobileViTBlock(256, transformer_dim=240, ffn_dim=480, n_heads=4),
        )

        self.conv_exp = ConvBNAct(256, 512, 1, 1, 0)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv_exp(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

# -------------------
# Quick Test
# -------------------

if __name__ == "__main__":
    model = MobileViT_Small(num_classes=10).to(device)
    
    count_params_and_shapes(model)
