#https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py

from data_loader import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

#This is the baseline model

class Inline_Residual(nn.Module):
    def __init__(self):
        super().__init__()

        # Learnable mixing weights for:
        #   a * x1 + b * x2 + c * (x1 * x2)
        self.coeffs = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))

    def forward(self, x1, x2):
        # Constrain coeffs to be positive
        coeffs = torch.relu(self.coeffs)

        # Normalize so they sum to 1 (avoid blow-up)
        coeffs = coeffs / (coeffs.sum() + 1e-6)

        a, b, c = coeffs
        return a * x1 + b * x2 + c * (x1 * x2)
    
#for input X, get its CBAM output, and add up with flexible residual
#input and output are the same shape
# return res(x, cbam(x))
class CBAMAttention_channel_first(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMAttention_channel_first, self).__init__()

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

        # Optional residual scaling weight
        self.inline_residual = Inline_Residual()

        self.bn = nn.BatchNorm2d(channels) 

    def forward(self, x):
        input_x = x  # <--- FIXED: store input for residual connection

        # --- Channel Attention shrink to a stick---
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_attention = self.channel_sigmoid((avg_out + max_out)/2.0)
        x = x * channel_attention

        # --- Spatial Attention shrink to a plane---
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_out, max_out], dim=1)))
        
        x_cbam = x * spatial_attention

        # Residual output with learnable scaling
        resid = self.inline_residual(input_x, x_cbam)
        
        return self.bn(resid)

class CBAMAttention_spatial_first(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        # ----- Spatial Attention -----
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.spatial_sigmoid = nn.Sigmoid()

        # ----- Channel Attention -----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        self.inline_residual = Inline_Residual()
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        # =====================================================
        # 1. SPATIAL ATTENTION FIRST
        # =====================================================
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_sigmoid(
            self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        )
        x = x * spatial_att

        # =====================================================
        # 2. CHANNEL ATTENTION SECOND
        # =====================================================
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        channel_att = self.channel_sigmoid((avg_out + max_out) / 2.0)
        x = x * channel_att

        # Residual mixing
        out = self.inline_residual(residual, x)
        return self.bn(out)

    
class ConvMixerStage(nn.Module):
    def __init__(self, in_ch, out_ch, patch_size, depth):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(out_ch)
        )

        blocks = []
        for _ in range(depth):
            blocks.append(nn.ModuleDict({
                'dw': nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch),
                    nn.GELU(),
                    nn.BatchNorm2d(out_ch)
                ),
                'pw': nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(out_ch)
                )
            }))
        self.blocks = nn.ModuleList(blocks)

        # Add CBAM here
        self.attn = CBAMAttention_channel_first(out_ch)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x_in = x
            x = block['dw'](x) 
            x = block['pw'](x)
            x = x + x_in

        # <- apply CBAM once at the end
        x = self.attn(x)
        return x
    
class FullModel(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm2d(64)
        )
        
        self.stage1 = ConvMixerStage(64, 128, patch_size=2, depth=3)
        self.stage2 = ConvMixerStage(128, 256, patch_size=2, depth=3)
        self.stage3 = ConvMixerStage(256, 512, patch_size=2, depth=3)
        self.stage4 = ConvMixerStage(512, 256, patch_size=2, depth=3)

        # Adaptive global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output [B, C, 1, 1]

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.global_pool(x)      # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)    # Flatten to [B, 512]

        x = self.dropout(x)          # Apply dropout here
        x = self.fc(x)               # [B, num_classes]
        return x

# Quick test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(num_classes=120).to(device)

    count_params_and_shapes(model, input_size=(1, 3, 224, 224))
    
