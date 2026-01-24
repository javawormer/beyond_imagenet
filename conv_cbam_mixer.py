from data_loader import *

import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        
        self.fn = fn
        self.coeffs = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))  # [a, b, c]
        
    def forward(self, x1):
        x2 = self.fn(x1)
        
        #raw = torch.relu(self.coeffs)  # keep positive
        #s = raw.sum().clamp_min(1e-6)
        a, b, c = self.coeffs

        return (a * x1 + b * x2 + c * (x1 * x2) )
        
class Inline_Residual(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.coeffs = nn.Parameter(torch.tensor([1.0, 1.0, 0.0]))  # [a, b, c]
        
    def forward(self, x1, x2):      
        #raw = torch.relu(self.coeffs)  # keep positive
        #s = raw.sum().clamp_min(1e-6)
        #a, b, c = raw / s #norm  
        a, b, c = self.coeffs
        
        return (a * x1 + b * x2 + c * (x1 * x2) )

#for input X, get its CBAM output
#input and output are the same shape
# return cbam(x)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()

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
       
        return self.bn(x_cbam)
    
#for input X, get its CBAM output, and add up with flexible residual
#input and output are the same shape
# return res(x, cbam(x))
class CBAMAttention(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMAttention, self).__init__()

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
        
#Concatenate two layers of ConvMixer, to shink dim to dim//2, then expand back to dim
#So one layer equals to two layers of ConvMixer
class ShrinkExpander(nn.Module):
    def __init__(self, dim, kernel_size, shrink_factor=2):
        super().__init__()
    
        self.shrinkExpander = nn.Sequential(
                    Residual(nn.Sequential(
                        #Shrink 
                        nn.Conv2d(dim, dim//shrink_factor, kernel_size, groups=dim//shrink_factor, padding='same'),
                        nn.GELU(),
                        nn.BatchNorm2d(dim//shrink_factor), #plane wise
                        
                        #bottleneck channel shuffle
                        nn.Conv2d(dim//shrink_factor, dim//shrink_factor, kernel_size=1), #channel wise
                        nn.GELU(),
                        nn.BatchNorm2d(dim//shrink_factor), #doing it after GELU will make the data even around 0
                        
                        #Expand 
                        nn.Conv2d(dim//shrink_factor, dim, kernel_size, groups=dim//shrink_factor, padding='same'),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.BatchNorm2d(dim),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim) #doing it after GELU will make the data even around 0
          )

        self.cbam = CBAMAttention(dim) 
        

    def forward(self, x):
        out = self.shrinkExpander(x)
        return self.cbam(out)
    
#Classic ConvMixer
class Conv_CBAM_Layer(nn.Module):
    def __init__(self, dim, kernel_size):    
        super().__init__()

        self.mixer = nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )), #residual before channnel mixing
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim) #doing it after GELU will make the data even around 0                    
          )
          
        self.cbam = CBAMAttention(dim) 
    
    def forward(self, x):
        out = self.mixer(x)
        return self.cbam(out)


class MB_CBAM_Conv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, shrinker=2):
        super().__init__()
        mid_channels = in_channels // shrinker

        self.mbconv = nn.Sequential( #1*1, k*k layerwise, 1*1
            # 1*1 Expansion
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),

            # Depthwise Convolution
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),

            # 1*1 Projection
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        
        self.cbam = CBAMAttention(in_channels)
        self.inline_residual = Inline_Residual()

    def forward(self, x):
        out = self.mbconv(x)
        res = self.inline_residual(x, out) 
        
        return self.cbam(res)

class ConvCBAMMixerModel(nn.Module):
    def __init__(self, dim=512, depth=6, kernel_size=5, patch_size=2, n_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        self.blocks = nn.Sequential(*[
            Conv_CBAM_Layer(dim, kernel_size) for _ in range(depth)
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

# 🔍 Check param count
if __name__ == "__main__":
    model = ConvCBAMMixerModel(dim=512, depth=6, kernel_size=5, patch_size=2, n_classes=10)
    count_params_and_shapes(model)
