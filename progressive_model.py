
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
##--------------------------DCT IDCT Layers------------------------

class DctLayer(nn.Module):
    _block_dct_basis = None
    _corner_masks = None
    
    def __init__(self,  block=8):
        super(DctLayer, self).__init__()
                
        self.block_size = block #default is 8                

        # Initialize shared static tensors if not already done
        if DctLayer._block_dct_basis is None:
            DctLayer._block_dct_basis = self.generate_block_dct_basis().to(device)
        if DctLayer._corner_masks is None:
            DctLayer._corner_masks = self.get_corner_masks().to(device)

        # Access them through self for code clarity
        self.block_dct_basis = DctLayer._block_dct_basis
        self.masks = DctLayer._corner_masks       

    def visualize_dct_basis(self, dct_basis_2d):
        N = dct_basis_2d.shape[0]
        fig, axes = plt.subplots(N, N, figsize=(10, 10))
        for u in range(N):
            for v in range(N):
                ax = axes[u, v]
                basis_img = dct_basis_2d[u, v]  # shape: [N, N]
                ax.imshow(basis_img, cmap='gray', interpolation='nearest')
                ax.axis('off')
                ax.set_title(f'({u},{v})', fontsize=6)
        plt.suptitle('2D DCT Basis Functions')
        plt.tight_layout()
        plt.show()
    
    #return 4 DCT block corner masks
    def get_corner_masks(self):
        N = self.block_size
        mid = N // 2
    
        masks = torch.zeros(4, N, N)  # Change from 3 → 4
    
        masks[0, :mid, :mid] = 1.0  # Top-Left (TL)
        masks[1, :mid, mid:] = 1.0  # Top-Right (TR)
        masks[2, mid:, :mid] = 1.0  # Bottom-Left (BL)
        masks[3, mid:, mid:] = 1.0  # Bottom-Right (BR) ← NEW
    
        masks = masks.view(4, 1, 1, 1, 1, N, N)  # Change 3 → 4
    
        return masks
    
    #Generate static DCT 2D basis kernels [N, N, N, N]
    def generate_block_dct_basis(self):
        N = self.block_size
        x = torch.arange(N, dtype=torch.int64).view(-1, 1)
        u = torch.arange(N, dtype=torch.int64).view(1, -1)
    
        alpha = torch.sqrt(torch.tensor(2.0 / N)) * torch.ones(N)
        alpha[0] = torch.sqrt(torch.tensor(1.0 / N))
    
        x_float = x.float()
        u_float = u.float()
    
        dct_basis = alpha[u] * torch.cos(torch.pi * u_float * (2 * x_float + 1) / (2 * N))  # [N, N]    
        dct_basis_2d = dct_basis.unsqueeze(1).unsqueeze(3) * dct_basis.unsqueeze(0).unsqueeze(2).contiguous()  # [N, N, N, N]
        #self.visualize_dct_basis(dct_basis_2d)
        
        return dct_basis_2d

    
    # input image: [b, d, h, w]
    # output dct_coeffs: [b, d, h_blocks, w_blocks, block_size, block_size]
    # Given the image and DCT 2D basis functions for the given block size, the image's DCT coefficients are returned for each block. 
    def img_to_block_dct_coeffs(self, image):
        N = self.block_size
        b, d, h, w = image.shape
        
        assert h % N == 0 and w % N == 0, "Image dimensions must be divisible by block size."    
        num_blocks_h = h // N
        num_blocks_w = w // N
    
        img_blocks = (
            image.view(b, d, num_blocks_h, N, num_blocks_w, N)
                 .permute(0, 1, 2, 4, 3, 5)  # → (b, d, h_blocks, w_blocks, block_h, block_w)
                 .reshape(b, d, -1, N, N).contiguous()  # Flatten each block to vector
        )  
    
        dct_coeffs = torch.einsum("sdbij,ijkl->sdbkl", img_blocks, self.block_dct_basis)  # Shape: (b, d, num_blocks, block_size, block_size)
        return dct_coeffs.view(b, d, num_blocks_h, num_blocks_w, N, N).contiguous() #contiguous stays in one memory block

    # Input dct_coeffs: [B, C, H_N, W_N, N, N]
    # Return: [3, B, C, H_N, W_N, N, N]
    def get_corner_dct_coeffs(self, dct_coeffs):
        return dct_coeffs.unsqueeze(0) * self.masks.contiguous()  # [4, B, D, H, W, bs, bs]
        
    # input dct_coeffs: [b, d, num_blocks_h, num_blocks_w, block_size, block_size]
    # outout: Merged DCT coeffs [b, d, h, w]
    def merge_dct_coefficients(self, dct_coeffs):
        b, d, num_blocks_h, num_blocks_w, block_size, block_size = dct_coeffs.shape        
        dct_coeffs = dct_coeffs.permute(0, 1, 2, 4, 3, 5)  # (b, d, num_blocks_h, block_size, num_blocks_w, block_size)        
        return dct_coeffs.reshape(b, d, num_blocks_h * block_size, num_blocks_w * block_size).contiguous()
    
    #Input dct_coeffs: [b, d, num_blocks_h, num_blocks_w, block_size, block_size]
    #output: image of [b, d, h, w]
    def idct_block_coeffs_to_img(self, dct_coeffs):
        b, d, h_blocks, w_blocks, N, N = dct_coeffs.shape
        h = h_blocks * N
        w = w_blocks * N
        assert N == self.block_size
    
        # Flatten DCT blocks
        dct_coeffs = dct_coeffs.reshape(b, d, h_blocks * w_blocks, N, N).contiguous()  # Shape: (d, num_blocks, block_size, block_size)    
        idct_blocks = torch.einsum("sdbkl,ijkl->sdbij", dct_coeffs, self.block_dct_basis)  # Shape: (b, d, num_blocks, block_size, block_size)
        return idct_blocks.reshape(b, d, h_blocks, w_blocks, N, N).permute(0, 1, 2, 4, 3, 5).reshape(b, d, h, w).contiguous()
    
    
    @torch.no_grad()
    def forward(self, x):
        
        dct_coeffs = self.img_to_block_dct_coeffs(x).to(device) #img to DCT coeff
        corner_coeffs = self.get_corner_dct_coeffs(dct_coeffs) # filter DCT coeff to 4 corners
        corner_images = torch.stack([self.idct_block_coeffs_to_img(c) for c in corner_coeffs], dim=0) #idct each corner back to img
       
        return corner_images
    
    
##--------------------------Input Layers------------------------

#Convert input [B, D, H, W] to [B, 5_OutC, H, W]
class DctInputLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=512, kernel_size=2, padding=0, block_size=8):
        super(DctInputLayer, self).__init__()
        
        self.dct_layer = DctLayer(block=block_size)
        dct_channels = out_channels//4

        def make_branch():
            return nn.Sequential(
                nn.Conv2d(in_channels, dct_channels, kernel_size=kernel_size, stride=kernel_size, padding=padding),
                nn.GELU(),
                nn.BatchNorm2d(dct_channels)
            )

        # One CNN for the original image, four for the IDCTs
        self.branches = nn.ModuleList([make_branch() for _ in range(4)])
               
        self.conv = nn.Sequential(
                nn.BatchNorm2d(dct_channels*4),
                nn.Conv2d(dct_channels*4, out_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU(),
                nn.BatchNorm2d(out_channels)
                )

    
    # x in the RGB image of [B, InC, H, W]
    # return [B, 4_OutC, H, W]
    def forward(self, x):
        outputs = []

        # Step 1: IDCT images from DCT-masked corners
        with torch.no_grad():    
            idct_imgs = self.dct_layer(x)  # [4, B, C, H, W] of 4 corners after IDCT

        #outputs.append(self.branches[0](x))
        for i in range(4): #skip the last HH corner
            outputs.append(self.branches[i](idct_imgs[i]))
            
        combined = torch.cat(outputs, dim=1)  # Shape: [B, 4*out_channels, H, W]

        output = self.conv(combined)

        return output
    
    
class InputLayer(nn.Module):
    def __init__(self, dim, patch_size, padding):    
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, padding=padding),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    
    def forward(self, x):
        x = self.input_layer(x)
        return x

  

################################## ATTENTION MODULES ###############################

     
class LastlayerAverage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.ln = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: [L, B, C, H, W]       
        bc = x[-1].mean(dim=(2, 3))# [B, C]
        
        return self.ln(bc)

class AlllayersAverage(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stack_bn = nn.BatchNorm2d(dim)
        
    def forward(self, layer_outputs):
        stacked = torch.stack(layer_outputs, dim=0)  # [L, B, C, H, W]
        weighted = stacked.mean(dim=0)  # Simple mean across L to get [B, C, H, W]    
        normalized = self.stack_bn(weighted)

        return normalized.mean(dim=(2, 3))# [B, C]
    
#Self attention, attach a random class token [C] to [B, C, H, W], use self attention to learn the class token across [B, C, H, W]
# NOT good as cls_token has to learn from scratch
class LastlayerSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [L, B, C, H, W]
        x = x[-1] #[B, C, H, W]
        B, C, H, W = x.size()

        x_mean = x.mean(dim=(2, 3))  # [B, C]
        
        x = x.view(B, C, H * W).permute(0, 2, 1)  # -> [B, N, C]

        # Expand cls token for batch
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, C]
        # Concatenate cls token with patch tokens
        x_full = torch.cat([cls_token, x], dim=1)  # [B, 1+N, C]

        # Self-attention
        attn_output, _ = self.attn(x_full, x_full, x_full)  # [B, 1+N, C]

        # Take CLS output and normalize
        cls_output = attn_output[:, 0, :]  # [B, C]
 
        return self.norm(cls_output)  # [B, C]

#-------------------------------------------------------------

#use the [B, C] mean value to attention to the full [B, N, C] to improve the mean representation
#This makes better sense, since the [B, C] avg is attentioned to the whole [B, N, C] to improve each [C] representation
class AllLayersPlaneCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(p=dropout)  # Tune rate based on overfitting signs

    def chaneel_cross_attn(self, x):
        B, C, H, W = x.shape
        # Compute global query from pooled features: [B, C], and improve it with attention 
        q = x.mean(dim=(2, 3)).unsqueeze(1)  # [B, 1, C]

        # Flatten spatial tokens: [B, C, H*W] → [B, H*W, C]
        context = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Cross-attention: query attends to spatial context
        attn_out, _ = self.attn(q, context, context)  # [B, 1, C]
        attn_out = self.dropout_attn(attn_out)

        return attn_out.squeeze(1)  # [B, C]

    def forward(self, x):
        # x: [L, B, C, H, W]
        x = torch.stack(x, dim=0).mean(dim=0)  # [L, B, C, H, W] -> [B, C, H, W]

        return self.chaneel_cross_attn(x)  # [B, C]
    
#-------------------------------------------------------------

#use the [B, C] mean value to attention to the full [B, N, C] to improve the mean representation
#This makes better sense, since the [B, C] avg is attentioned to the whole [B, N, C] to improve each [C] representation
class AllLayersChannelCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.dropout_attn = nn.Dropout(p=dropout)  # Tune rate based on overfitting signs

    def chaneel_cross_attn(self, x):
        B, C, H, W = x.shape
        # Compute global query from pooled features: [B, C], and improve it with attention 
        q = x.mean(dim=(2, 3)).unsqueeze(1)  # [B, 1, C]

        # Flatten spatial tokens: [B, C, H*W] → [B, H*W, C]
        context = x.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Cross-attention: query attends to spatial context
        attn_out, _ = self.attn(q, context, context)  # [B, 1, C]
        attn_out = self.dropout_attn(attn_out)

        return attn_out.squeeze(1)  # [B, C]

    def forward(self, x):
        # x: [L, B, C, H, W]
        x = torch.stack(x, dim=0).mean(dim=0)  # [L, B, C, H, W] -> [B, C, H, W]

        return self.chaneel_cross_attn(x)  # [B, C]

    
#use the [B, C] mean value to attention to the full [B, N, C] to improve the mean representation
#This makes better sense, since the [B, C] avg is attentioned to the whole [B, N, C] to improve each [C] representation
class LastlayerChannelCrossAttention(AllLayersChannelCrossAttention):
    def __init__(self, dim, num_heads=4):
        super().__init__(dim, num_heads)
        
    def forward(self, x):
        # x: [L, B, C, H, W]
        x = x[-1] #[B, C, H, W]

        return self.chaneel_cross_attn(x)  # [B, C]

#-------------------------------------------------------------
class WeightedSumAttention(nn.Module):
    def __init__(self, num_layers, dim):
        super().__init__()
        
        self.num_layers = num_layers
        self.layer_weights = nn.Parameter(torch.ones(num_layers)) #equal to 1 first

    def forward(self, x):        
        weights = F.softmax(self.layer_weights, dim=0)  # [L]
        x = (x * weights[:, None, None, None, None]).sum(dim=0)  # [L, B, C, H, W] -> [B, C, H, W]    
        x = x.mean(dim=[2, 3]) # [B, C, H, W] -> [B, C]
        
        return x

#Get the 3D patch
class AllLayers3DCrossAttention(nn.Module):
    def __init__(self, dim, patch_size=2, stride=None,  num_heads=4):
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.embed_dim = dim * patch_size * patch_size  # Flattened patch dim
        self.proj_dim = dim  # Reduced dimension for attention

        # FCN projection from embed_dim → proj_dim
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, self.proj_dim),
            nn.GELU(),            
            nn.LayerNorm(self.proj_dim),  # prep for input to next layer
            nn.Dropout(p=dropout)
        )
        
        self.attn = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=num_heads, batch_first=True)

    def extract_patches(self, x):
        B, C, H, W = x.shape
        h, w = self.patch_size, self.patch_size

        # Extract overlapping patches
        patches = x.unfold(2, h, self.stride).unfold(3, w, self.stride)  # [B, C, num_H, num_W, h, w]
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [B, num_H, num_W, C, h, w]
        B, num_H, num_W, C, h, w = patches.shape
        patches = patches.reshape(B, num_H * num_W, C * h * w)  # [B, N, embed_dim]

        return patches

    def cross_attn(self, x): # x: [B, C, H, W]
        patches = self.extract_patches(x)  # [B, N, embed_dim]
        patches_proj = self.proj(patches)  # [B, N, proj_dim]

        # Use mean of projected patches as query
        q = patches_proj.mean(dim=1, keepdim=True)  # [B, 1, proj_dim]

        # Attention
        attn_out, _ = self.attn(q, patches_proj, patches_proj)  # [B, 1, proj_dim]

        return attn_out.squeeze(1)  # [B, proj_dim]

    def forward(self, x):
        # x: [L, B, C, H, W]
        x = torch.stack(x, dim=0).mean(dim=0)  # [L, B, C, H, W] -> [B, C, H, W]
        
        return self.cross_attn(x)  # [B, proj_dim]

class LastLayer3DCrossAttention(AllLayers3DCrossAttention):
    def __init__(self, dim, patch_size=2, stride=None,  num_heads=4):
        super().__init__(dim, patch_size, stride, num_heads)

    def forward(self, x):
        # x: [L, B, C, H, W]
        x = x[-1]  # Use final feature map [B, C, H, W]
        
        return self.cross_attn(x)  # [B, proj_dim]
    
    
#-------------------------------FINAL LAYER------------------------------
    
class FinalLayer(nn.Module):
    def __init__(self, num_layers, dim, n_classes, dropout=0.1):    
        super().__init__()
        
        self.num_layers = num_layers
        self.attn_layer = WeightedSumAttention(num_layers, dim)  #[L, B, C, H, W] -> [B, C]
        #LastlayerChannelCrossAttention(dim), #channel as attention vector
        #AllLayersChannelCrossAttention(dim), #channel as attention vector
        #LastLayer3DCrossAttention(dim, patch_size=2,  =1), #3D Attention
        #AllLayers3DCrossAttention(dim, patch_size=2, stride=1), #3D Attention
        #LastlayerAverage(dim),
        #AlllayersAverage(dim),
        #return AllLayers3DCrossAttention(dim, patch_size=2, stride=1) #3D Attention
        #return WeightedSumAttention(self.num_layers, dim)    

        self.fcn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(p=dropout),
            
            nn.Linear(dim, n_classes)
        )
    
    def forward(self, x):
        x = self.attn_layer(x)
        x = self.fcn(x)
        return x

    
    
##--------------------------Progressive Architecture------------------------
    
# Main model with N layers, curr_layer_index is the current layer that is trained up to
# It has one InputLayer, one Final Layer, and multiple mid layers
# The model can be trained layer by layer for the middle layers
# The head and final FCN will always be trained
class ProgressiveModel(nn.Module):
    def __init__(self, mid_layer_cls, num_layers, patch_size, kernel_size, dim=128, padding=0, n_classes=10):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input layer RGB to dim-tensor
        #self.input_layer = InputLayer(dim, patch_size, padding)
        self.input_layer = DctInputLayer(in_channels=3, out_channels=dim, kernel_size=patch_size, padding=padding)

        # Intermediate layers (residual depth-wise conv + point-wise conv)
        self.mid_layers = nn.ModuleList([
            mid_layer_cls(dim, kernel_size) for _ in range(num_layers)
        ])
        
        # Final FCN layer,  [L, B, C, H, W] -> [B, C] -> FCN
        self.final_layer = FinalLayer(num_layers, dim, n_classes)
        
              
    #NOT USED
    def get_trainable_layers_params(self, curr_layer_index=None):
        # Make the first, final and the transformer layer backprop-able
        [param.__setattr__('requires_grad', True) for param in self.input_layer.parameters()]
        [param.__setattr__('requires_grad', True) for param in self.final_layer.parameters()]

        # Make the current mid layer backprop-able, all others false
        [param.__setattr__('requires_grad', i == curr_layer_index) for i, layer in enumerate(self.mid_layers) for param in layer.parameters()]

        # Only optimize the input layer, current layer, and the final head
        params = (
            list(self.input_layer.parameters()) +
            list(self.mid_layers[curr_layer_index].parameters()) +
            list(self.final_layer.parameters())
        )
        
        return params

    def set_trainable_layers_params(self, curr_layer_index=None):
        # 1. Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
    
        # 2. Enable gradient for first and final layer
        for param in self.input_layer.parameters():
            param.requires_grad = True
    
        for param in self.final_layer.parameters():
            param.requires_grad = True
    
        # 3. Enable gradient ONLY for current mid layer
        if curr_layer_index is not None and 0 <= curr_layer_index < len(self.mid_layers):
            for param in self.mid_layers[curr_layer_index].parameters():
                param.requires_grad = True
            
    #For layer-wise training, the layer output will stop at the current layer, so the subsequent layers do not contribute
    def forward(self, x, curr_layer_index=None):
        
        x = self.input_layer(x)  # Process the input through the first layer
        
        layer_outputs = []

        for i, layer in enumerate(self.mid_layers): #run up to the current layer
            x = layer(x)
            
            if curr_layer_index is None: #this is for all layer training
                layer_outputs.append(x)  # collect all layer output
            else: #this condition is for layer-wise training
                layer_outputs.append(x if i == curr_layer_index else x.detach())  # only backprop current layer            
            if(i == curr_layer_index): #do not proceed
                break

        # Pass through final classification head
        layer_outputs = torch.stack(layer_outputs, dim=0)              # [L, B, C, H, W]
        logits = self.final_layer(layer_outputs)

        return logits
    
if __name__ == "__main__":
    from torchinfo import summary
    from conv_cbam_mixer import *
    
    dim = 512
    num_layers = 6

    patch_size = 5
    kernel_size = 5
    num_classes = 10
    dropout=0.1
    
    model = ProgressiveModel(Conv_CBAM_Layer, num_layers=num_layers, dim=dim, patch_size=patch_size, kernel_size=kernel_size, n_classes=num_classes).to(device)
    
    print("-------------------------Model Summary----------------------")
    summary(model, input_size=(1, 3, 160, 160))

    count_params_and_shapes(model)

