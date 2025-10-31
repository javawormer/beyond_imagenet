import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch.nn as nn
import os
import math

from fvcore.nn import FlopCountAnalysis, parameter_count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_up_down_lr(epochs, current_epoch, lr_min, lr_max):
    half_cycle = epochs // 2
    if current_epoch <= half_cycle:
        # increasing phase
        lr = lr_min + (lr_max - lr_min) * (current_epoch / half_cycle)
    else:
        # decreasing phase
        lr = lr_max - (lr_max - lr_min) * ((current_epoch - half_cycle) / (epochs - half_cycle))
    return lr

def get_up_cos_lr(epochs, current_epoch, lr_min, lr_max, warmup_epochs):
    if current_epoch < warmup_epochs:
        return lr_max * (current_epoch + 1) / warmup_epochs  # Linear warm-up from 0 → 1
    else:
        # Cosine annealing from base_lr → min_lr
        cosine_epochs = epochs - warmup_epochs
        progress = (current_epoch+1 - warmup_epochs) / cosine_epochs
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return lr_min + (lr_max - lr_min) * cosine_decay

def load_model_wts(model, dataset, patch_size, hidden_dim, num_layers):
    weights_path = f"progressive_{dataset}_{patch_size}_{hidden_dim}_{num_layers}_weights.pth"
    if os.path.exists(weights_path):        
        model.load_state_dict(torch.load(weights_path))
        
def count_params_and_shapes(model, input_size=(1, 3, 160, 160)):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    hooks = []
    total_params = 0
    total_trainable = 0
    total_flops = 0
    
    seen_params = set()
    layer_infos = []

    module_to_name = {m: n for n, m in model.named_modules()}

    def hook_fn(module, input, output):
        # Skip container modules
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)) or len(list(module.children())) > 0:
            return

        name = module.__class__.__name__
        full_path = module_to_name.get(module, '')
        block = '.'.join(full_path.split('.')[:-1])  # e.g., stem.0 → stem

        input_shape = tuple(input[0].shape) if isinstance(input, (tuple, list)) else tuple(input.shape)
        output_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else str(output)

        # Collect kernel shapes and count params
        kernel_shapes = []
        num_params = 0
        trainable_params = 0

        for p in module.parameters(recurse=False):
            if id(p) in seen_params:
                continue
            seen_params.add(id(p))

            num_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()

            if p.ndim == 4:
                reordered = (p.shape[1], p.shape[0], p.shape[2], p.shape[3])  # [in, out, h, w]
                kernel_shapes.append(reordered)
            elif p.ndim >= 2:
                kernel_shapes.append(tuple(p.shape))

        nonlocal total_params, total_trainable
        total_params += num_params
        total_trainable += trainable_params

        layer_infos.append((block, name, input_shape, output_shape, trainable_params, kernel_shapes))

    # Register hooks
    for module in model.modules():
        if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            hooks.append(module.register_forward_hook(hook_fn))

    # Forward pass with dummy input to trigger hooks
    model.eval()
    dummy_input = torch.randn(*input_size).to(device)
    with torch.no_grad():
        model(dummy_input)
        total_flops = FlopCountAnalysis(model, dummy_input)
        total_flops.unsupported_ops_warnings(False)


    # Remove hooks
    for h in hooks:
        h.remove()

    # Print results
    print("\nLayer-wise grouped shape and parameter analysis:")
    prev_block = None
    for idx, (block, name, in_shape, out_shape, params, kernels) in enumerate(layer_infos):
        if block != prev_block:
            print(f"\nBlock: {block or '[root]'}")
            print("-" * 120)
            prev_block = block

        k_str = ', '.join([str(k) for k in kernels]) if kernels else '-'
        print(f"{idx+1:>2}. {name:<20} ({params:,})".ljust(40) +
              f"| In: {str(in_shape):<20} → Out: {str(out_shape):<20} | Kernels: {k_str:<40}")

    print(f"\n✅ Total Trainable/Total Parameters: {total_trainable:,} / {total_params:,}")
    m_flops = total_flops.total() / 1e6 
    print(f"\n✅ Total FLOPS (M): {m_flops:.2f}M")

def count_params(model):
    print("Trainable parameters by layer:")   
    total_trainable = 0
    total =0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            count = param.numel()
            total_trainable += param.numel()
            print(name, count)
    print(f"\Trainable/Total parameters: {total_trainable}, {total}")
    

def plot_last_3_channels(stacked, cmap='viridis'):
    """
    Plots the [H, W] planes of the last 3 channels from a stacked tensor.

    Args:
        stacked (Tensor): Input tensor of shape [L*B, C, H, W].
    """
    sample = stacked[0]  # [C, H, W]
    C = sample.shape[0]

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()  # Flatten 2D array [2,5] -> [10]

    for ch in range(10):
        ax = axes[ch]
        img = sample[ch].cpu().detach().numpy()
        ax.imshow(img, cmap=cmap)
        ax.set_title(f'Channel {ch}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def cutmix(x, y, alpha=0.5):
    """
    Perform CutMix augmentation on a batch of images.
    :param x: The input batch of images (tensor)
    :param y: The labels corresponding to the batch (tensor)
    :param alpha: The hyperparameter for the Beta distribution to sample the cut region size
    :return: Mixed images and mixed labels, and lambda
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()  # Random permutation of indices

    # Get the cut region
    W = x.size(2)
    H = x.size(3)
    cut_rat = torch.sqrt(torch.tensor(1. - lam)).item()  # Calculate cut ratio
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    # Calculate the corners of the patch
    bbx1 = max(cx - cut_w // 2, 0)
    bby1 = max(cy - cut_h // 2, 0)
    bbx2 = min(cx + cut_w // 2, W)
    bby2 = min(cy + cut_h // 2, H)

    # CutMix
    x_cutmix = x.clone()
    x_cutmix[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    # Mix labels
    targets_a = y.clone()
    targets_b = y[index]
    return x_cutmix, targets_a, targets_b, lam
   