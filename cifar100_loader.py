import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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

def get_cifar100_dataloaders(data_root, batch_size):

    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    # Define transforms (normalize & convert to tensors)
    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4),
        #transforms.RandomResizedCrop(32, scale=(0.9, 1.1), ratio=(1.0, 1.0)), #less performant
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(1.0, 1.0)), #less performant

        transforms.RandomHorizontalFlip(p=0.5),        
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std),  # Mean and std for CIFAR-100
        transforms.RandomErasing(p=0.25)
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(224),         # Resize shorter edge to 224
        transforms.CenterCrop(224),     # Center crop to 224x224

        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    
    # Load CIFAR-100 training set
    data_root = '../cifar-100-data/'

    cifar100_dataset = datasets.CIFAR100(root=data_root, train=True, download=False, transform=train_transform)    
    train_loader = torch.utils.data.DataLoader(cifar100_dataset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR100(root=data_root, train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, val_loader = get_cifar100_dataloaders(data_root='../cifar-100-data/', batch_size=256)
    
    # Example: iterate over one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [64, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")  # Expect [64]    