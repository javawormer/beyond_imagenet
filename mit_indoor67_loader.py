import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_indoor67_dataloaders(data_root, batch_size):
    train_dir = data_root+"/train"
    test_dir = data_root+"/test"

    norm_mean = (0.485, 0.456, 0.406)
    norm_std = (0.229, 0.224, 0.22)

    train_transform = transforms.Compose([
        #transforms.RandomCrop(32, padding=4), #better
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(1.0, 1.0)), #less performant

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
        
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(224),         # Resize shorter edge to 224
        transforms.CenterCrop(224),     # Center crop to 224x224

        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 2. Load CIFAR-10 train and test datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
       
    # 3. Create DataLoaders for the datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   
    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = get_indoor67_dataloaders(data_root='data/MIT_Indoor_67', batch_size=256)
    
    # Example: iterate over one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [64, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")  # Expect [64]    