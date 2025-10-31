import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class ValDataset(Dataset):
    def __init__(self, val_dir, txt_file, transform=None):
        """
        val_dir: path to the folder containing all validation images
        txt_file: path to val.txt containing "image_name class_index"
        transform: optional torchvision transforms
        """
        self.val_dir = val_dir
        self.transform = transform
        self.samples = []

        with open(txt_file, "r") as f:
            for line in f:
                img_name, label = line.strip().split()
                img_path = os.path.join(val_dir, os.path.basename(img_name))
                self.samples.append((img_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_100places_dataloaders(data_root, batch_size):
    train_dir = data_root+"/images/train"
    test_dir = data_root+"/images/test"

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

    
    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ValDataset(
        val_dir = data_root+"/images/val",
        txt_file = data_root+"/images/val.txt",
        transform=test_transform
    )
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  
    return train_loader, test_loader


if __name__ == "__main__":

    train_loader, test_loader = get_100places_dataloaders(data_root='data/100places', batch_size=256)
    
    print("train/test size",len(train_loader)*256, len(test_loader)*256)
    # Example: iterate over one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [64, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")  # Expect [64]    