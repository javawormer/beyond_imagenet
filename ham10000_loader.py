import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_loader import *

# Download the HAM10000 dataset
#kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
    
import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.classes = sorted(self.df['dx'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.class_to_idx[self.df.iloc[idx]['dx']]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        return image, label


def load_ham10000_dataset(root_dir, test_size=0.15, val_size=0.15, seed=42):
    # Metadata
    metadata_path = os.path.join(root_dir, "HAM10000_metadata.csv")
    df = pd.read_csv(metadata_path)

    # Combine image paths
    img_dir1 = os.path.join(root_dir, "HAM10000_images_part_1")
    img_dir2 = os.path.join(root_dir, "HAM10000_images_part_2")

    def get_path(img_id):
        fname = img_id + ".jpg"
        if os.path.exists(os.path.join(img_dir1, fname)):
            return os.path.join(img_dir1, fname)
        else:
            return os.path.join(img_dir2, fname)

    df['image_path'] = df['image_id'].apply(get_path)

    # First split: Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['dx'], random_state=seed
    )

    # Second split: Train vs Val
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df['dx'], random_state=seed
    )

    return train_df, val_df, test_df


def create_ham10000_loaders(data_root, batch_size=32, img_size=224):
    train_df, val_df, test_df = load_ham10000_dataset(data_root)
    
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = HAM10000Dataset(train_df, transform=transform_train)
    val_dataset = HAM10000Dataset(val_df, transform=transform_eval)
    test_dataset = HAM10000Dataset(test_df, transform=transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":

    #train_loader, val_loader = get_tiny_imagenet_200_dataloaders(data_root='tiny-imagenet-200/', batch_size=10)
    # ==== Example usage ====
    """
    root_dir = "data/ham10000"  
    train_df, val_df, test_df = load_ham10000_dataset(root_dir)
    
    print("Train size:", len(train_df))
    print("Val size:", len(val_df))
    print("Test size:", len(test_df))
    print("Class distribution in train:\n", train_df['dx'].value_counts())
    """
    train_loader, val_loader, test_loader = create_ham10000_loaders("data/ham10000")
    
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [64, 3, 64, 64]
    print(f"Batch labels shape: {labels.shape}")  # Expect [64]  