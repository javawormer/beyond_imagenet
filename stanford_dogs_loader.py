import torch
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import json
import shutil

# ==========================
# 1. Create physical train/test folder split
# ==========================

def create_train_test_folders(data_tensor, label_tensor, train_indices, test_indices, class_names, cache_dir="cache"):
    """
    Save train/test images into separate folders by class.
    Undo normalization correctly.
    """
    train_dir = Path(cache_dir) / "train_fixed"
    test_dir = Path(cache_dir) / "test_fixed"

    if train_dir.exists() and test_dir.exists():
        print("✅ Using fixed folders for training/testing.", train_dir, test_dir)
        return

    print("📂 Creating train/test folder structure...")
    for d in [train_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Correct undo normalization: make sure tensor is CPU float
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3,1,1)

    def save_images(indices, out_dir):
        for idx in indices:
            cls = class_names[label_tensor[idx].item()]
            cls_folder = out_dir / cls
            cls_folder.mkdir(parents=True, exist_ok=True)

            # Move tensor to CPU and float
            img_tensor = data_tensor[idx].detach().cpu().float()

            # Undo normalization
            img_tensor = img_tensor * std + mean
            img_tensor = img_tensor.clamp(0,1)

            # Convert to PIL
            img = transforms.ToPILImage()(img_tensor)
            img.save(cls_folder / f"{idx}.jpg")

    save_images(train_indices, train_dir)
    save_images(test_indices, test_dir)
    print("✅ Saved train/test images into folders!")


# ==========================
# 2. Main Loader Function
# ==========================

def get_dataloaders(data_root, batch_size, cache_dir="cache", train_ratio=0.8, drop_last=True):
    os.makedirs(cache_dir, exist_ok=True)

    data_path = os.path.join(cache_dir, "images.pt")
    label_path = os.path.join(cache_dir, "labels.pt")
    split_path = os.path.join(cache_dir, "split_indices.json")

    # --------------------------
    # Load or cache tensors
    # --------------------------
    if os.path.exists(data_path) and os.path.exists(label_path):
        print("✅ Loading cached tensors...")
        data_tensor = torch.load(data_path)
        label_tensor = torch.load(label_path)
        raw_dataset = datasets.ImageFolder(root=data_root)  # for class names
    else:
        print("🔄 Caching base ToTensor+Normalize...")

        preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        raw_dataset = datasets.ImageFolder(root=data_root)
        images = []
        labels = []

        for img, label in raw_dataset:
            img_tensor = preprocess(img)
            images.append(img_tensor)
            labels.append(label)

        data_tensor = torch.stack(images)
        label_tensor = torch.tensor(labels)

        torch.save(data_tensor, data_path)
        torch.save(label_tensor, label_path)
        print("✅ Saved tensors to cache!")

    # --------------------------
    # Fixed train/test split
    # --------------------------
    N = len(data_tensor)
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split = json.load(f)
        train_indices = split["train"]
        test_indices = split["test"]
        print("✅ Loaded saved train/test split.")
    else:
        all_indices = torch.randperm(N).tolist()
        train_len = int(train_ratio * N)
        train_indices = all_indices[:train_len]
        test_indices = all_indices[train_len:]
        with open(split_path, "w") as f:
            json.dump({"train": train_indices, "test": test_indices}, f)
        print("🔒 Saved new train/test split.")

    # --------------------------
    # Create physical folders
    # --------------------------
    create_train_test_folders(data_tensor, label_tensor, train_indices, test_indices,
                              raw_dataset.classes, cache_dir=cache_dir)

    # --------------------------
    # Define transforms
    # --------------------------
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_aug = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --------------------------
    # Load datasets via ImageFolder
    # --------------------------
    train_folder = os.path.join(cache_dir, "train_fixed")
    test_folder = os.path.join(cache_dir, "test_fixed")

    train_dataset = datasets.ImageFolder(root=train_folder, transform=train_aug)
    test_dataset = datasets.ImageFolder(root=test_folder, transform=test_aug)

    # --------------------------
    # Build dataloaders
    # --------------------------
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(
        data_root="data/dogs/images/Images",
        batch_size=256,
        cache_dir="cache",
        drop_last=True
    )

    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [256,3,224,224]
    print(f"Batch labels shape: {labels.shape}")  # Expect [256]
