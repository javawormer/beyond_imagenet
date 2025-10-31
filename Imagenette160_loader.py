import os
import tarfile
import urllib.request
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#training 9469, testing 3925, total 13394. 10 classes.

def download_and_extract_imagenette(dest_folder='data/imagenette'):
    url = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_folder, filename)

    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    if not os.path.exists(filepath):
        print(f'Downloading {url}...')
        urllib.request.urlretrieve(url, filepath)
        print('Download complete.')

    print('Extracting files...')
    with tarfile.open(filepath, 'r:gz') as tar:
        tar.extractall(path=dest_folder)
    print('Extraction complete.')

# ==========================
# 2. Main Loader Function
# ==========================

def get_dataloaders(data_root, batch_size, cache_dir="cache", train_ratio=0.8):
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(160, scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.3, 0.3, 0.2),
        transforms.RandAugment(num_ops=2, magnitude=10),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(192),
        transforms.CenterCrop(160),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=f'{data_root}/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root=f'{data_root}/val', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)
    print("loading training and testing datasets")


    return train_loader, val_loader



if __name__ == "__main__":
    #download_and_extract_imagenette()
    train_loader, test_loader = get_dataloaders(data_root="data/imagenette/imagenette2-160", batch_size=256)
    
    # Example: iterate over one batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # Expect [256, 3, 224, 224]
    print(f"Batch labels shape: {len(train_loader)*256}")  # Expect [64]
    