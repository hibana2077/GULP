import os
import requests
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import io
from typing import Tuple, Dict, Any


class DownloadManager:
    """Download and manage datasets based on dataset.md specifications"""
    
    @staticmethod
    def download_file(url: str, save_path: str, chunk_size: int = 8192) -> None:
        """Download file from URL to save_path"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        print(f"\nDownloaded: {save_path}")
    
    @staticmethod
    def download_cifar10(data_dir: str = "./data") -> str:
        """Download CIFAR-10 parquet file"""
        url = "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/cifar10.parquet?download=true"
        save_path = os.path.join(data_dir, "cifar10.parquet")
        
        if not os.path.exists(save_path):
            print("Downloading CIFAR-10...")
            DownloadManager.download_file(url, save_path)
        else:
            print(f"CIFAR-10 already exists: {save_path}")
        
        return save_path
    
    @staticmethod
    def download_cifar100(data_dir: str = "./data") -> str:
        """Download CIFAR-100 parquet file"""
        url = "https://huggingface.co/datasets/hibana2077/CV-dataset-all-in-parquet/resolve/main/datasets/cifar100.parquet?download=true"
        save_path = os.path.join(data_dir, "cifar100.parquet")
        
        if not os.path.exists(save_path):
            print("Downloading CIFAR-100...")
            DownloadManager.download_file(url, save_path)
        else:
            print(f"CIFAR-100 already exists: {save_path}")
        
        return save_path
    
    @staticmethod
    def download_iwslt14(data_dir: str = "./data") -> str:
        """Download IWSLT14 De-En dataset"""
        url = "https://huggingface.co/datasets/bbaaaa/iwslt14-de-en/resolve/main/data/de-en.zip?download=true"
        zip_path = os.path.join(data_dir, "iwslt14-de-en.zip")
        extract_path = os.path.join(data_dir, "iwslt14-de-en")
        
        if not os.path.exists(extract_path):
            if not os.path.exists(zip_path):
                print("Downloading IWSLT14 De-En...")
                DownloadManager.download_file(url, zip_path)
            
            print("Extracting IWSLT14 De-En...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
        else:
            print(f"IWSLT14 De-En already exists: {extract_path}")
        
        return extract_path


class CifarDataset(Dataset):
    """CIFAR-10/100 Dataset from parquet files"""
    
    def __init__(self, parquet_path: str, split: str = "train", transform=None):
        """
        Args:
            parquet_path: Path to the parquet file
            split: 'train' or 'test'
            transform: Optional transform to be applied on a sample
        """
        self.df = pd.read_parquet(parquet_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
        # Create label mapping
        self.classes = sorted(self.df['class_name'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        print(f"Loaded {len(self.df)} {split} samples with {self.num_classes} classes")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Convert bytes to PIL Image
        image_bytes = row['image']
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Get label
        label = self.class_to_idx[row['class_name']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_cifar_transforms(dataset_name: str, is_train: bool = True) -> transforms.Compose:
    """Get appropriate transforms for CIFAR datasets"""
    
    if dataset_name.lower() == "cifar10":
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    elif dataset_name.lower() == "cifar100":
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )
    else:
        # Default normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloader(dataset_name: str, data_dir: str = "./data", batch_size: int = 128, 
                  num_workers: int = 4, download: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    Get train and test dataloaders for specified dataset
    
    Returns:
        train_loader, test_loader, num_classes
    """
    
    if dataset_name.lower() == "cifar10":
        if download:
            parquet_path = DownloadManager.download_cifar10(data_dir)
        else:
            parquet_path = os.path.join(data_dir, "cifar10.parquet")
        
        train_transform = get_cifar_transforms("cifar10", is_train=True)
        test_transform = get_cifar_transforms("cifar10", is_train=False)
        
        train_dataset = CifarDataset(parquet_path, split="train", transform=train_transform)
        test_dataset = CifarDataset(parquet_path, split="test", transform=test_transform)
        
    elif dataset_name.lower() == "cifar100":
        if download:
            parquet_path = DownloadManager.download_cifar100(data_dir)
        else:
            parquet_path = os.path.join(data_dir, "cifar100.parquet")
        
        train_transform = get_cifar_transforms("cifar100", is_train=True)
        test_transform = get_cifar_transforms("cifar100", is_train=False)
        
        train_dataset = CifarDataset(parquet_path, split="train", transform=train_transform)
        test_dataset = CifarDataset(parquet_path, split="test", transform=test_transform)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset.num_classes


if __name__ == "__main__":
    # Test the dataset
    train_loader, test_loader, num_classes = get_dataloader("cifar10", batch_size=32)
    
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Label shape: {labels.shape}")
        print(f"Label range: {labels.min().item()} - {labels.max().item()}")
        break
