#!/usr/bin/env python3
"""
🔍 Simple Deepfake Detection Dataset Utilities

This module provides simple utilities for loading and processing local image datasets
for deepfake detection training and testing.

Features:
- 📂 Local folder-based dataset loading
- 🎯 Image preprocessing and validation
- 📊 Dataset statistics and visualization
- 🔍 Basic face detection validation

Author: Advanced Deepfake Detection Team
License: MIT
"""

# Standard Library
import os
import glob
import random
from typing import List, Tuple, Optional

# Third Party - Core
import numpy as np
import cv2
from PIL import Image

# Third Party - ML/DL
import torch
from torchvision import transforms

# Visualization
import matplotlib.pyplot as plt


class SimpleDatasetLoader:
    """Simple local dataset loader for deepfake detection"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Directory containing 'fake' and 'real' subdirectories
        """
        self.data_dir = data_dir
        self.fake_dir = os.path.join(data_dir, "fake")
        self.real_dir = os.path.join(data_dir, "real")
        
        # Create directories if they don't exist
        os.makedirs(self.fake_dir, exist_ok=True)
        os.makedirs(self.real_dir, exist_ok=True)
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"📁 Dataset directories:")
        print(f"   Fake: {self.fake_dir}")
        print(f"   Real: {self.real_dir}")
    
    def get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory"""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        files = []
        
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
            files.extend(glob.glob(os.path.join(directory, ext.upper())))
        
        return sorted(files)
    
    def load_dataset_stats(self) -> dict:
        """Load dataset statistics"""
        fake_files = self.get_image_files(self.fake_dir)
        real_files = self.get_image_files(self.real_dir)
        
        stats = {
            'fake_count': len(fake_files),
            'real_count': len(real_files),
            'total_count': len(fake_files) + len(real_files),
            'fake_files': fake_files[:10],  # First 10 for preview
            'real_files': real_files[:10]   # First 10 for preview
        }
        
        print(f"📊 Dataset Statistics:")
        print(f"   Fake images: {stats['fake_count']:,}")
        print(f"   Real images: {stats['real_count']:,}")
        print(f"   Total images: {stats['total_count']:,}")
        
        if stats['total_count'] == 0:
            print(f"⚠️  No images found in {self.data_dir}")
            print(f"   Please add images to:")
            print(f"   - {self.fake_dir} (for fake/manipulated images)")
            print(f"   - {self.real_dir} (for real/authentic images)")
        
        return stats
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """Load and preprocess an image"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            if target_size:
                img = cv2.resize(img, target_size)
            
            return img
            
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def create_data_loaders(self, batch_size: int = 32, test_split: float = 0.2, random_state: int = 42):
        """Create PyTorch data loaders for training and testing"""
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
        
        # Get all image files
        fake_files = self.get_image_files(self.fake_dir)
        real_files = self.get_image_files(self.real_dir)
        
        # Create labels
        fake_labels = [0] * len(fake_files)  # 0 for fake
        real_labels = [1] * len(real_files)  # 1 for real
        
        # Combine
        all_files = fake_files + real_files
        all_labels = fake_labels + real_labels
        
        if len(all_files) == 0:
            print("⚠️  No images found for training!")
            return None, None
        
        # Split into train/test
        train_files, test_files, train_labels, test_labels = train_test_split(
            all_files, all_labels, test_size=test_split, 
            random_state=random_state, stratify=all_labels
        )
        
        print(f"🎯 Data split:")
        print(f"   Training: {len(train_files)} images")
        print(f"   Testing: {len(test_files)} images")
        
        # Create datasets
        train_dataset = ImageDataset(train_files, train_labels, self.transform)
        test_dataset = ImageDataset(test_files, test_labels, self.transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def visualize_samples(self, num_samples: int = 8, save_path: Optional[str] = None):
        """Visualize sample images from the dataset"""
        fake_files = self.get_image_files(self.fake_dir)
        real_files = self.get_image_files(self.real_dir)
        
        # Sample files
        fake_samples = random.sample(fake_files, min(num_samples//2, len(fake_files)))
        real_samples = random.sample(real_files, min(num_samples//2, len(real_files)))
        
        # Create plot
        fig, axes = plt.subplots(2, max(len(fake_samples), len(real_samples)), 
                                figsize=(15, 6))
        
        if len(fake_samples) == 0 and len(real_samples) == 0:
            plt.text(0.5, 0.5, 'No images found', ha='center', va='center')
            plt.show()
            return
        
        # Plot fake samples
        for i, file_path in enumerate(fake_samples):
            img = self.load_image(file_path)
            if img is not None:
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Fake {i+1}', color='red')
                axes[0, i].axis('off')
        
        # Plot real samples
        for i, file_path in enumerate(real_samples):
            img = self.load_image(file_path)
            if img is not None:
                axes[1, i].imshow(img)
                axes[1, i].set_title(f'Real {i+1}', color='green')
                axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📸 Visualization saved to {save_path}")
        
        plt.show()


class ImageDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load with OpenCV and convert to PIL
        img = cv2.imread(image_path)
        if img is None:
            # Return a dummy image if loading fails
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, label


def main():
    """Demo function to test the dataset loader"""
    print("🔍 Simple Dataset Loader Demo")
    
    # Initialize loader
    loader = SimpleDatasetLoader("data")
    
    # Load statistics
    stats = loader.load_dataset_stats()
    
    if stats['total_count'] > 0:
        # Visualize samples
        loader.visualize_samples(num_samples=8)
        
        # Create data loaders
        train_loader, test_loader = loader.create_data_loaders(batch_size=16)
        
        if train_loader:
            print(f"✅ Data loaders created successfully!")
            print(f"   Training batches: {len(train_loader)}")
            print(f"   Testing batches: {len(test_loader)}")
    else:
        print("📝 To use this dataset loader:")
        print("   1. Create 'data/fake' and 'data/real' directories")
        print("   2. Add fake/manipulated images to 'data/fake'")
        print("   3. Add real/authentic images to 'data/real'")
        print("   4. Run this script again")


if __name__ == "__main__":
    main()
