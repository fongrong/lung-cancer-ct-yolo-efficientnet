"""
Dataset module for Lung Cancer CT Classification.

This module provides data loading, preprocessing, and augmentation
utilities for the Lung-PET-CT-Dx dataset.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image

# Class definitions
CLASS_NAMES = ["Adenocarcinoma", "Small_Cell", "Large_Cell", "Squamous_Cell"]
CLASS_MAPPING = {'A': 0, 'B': 1, 'E': 2, 'G': 3}
NUM_CLASSES = 4


class LungCancerDataset(Dataset):
    """
    Custom PyTorch Dataset for Lung-PET-CT-Dx data.
    
    Handles loading of CT images with annotations in YOLO format.
    Supports CLAHE preprocessing and various augmentation strategies.
    
    Args:
        data_dir: Root directory containing images and labels
        split: One of 'train', 'val', or 'test'
        img_size: Target image size (default: 640)
        augment: Whether to apply augmentation
        clahe: Whether to apply CLAHE preprocessing
        clahe_clip_limit: CLAHE clip limit (default: 2.0)
        clahe_tile_size: CLAHE tile grid size (default: (8, 8))
        cache: Whether to cache images in memory
        
    Example:
        >>> dataset = LungCancerDataset(
        ...     data_dir="./data/lung-pet-ct-dx",
        ...     split="train",
        ...     img_size=640,
        ...     augment=True
        ... )
        >>> image, labels = dataset[0]
        >>> print(f"Image shape: {image.shape}, Labels: {labels.shape}")
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        img_size: int = 640,
        augment: Optional[bool] = None,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        cache: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment if augment is not None else (split == 'train')
        self.clahe_enabled = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.cache = cache
        self.cached_images = {}
        
        # Initialize CLAHE
        if self.clahe_enabled:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_size
            )
        
        # Load samples
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Dict]:
        """Load all image paths and their annotations."""
        samples = []
        
        img_dir = self.data_dir / 'images' / self.split
        label_dir = self.data_dir / 'labels' / self.split
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(glob.glob(str(img_dir / ext)))
        
        for img_path in sorted(image_files):
            img_path = Path(img_path)
            label_path = label_dir / (img_path.stem + '.txt')
            
            # Load labels
            labels = []
            if label_path.exists():
                labels = self._load_yolo_labels(label_path)
            
            if labels:
                samples.append({
                    'image_path': str(img_path),
                    'label_path': str(label_path),
                    'labels': labels,
                    'class': labels[0][0] if labels else 0
                })
        
        return samples
    
    def _load_yolo_labels(self, label_path: Path) -> List[List[float]]:
        """
        Load labels in YOLO format.
        
        Format: class x_center y_center width height
        All values normalized to [0, 1]
        """
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    labels.append([cls, x_center, y_center, width, height])
        return labels
    
    def _print_class_distribution(self):
        """Print class distribution of the dataset."""
        class_counts = {name: 0 for name in CLASS_NAMES}
        for sample in self.samples:
            cls_idx = sample['class']
            if 0 <= cls_idx < NUM_CLASSES:
                class_counts[CLASS_NAMES[cls_idx]] += 1
        
        print(f"\nClass distribution ({self.split}):")
        total = len(self.samples)
        for name, count in class_counts.items():
            pct = 100 * count / total if total > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            image: Tensor of shape (3, H, W), normalized to [0, 1]
            labels: Tensor of shape (N, 5) with [class, x, y, w, h]
        """
        sample = self.samples[idx]
        
        # Load image (from cache if available)
        if self.cache and idx in self.cached_images:
            image = self.cached_images[idx].copy()
        else:
            image = self._load_image(sample['image_path'])
            if self.cache:
                self.cached_images[idx] = image.copy()
        
        labels = np.array(sample['labels'], dtype=np.float32)
        
        # Apply CLAHE preprocessing
        if self.clahe_enabled:
            image = self._apply_clahe(image)
        
        # Resize with letterboxing
        image, labels = self._resize(image, labels)
        
        # Apply augmentation
        if self.augment:
            image, labels = self._augment(image, labels)
        
        # Convert to tensor
        image = self._to_tensor(image)
        labels = torch.from_numpy(labels) if len(labels) > 0 else torch.zeros((0, 5))
        
        return image, labels
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image from path."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to improve contrast."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return image
    
    def _resize(
        self, 
        image: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Resize image with letterboxing to preserve aspect ratio."""
        h, w = image.shape[:2]
        
        # Calculate scale
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = image
        
        # Adjust labels
        if len(labels) > 0:
            labels[:, 1] = (labels[:, 1] * new_w + pad_w) / self.img_size
            labels[:, 2] = (labels[:, 2] * new_h + pad_h) / self.img_size
            labels[:, 3] = labels[:, 3] * new_w / self.img_size
            labels[:, 4] = labels[:, 4] * new_h / self.img_size
        
        return padded, labels
    
    def _augment(
        self, 
        image: np.ndarray, 
        labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation."""
        # Horizontal flip
        if random.random() < 0.5:
            image = np.fliplr(image).copy()
            if len(labels) > 0:
                labels[:, 1] = 1 - labels[:, 1]
        
        # Brightness/contrast adjustment
        if random.random() < 0.5:
            alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
            beta = -20 + random.random() * 40     # -20 to 20
            image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
        
        # HSV augmentation
        if random.random() < 0.5:
            image = self._hsv_augment(image)
        
        return image, labels
    
    def _hsv_augment(
        self, 
        image: np.ndarray,
        h_gain: float = 0.015,
        s_gain: float = 0.7,
        v_gain: float = 0.4
    ) -> np.ndarray:
        """Apply HSV color augmentation."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] * r[0], 0, 180)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 255)
        
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        # HWC to CHW
        image = np.transpose(image, (2, 0, 1))
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image)


def collate_fn(batch):
    """
    Custom collate function to handle variable-length labels.
    
    Args:
        batch: List of (image, labels) tuples
        
    Returns:
        images: Tensor of shape (B, 3, H, W)
        labels: List of label tensors
    """
    images, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(labels)


def create_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    img_size: int = 640,
    workers: int = 8,
    augment: bool = True,
    clahe: bool = True,
    cache: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory of the dataset
        batch_size: Batch size
        img_size: Target image size
        workers: Number of data loading workers
        augment: Whether to apply augmentation to training data
        clahe: Whether to apply CLAHE preprocessing
        cache: Whether to cache images in memory
        
    Returns:
        train_loader, val_loader, test_loader
        
    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     data_dir="./data/lung-pet-ct-dx",
        ...     batch_size=16,
        ...     img_size=640
        ... )
    """
    # Create datasets
    train_dataset = LungCancerDataset(
        data_dir, split='train', img_size=img_size, 
        augment=augment, clahe=clahe, cache=cache
    )
    val_dataset = LungCancerDataset(
        data_dir, split='val', img_size=img_size, 
        augment=False, clahe=clahe, cache=cache
    )
    test_dataset = LungCancerDataset(
        data_dir, split='test', img_size=img_size, 
        augment=False, clahe=clahe, cache=cache
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    print("Testing LungCancerDataset...")
    
    dataset = LungCancerDataset(
        data_dir="./data/lung-pet-ct-dx",
        split="train",
        img_size=640,
        augment=True,
        clahe=True
    )
    
    if len(dataset) > 0:
        image, labels = dataset[0]
        print(f"\nSample:")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"  Labels shape: {labels.shape}")
        if len(labels) > 0:
            print(f"  First label: {labels[0]}")
    
    print("\nDataset test complete!")
