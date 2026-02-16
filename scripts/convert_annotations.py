#!/usr/bin/env python3
"""
Convert Lung-PET-CT-Dx dataset annotations from VOC to YOLO format.

This script processes DICOM images and PASCAL VOC annotations,
converting them to YOLO format for training.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Usage:
    python scripts/convert_annotations.py \
        --input_dir data/raw \
        --output_dir data/processed \
        --img_size 640
"""

import argparse
import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM support disabled.")


# Class mapping
CLASS_MAPPING = {
    'A': 0,  # Adenocarcinoma
    'B': 1,  # Small Cell
    'E': 2,  # Large Cell
    'G': 3,  # Squamous Cell
}

CLASS_NAMES = ['Adenocarcinoma', 'Small_Cell', 'Large_Cell', 'Squamous_Cell']


def parse_voc_annotation(xml_path: str) -> List[Dict]:
    """
    Parse PASCAL VOC format annotation.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of annotation dictionaries
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image size
    size = root.find('size')
    img_width = int(size.find('width').text)
    img_height = int(size.find('height').text)
    
    annotations = []
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        
        # Skip unknown classes
        if name not in CLASS_MAPPING:
            continue
        
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        
        annotations.append({
            'class': name,
            'class_id': CLASS_MAPPING[name],
            'bbox': [xmin, ymin, xmax, ymax],
            'img_size': (img_width, img_height),
        })
    
    return annotations


def voc_to_yolo(annotations: List[Dict], img_width: int, img_height: int) -> List[str]:
    """
    Convert VOC annotations to YOLO format.
    
    YOLO format: class_id x_center y_center width height (normalized)
    
    Args:
        annotations: List of annotation dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        List of YOLO format strings
    """
    yolo_lines = []
    
    for ann in annotations:
        xmin, ymin, xmax, ymax = ann['bbox']
        
        # Calculate center and size (normalized)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        
        # Clip to [0, 1]
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_lines.append(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    return yolo_lines


def convert_dicom_to_png(
    dicom_path: str,
    output_path: str,
    img_size: int = 640,
    window_center: float = -600,
    window_width: float = 1500,
) -> bool:
    """
    Convert DICOM file to PNG.
    
    Args:
        dicom_path: Path to DICOM file
        output_path: Output PNG path
        img_size: Target image size
        window_center: CT window center (HU)
        window_width: CT window width (HU)
        
    Returns:
        True if successful
    """
    if not PYDICOM_AVAILABLE:
        print("Error: pydicom required for DICOM conversion")
        return False
    
    try:
        # Read DICOM
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array.astype(np.float32)
        
        # Apply rescale slope and intercept
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Apply CT windowing
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        pixel_array = np.clip(pixel_array, lower, upper)
        
        # Normalize to [0, 255]
        pixel_array = ((pixel_array - lower) / window_width * 255).astype(np.uint8)
        
        # Convert to RGB
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        
        # Resize
        pixel_array = cv2.resize(pixel_array, (img_size, img_size))
        
        # Save
        cv2.imwrite(output_path, cv2.cvtColor(pixel_array, cv2.COLOR_RGB2BGR))
        
        return True
        
    except Exception as e:
        print(f"Error converting {dicom_path}: {e}")
        return False


def split_dataset(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test at patient level.
    
    Args:
        samples: List of sample dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    # Group by patient
    patient_samples = {}
    for sample in samples:
        patient_id = sample.get('patient_id', 'unknown')
        if patient_id not in patient_samples:
            patient_samples[patient_id] = []
        patient_samples[patient_id].append(sample)
    
    # Shuffle patients
    random.seed(seed)
    patients = list(patient_samples.keys())
    random.shuffle(patients)
    
    # Calculate split indices
    n_patients = len(patients)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train+n_val]
    test_patients = patients[n_train+n_val:]
    
    # Collect samples
    train_samples = [s for p in train_patients for s in patient_samples[p]]
    val_samples = [s for p in val_patients for s in patient_samples[p]]
    test_samples = [s for p in test_patients for s in patient_samples[p]]
    
    return train_samples, val_samples, test_samples


def process_dataset(
    input_dir: str,
    output_dir: str,
    img_size: int = 640,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Process the entire dataset.
    
    Args:
        input_dir: Input directory containing raw data
        output_dir: Output directory for processed data
        img_size: Target image size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        seed: Random seed
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Find all annotation files
    xml_files = list(input_dir.glob('**/*.xml'))
    print(f"Found {len(xml_files)} annotation files")
    
    samples = []
    
    for xml_path in xml_files:
        # Parse annotation
        annotations = parse_voc_annotation(str(xml_path))
        
        if not annotations:
            continue
        
        # Find corresponding image
        img_name = xml_path.stem
        patient_id = xml_path.parent.name
        
        # Look for DICOM or PNG
        possible_images = list(xml_path.parent.glob(f'{img_name}.*'))
        img_path = None
        
        for p in possible_images:
            if p.suffix.lower() in ['.dcm', '.png', '.jpg', '.jpeg']:
                img_path = p
                break
        
        if img_path is None:
            continue
        
        samples.append({
            'img_path': str(img_path),
            'xml_path': str(xml_path),
            'annotations': annotations,
            'patient_id': patient_id,
            'img_name': img_name,
        })
    
    print(f"Found {len(samples)} valid samples")
    
    # Split dataset
    train_samples, val_samples, test_samples = split_dataset(
        samples, train_ratio, val_ratio, seed
    )
    
    print(f"Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    # Process each split
    splits = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples,
    }
    
    for split_name, split_samples in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for i, sample in enumerate(split_samples):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(split_samples)}")
            
            # Output paths
            out_img_path = output_dir / 'images' / split_name / f"{sample['img_name']}.png"
            out_label_path = output_dir / 'labels' / split_name / f"{sample['img_name']}.txt"
            
            # Convert/copy image
            img_path = Path(sample['img_path'])
            
            if img_path.suffix.lower() == '.dcm':
                success = convert_dicom_to_png(
                    str(img_path), str(out_img_path), img_size
                )
                if not success:
                    continue
            else:
                # Copy and resize
                img = cv2.imread(str(img_path))
                img = cv2.resize(img, (img_size, img_size))
                cv2.imwrite(str(out_img_path), img)
            
            # Convert annotations
            ann = sample['annotations'][0]
            yolo_lines = voc_to_yolo(sample['annotations'], img_size, img_size)
            
            # Save labels
            with open(out_label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
    
    # Print summary
    print("\n" + "=" * 50)
    print("CONVERSION COMPLETE")
    print("=" * 50)
    
    for split_name in ['train', 'val', 'test']:
        n_images = len(list((output_dir / 'images' / split_name).glob('*.png')))
        n_labels = len(list((output_dir / 'labels' / split_name).glob('*.txt')))
        print(f"{split_name}: {n_images} images, {n_labels} labels")
    
    print(f"\nOutput: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert Lung-PET-CT-Dx annotations to YOLO format'
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Target image size')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
