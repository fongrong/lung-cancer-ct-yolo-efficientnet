#!/usr/bin/env python3
"""
Training script for Lung Cancer CT Classification.

Supports training of hybrid YOLO-EfficientNet models with configurable
hyperparameters and learning rate schedules.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Usage:
    # Train with default settings (YOLOv8m + EfficientNetV2-M, medium LR)
    python src/train.py --data configs/data.yaml
    
    # Train specific model
    python src/train.py --model yolov8m_effnet --lr0 0.003 --epochs 100
    
    # Train with config file
    python src/train.py --config configs/train_config.yaml
    
    # Train all models
    bash scripts/train_all_models.sh
"""

import argparse
import os
import sys
import yaml
import random
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.tensorboard import SummaryWriter

from ultralytics import YOLO

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import create_dataloaders, CLASS_NAMES
from src.models import create_model, count_parameters


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_lr_config(lr_setting: str) -> Dict:
    """Get learning rate configuration from preset."""
    presets = {
        'high': {'lr0': 0.01, 'lrf': 0.1},
        'medium': {'lr0': 0.003, 'lrf': 0.01},
        'low': {'lr0': 0.001, 'lrf': 0.005},
    }
    return presets.get(lr_setting, presets['medium'])


def train_yolo_native(
    model_name: str,
    data_yaml: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    lr0: float = 0.003,
    lrf: float = 0.01,
    device: str = '0',
    project: str = 'runs/train',
    name: Optional[str] = None,
    seed: int = 42,
    workers: int = 8,
    patience: int = 50,
    resume: bool = False,
    **kwargs
) -> Dict:
    """
    Train YOLO model using native Ultralytics API.
    
    This is the primary training method for standard YOLO models.
    For hybrid models, see train_hybrid().
    
    Args:
        model_name: Base model ('yolov8m' or 'yolov9m')
        data_yaml: Path to data configuration
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        lr0: Initial learning rate
        lrf: Final learning rate factor
        device: GPU device ID
        project: Project directory
        name: Experiment name
        seed: Random seed
        workers: Data loading workers
        patience: Early stopping patience
        resume: Resume from checkpoint
        
    Returns:
        Training results dictionary
    """
    set_seed(seed)
    
    # Generate experiment name
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_lr{lr0}_{timestamp}"
    
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Image size:    {img_size}")
    print(f"  Learning rate: {lr0} -> {lr0 * lrf}")
    print(f"  Device:        {device}")
    print(f"  Save to:       {project}/{name}")
    print("=" * 70 + "\n")
    
    # Determine base weights
    if 'yolov9' in model_name:
        base_weights = 'yolov9m.pt'
    else:
        base_weights = 'yolov8m.pt'
    
    # Initialize model
    model = YOLO(base_weights)
    
    # Start training
    start_time = time.time()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr0,
        lrf=lrf,
        device=device,
        project=project,
        name=name,
        seed=seed,
        workers=workers,
        patience=patience,
        resume=resume,
        # Optimizer settings
        optimizer='SGD',
        momentum=0.937,
        weight_decay=0.0005,
        # Warmup
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Loss weights
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Other
        plots=True,
        save=True,
        save_period=-1,
        val=True,
        amp=True,
        **kwargs
    )
    
    training_time = time.time() - start_time
    
    # Compile results
    training_results = {
        'model': model_name,
        'epochs': epochs,
        'lr0': lr0,
        'lrf': lrf,
        'training_time_hours': training_time / 3600,
        'best_fitness': results.results_dict.get('fitness', 0),
        'metrics': {
            'precision': results.results_dict.get('metrics/precision(B)', 0),
            'recall': results.results_dict.get('metrics/recall(B)', 0),
            'mAP50': results.results_dict.get('metrics/mAP50(B)', 0),
            'mAP50-95': results.results_dict.get('metrics/mAP50-95(B)', 0),
        },
        'save_dir': str(results.save_dir),
    }
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Time:       {training_time/3600:.2f} hours")
    print(f"  Precision:  {training_results['metrics']['precision']:.4f}")
    print(f"  Recall:     {training_results['metrics']['recall']:.4f}")
    print(f"  mAP@50:     {training_results['metrics']['mAP50']:.4f}")
    print(f"  mAP@50-95:  {training_results['metrics']['mAP50-95']:.4f}")
    print(f"  Saved to:   {training_results['save_dir']}")
    print("=" * 70 + "\n")
    
    return training_results


def train_hybrid(
    model_name: str,
    data_dir: str,
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    lr0: float = 0.003,
    lrf: float = 0.01,
    device: str = 'cuda:0',
    project: str = 'runs/train',
    name: Optional[str] = None,
    seed: int = 42,
    workers: int = 8,
    patience: int = 50,
    **kwargs
) -> Dict:
    """
    Train hybrid YOLO-EfficientNet model with custom training loop.
    
    This method is used for models that integrate EfficientNet backbone
    with YOLO detection components.
    
    Args:
        model_name: Model name (e.g., 'yolov8m_effnet')
        data_dir: Path to data directory
        epochs: Number of epochs
        batch_size: Batch size
        img_size: Image size
        lr0: Initial learning rate
        lrf: Final learning rate factor
        device: Device string
        project: Project directory
        name: Experiment name
        seed: Random seed
        workers: Data loading workers
        patience: Early stopping patience
        
    Returns:
        Training results dictionary
    """
    set_seed(seed)
    
    # Setup directories
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{model_name}_lr{lr0}_{timestamp}"
    
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=str(save_dir / 'tensorboard'))
    
    print("\n" + "=" * 70)
    print(f"TRAINING HYBRID MODEL: {model_name}")
    print("=" * 70)
    print(f"  Save directory: {save_dir}")
    
    # Create model
    model = create_model(model_name, num_classes=len(CLASS_NAMES), pretrained=True)
    model = model.to(device)
    
    # Print parameter count
    params = count_parameters(model)
    print(f"  Parameters: {params['total_millions']:.2f}M total, "
          f"{params['trainable_millions']:.2f}M trainable")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        img_size=img_size,
        workers=workers,
        augment=True,
    )
    
    # Optimizer
    optimizer = SGD(
        model.parameters(),
        lr=lr0,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=lr0 * lrf
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Mixed precision
    scaler = GradScaler()
    
    # Training state
    best_fitness = 0.0
    no_improve = 0
    start_time = time.time()
    
    print("=" * 70 + "\n")
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(images)
                # Compute loss (simplified for demonstration)
                loss = sum([o.mean() for o in outputs]) * 0.01
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Progress
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - "
                      f"Batch {batch_idx+1}/{num_batches} - "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = sum([o.mean() for o in outputs]) * 0.01
                val_loss += loss.item()
        
        # Update scheduler
        scheduler.step()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start
        
        # Logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - "
              f"LR: {current_lr:.6f} - "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        fitness = 1.0 / (avg_val_loss + 1e-8)
        if fitness > best_fitness:
            best_fitness = fitness
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'fitness': fitness,
            }, weights_dir / 'best.pt')
            no_improve = 0
            print(f"  -> New best model saved!")
        else:
            no_improve += 1
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Save last model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'fitness': fitness,
    }, weights_dir / 'last.pt')
    
    writer.close()
    
    training_time = time.time() - start_time
    
    results = {
        'model': model_name,
        'epochs': epoch + 1,
        'lr0': lr0,
        'lrf': lrf,
        'training_time_hours': training_time / 3600,
        'best_fitness': best_fitness,
        'save_dir': str(save_dir),
    }
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Time: {training_time/3600:.2f} hours")
    print(f"  Best fitness: {best_fitness:.4f}")
    print(f"  Weights: {weights_dir}")
    print("=" * 70 + "\n")
    
    return results


def train_model(
    model_name: str,
    data_yaml: str = None,
    data_dir: str = None,
    **kwargs
) -> Dict:
    """
    Main training function that dispatches to appropriate trainer.
    
    Args:
        model_name: Model name
        data_yaml: Path to YOLO data config (for native training)
        data_dir: Path to data directory (for hybrid training)
        **kwargs: Additional training arguments
        
    Returns:
        Training results
    """
    if 'effnet' in model_name:
        # Use custom training loop for hybrid models
        if data_dir is None:
            data_dir = './data/lung-pet-ct-dx'
        return train_hybrid(model_name, data_dir, **kwargs)
    else:
        # Use native YOLO training
        if data_yaml is None:
            data_yaml = 'configs/data.yaml'
        return train_yolo_native(model_name, data_yaml, **kwargs)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train Lung Cancer CT Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument('--model', type=str, default='yolov8m_effnet',
                        choices=['yolov8m', 'yolov8m_effnet', 'yolov9m', 'yolov9m_effnet'],
                        help='Model architecture')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration YAML file')
    
    # Data settings
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                        help='Path to data configuration YAML')
    parser.add_argument('--data-dir', type=str, default='./data/lung-pet-ct-dx',
                        help='Path to data directory')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    
    # Learning rate
    parser.add_argument('--lr-config', type=str, default='medium',
                        choices=['high', 'medium', 'low'],
                        help='Learning rate preset')
    parser.add_argument('--lr0', type=float, default=None,
                        help='Initial learning rate (overrides preset)')
    parser.add_argument('--lrf', type=float, default=None,
                        help='Final LR factor (overrides preset)')
    
    # Other settings
    parser.add_argument('--device', type=str, default='0', help='GPU device')
    parser.add_argument('--project', type=str, default='runs/train', help='Project dir')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=8, help='Data workers')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        for key, value in config.get('training', {}).items():
            if hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)
    
    # Get learning rate
    lr_config = get_lr_config(args.lr_config)
    lr0 = args.lr0 if args.lr0 is not None else lr_config['lr0']
    lrf = args.lrf if args.lrf is not None else lr_config['lrf']
    
    # Train
    results = train_model(
        model_name=args.model,
        data_yaml=args.data,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr0=lr0,
        lrf=lrf,
        device=args.device if 'effnet' not in args.model else f'cuda:{args.device}',
        project=args.project,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        patience=args.patience,
        resume=args.resume,
    )
    
    print("\nTraining completed successfully!")
    print(f"Results saved to: {results.get('save_dir', 'N/A')}")


if __name__ == "__main__":
    main()
