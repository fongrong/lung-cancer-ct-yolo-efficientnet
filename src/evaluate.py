#!/usr/bin/env python3
"""
Evaluation script for Lung Cancer CT Classification.

Evaluates trained models and generates comprehensive metrics including
precision, recall, mAP, and confusion matrices.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Usage:
    # Evaluate single model
    python src/evaluate.py --weights runs/train/exp/weights/best.pt --data configs/data.yaml
    
    # Evaluate all models in directory
    python src/evaluate.py --eval-all --results-dir runs/train
    
    # Generate comparison plots
    python src/evaluate.py --weights best.pt --data configs/data.yaml --plots
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CLASS_NAMES, NUM_CLASSES


def evaluate_yolo(
    weights_path: str,
    data_yaml: str,
    batch_size: int = 16,
    img_size: int = 640,
    conf_thres: float = 0.001,
    iou_thres: float = 0.7,
    device: str = '0',
    save_json: bool = False,
    save_dir: Optional[str] = None,
    plots: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate YOLO model on test set.
    
    Args:
        weights_path: Path to model weights
        data_yaml: Path to data configuration
        batch_size: Batch size for evaluation
        img_size: Input image size
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        device: GPU device
        save_json: Whether to save results as JSON
        save_dir: Directory to save results
        plots: Whether to generate plots
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if verbose:
        print("\n" + "=" * 70)
        print(f"EVALUATING: {weights_path}")
        print("=" * 70)
    
    # Load model
    model = YOLO(weights_path)
    
    # Run validation
    results = model.val(
        data=data_yaml,
        batch=batch_size,
        imgsz=img_size,
        conf=conf_thres,
        iou=iou_thres,
        device=device,
        plots=plots,
        save_json=save_json,
    )
    
    # Extract metrics
    metrics = {
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
    }
    
    # Per-class metrics
    if hasattr(results, 'maps') and results.maps is not None:
        metrics['mAP_per_class'] = {
            CLASS_NAMES[i]: float(results.maps[i])
            for i in range(min(len(CLASS_NAMES), len(results.maps)))
        }
    
    # Print results
    if verbose:
        print(f"\nResults:")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  mAP@50:     {metrics['mAP50']:.4f}")
        print(f"  mAP@50-95:  {metrics['mAP50-95']:.4f}")
        
        if 'mAP_per_class' in metrics:
            print(f"\nPer-class mAP@50-95:")
            for cls_name, map_val in metrics['mAP_per_class'].items():
                print(f"  {cls_name}: {map_val:.4f}")
    
    # Save results
    if save_json and save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {save_path / 'metrics.json'}")
    
    return metrics


def evaluate_all_models(
    results_dir: str,
    data_yaml: str,
    output_file: str = 'comparison_results.json',
    **kwargs
) -> Dict:
    """
    Evaluate all trained models in a directory and compare results.
    
    Args:
        results_dir: Directory containing trained model folders
        data_yaml: Path to data configuration
        output_file: Output filename for comparison results
        
    Returns:
        Dictionary containing results for all models
    """
    results_dir = Path(results_dir)
    all_results = {}
    
    # Find all model weights
    weight_files = list(results_dir.glob('*/weights/best.pt'))
    
    if not weight_files:
        print(f"No model weights found in {results_dir}")
        return {}
    
    print(f"\nFound {len(weight_files)} models to evaluate")
    
    for weights_path in tqdm(weight_files, desc="Evaluating models"):
        model_name = weights_path.parent.parent.name
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print('='*50)
        
        try:
            metrics = evaluate_yolo(
                weights_path=str(weights_path),
                data_yaml=data_yaml,
                verbose=True,
                **kwargs
            )
            all_results[model_name] = metrics
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Save comparison
    output_path = results_dir / output_file
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    print(f"{'Model':<30} {'Precision':>10} {'Recall':>10} {'mAP@50':>10} {'mAP@50-95':>12}")
    print("-" * 90)
    
    for model_name, metrics in all_results.items():
        if 'error' not in metrics:
            print(f"{model_name:<30} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['mAP50']:>10.4f} "
                  f"{metrics['mAP50-95']:>12.4f}")
    
    print("=" * 90)
    print(f"\nResults saved to: {output_path}")
    
    return all_results


def compute_confusion_matrix(
    predictions: List[Dict],
    ground_truth: List[Dict],
    num_classes: int = NUM_CLASSES,
    conf_thres: float = 0.25,
    iou_thres: float = 0.5,
) -> np.ndarray:
    """
    Compute confusion matrix from predictions and ground truth.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: List of ground truth dictionaries
        num_classes: Number of classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for matching
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, gt in zip(predictions, ground_truth):
        # Filter predictions by confidence
        pred_boxes = pred['boxes'][pred['scores'] >= conf_thres]
        pred_labels = pred['labels'][pred['scores'] >= conf_thres]
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        # Match predictions to ground truth using IoU
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            iou_matrix = compute_iou(pred_boxes, gt_boxes)
            
            for i, (pred_label, ious) in enumerate(zip(pred_labels, iou_matrix)):
                max_iou_idx = np.argmax(ious)
                if ious[max_iou_idx] >= iou_thres:
                    gt_label = gt_labels[max_iou_idx]
                    confusion[gt_label, pred_label] += 1
    
    return confusion


def compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2] format
        
    Returns:
        IoU matrix of shape (N, M)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute intersection
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    
    # Compute union
    union = area1[:, None] + area2[None, :] - intersection
    
    return intersection / (union + 1e-8)


def generate_classification_report(
    confusion_matrix: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Generate classification report from confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names
        
    Returns:
        Dictionary containing per-class and overall metrics
    """
    num_classes = len(class_names)
    report = {}
    
    # Per-class metrics
    for i, cls_name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        report[cls_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1-score': float(f1),
            'support': int(confusion_matrix[i, :].sum()),
        }
    
    # Overall metrics (macro average)
    precisions = [report[cls]['precision'] for cls in class_names]
    recalls = [report[cls]['recall'] for cls in class_names]
    f1s = [report[cls]['f1-score'] for cls in class_names]
    
    report['macro_avg'] = {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1-score': float(np.mean(f1s)),
    }
    
    # Weighted average
    supports = [report[cls]['support'] for cls in class_names]
    total_support = sum(supports)
    
    report['weighted_avg'] = {
        'precision': float(sum(p * s for p, s in zip(precisions, supports)) / total_support),
        'recall': float(sum(r * s for r, s in zip(recalls, supports)) / total_support),
        'f1-score': float(sum(f * s for f, s in zip(f1s, supports)) / total_support),
    }
    
    return report


def evaluate_model(
    weights_path: str,
    data_yaml: str,
    **kwargs
) -> Dict:
    """
    Convenience function for model evaluation.
    
    Args:
        weights_path: Path to model weights
        data_yaml: Path to data configuration
        **kwargs: Additional arguments for evaluate_yolo
        
    Returns:
        Evaluation metrics dictionary
    """
    return evaluate_yolo(weights_path, data_yaml, **kwargs)


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate Lung Cancer CT Classification Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights')
    parser.add_argument('--data', type=str, default='configs/data.yaml',
                        help='Path to data configuration')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='IoU threshold')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results as JSON')
    parser.add_argument('--save-dir', type=str, default='runs/evaluate',
                        help='Save directory')
    parser.add_argument('--plots', action='store_true',
                        help='Generate plots')
    parser.add_argument('--eval-all', action='store_true',
                        help='Evaluate all models in results directory')
    parser.add_argument('--results-dir', type=str, default='runs/train',
                        help='Directory containing trained models')
    
    args = parser.parse_args()
    
    if args.eval_all:
        # Evaluate all models
        results = evaluate_all_models(
            results_dir=args.results_dir,
            data_yaml=args.data,
            batch_size=args.batch_size,
            img_size=args.img_size,
            conf_thres=args.conf,
            iou_thres=args.iou,
            device=args.device,
            plots=args.plots,
        )
    elif args.weights:
        # Evaluate single model
        results = evaluate_yolo(
            weights_path=args.weights,
            data_yaml=args.data,
            batch_size=args.batch_size,
            img_size=args.img_size,
            conf_thres=args.conf,
            iou_thres=args.iou,
            device=args.device,
            save_json=args.save_json,
            save_dir=args.save_dir,
            plots=args.plots,
        )
    else:
        parser.print_help()
        print("\nError: Either --weights or --eval-all must be specified")
        sys.exit(1)
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
