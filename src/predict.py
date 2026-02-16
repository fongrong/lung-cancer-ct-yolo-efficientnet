#!/usr/bin/env python3
"""
Inference script for Lung Cancer CT Classification.

Performs predictions on new CT images and visualizes results.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Usage:
    # Single image prediction
    python src/predict.py --weights best.pt --source image.png
    
    # Batch prediction on directory
    python src/predict.py --weights best.pt --source images/ --save-dir results/
    
    # With visualization
    python src/predict.py --weights best.pt --source image.png --show --save
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CLASS_NAMES, CLASS_MAPPING

# Color palette for visualization (BGR format for OpenCV)
COLORS = {
    'Adenocarcinoma': (255, 100, 100),    # Blue
    'Small_Cell': (100, 255, 100),         # Green
    'Large_Cell': (100, 100, 255),         # Red
    'Squamous_Cell': (255, 255, 100),      # Cyan
}


def predict(
    weights_path: str,
    source: Union[str, np.ndarray, List[str]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 100,
    device: str = '0',
    img_size: int = 640,
    classes: Optional[List[int]] = None,
    agnostic_nms: bool = False,
    half: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """
    Run inference on images.
    
    Args:
        weights_path: Path to model weights
        source: Image path, numpy array, or list of paths
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
        device: GPU device
        img_size: Input image size
        classes: Filter by class indices
        agnostic_nms: Class-agnostic NMS
        half: Use FP16 inference
        verbose: Print results
        
    Returns:
        List of prediction dictionaries
    """
    # Load model
    model = YOLO(weights_path)
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf_thres,
        iou=iou_thres,
        max_det=max_det,
        device=device,
        imgsz=img_size,
        classes=classes,
        agnostic_nms=agnostic_nms,
        half=half,
        verbose=verbose,
    )
    
    # Process results
    predictions = []
    for result in results:
        pred = {
            'image_path': result.path if hasattr(result, 'path') else None,
            'image_shape': result.orig_shape,
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': [],
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()
            
            for box, cls, conf in zip(boxes, classes, confidences):
                pred['boxes'].append(box.tolist())
                pred['classes'].append(int(cls))
                pred['confidences'].append(float(conf))
                pred['class_names'].append(CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f'class_{cls}')
        
        predictions.append(pred)
    
    return predictions


def predict_single_image(
    weights_path: str,
    image_path: str,
    conf_thres: float = 0.25,
    device: str = '0',
    save_path: Optional[str] = None,
    show: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Predict on a single image with visualization.
    
    Args:
        weights_path: Path to model weights
        image_path: Path to input image
        conf_thres: Confidence threshold
        device: GPU device
        save_path: Path to save annotated image
        show: Whether to display image
        
    Returns:
        Tuple of (annotated_image, predictions)
    """
    # Run prediction
    predictions = predict(
        weights_path=weights_path,
        source=image_path,
        conf_thres=conf_thres,
        device=device,
        verbose=False,
    )
    
    pred = predictions[0]
    
    # Load and annotate image
    image = cv2.imread(image_path)
    annotated = draw_predictions(image, pred)
    
    # Save if requested
    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image to: {save_path}")
    
    # Show if requested
    if show:
        cv2.imshow('Prediction', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return annotated, pred


def draw_predictions(
    image: np.ndarray,
    prediction: Dict,
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """
    Draw prediction boxes and labels on image.
    
    Args:
        image: Input image (BGR format)
        prediction: Prediction dictionary
        line_thickness: Box line thickness
        font_scale: Font scale for labels
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    for box, cls_name, conf in zip(
        prediction['boxes'],
        prediction['class_names'],
        prediction['confidences']
    ):
        # Get color for this class
        color = COLORS.get(cls_name, (128, 128, 128))
        
        # Draw box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
        
        # Draw label background
        label = f'{cls_name}: {conf:.2f}'
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return annotated


def batch_predict(
    weights_path: str,
    source_dir: str,
    save_dir: str,
    conf_thres: float = 0.25,
    device: str = '0',
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
) -> List[Dict]:
    """
    Run batch prediction on a directory of images.
    
    Args:
        weights_path: Path to model weights
        source_dir: Directory containing images
        save_dir: Directory to save results
        conf_thres: Confidence threshold
        device: GPU device
        extensions: Allowed image extensions
        
    Returns:
        List of prediction dictionaries
    """
    source_dir = Path(source_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_paths = []
    for ext in extensions:
        image_paths.extend(source_dir.glob(f'*{ext}'))
    
    if not image_paths:
        print(f"No images found in {source_dir}")
        return []
    
    print(f"Found {len(image_paths)} images to process")
    
    all_predictions = []
    
    for img_path in image_paths:
        save_path = save_dir / f"{img_path.stem}_pred{img_path.suffix}"
        
        _, pred = predict_single_image(
            weights_path=weights_path,
            image_path=str(img_path),
            conf_thres=conf_thres,
            device=device,
            save_path=str(save_path),
        )
        
        pred['source_path'] = str(img_path)
        pred['save_path'] = str(save_path)
        all_predictions.append(pred)
        
        # Print detection summary
        n_detections = len(pred['boxes'])
        print(f"  {img_path.name}: {n_detections} detections")
    
    # Save predictions to JSON
    import json
    results_file = save_dir / 'predictions.json'
    with open(results_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    print(f"Predictions JSON: {results_file}")
    
    return all_predictions


def print_prediction_summary(predictions: List[Dict]):
    """Print summary of predictions."""
    total_detections = sum(len(p['boxes']) for p in predictions)
    
    # Count detections per class
    class_counts = {name: 0 for name in CLASS_NAMES}
    for pred in predictions:
        for cls_name in pred['class_names']:
            if cls_name in class_counts:
                class_counts[cls_name] += 1
    
    print("\n" + "=" * 50)
    print("PREDICTION SUMMARY")
    print("=" * 50)
    print(f"Total images:     {len(predictions)}")
    print(f"Total detections: {total_detections}")
    print("\nDetections per class:")
    for cls_name, count in class_counts.items():
        print(f"  {cls_name}: {count}")
    print("=" * 50)


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description='Lung Cancer CT Classification Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--source', type=str, required=True,
                        help='Image path or directory')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size')
    parser.add_argument('--save-dir', type=str, default='runs/predict',
                        help='Save directory')
    parser.add_argument('--save', action='store_true',
                        help='Save annotated images')
    parser.add_argument('--show', action='store_true',
                        help='Show images')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as text files')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in text files')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    
    if source_path.is_dir():
        # Batch prediction
        predictions = batch_predict(
            weights_path=args.weights,
            source_dir=args.source,
            save_dir=args.save_dir,
            conf_thres=args.conf,
            device=args.device,
        )
        print_prediction_summary(predictions)
        
    elif source_path.is_file():
        # Single image prediction
        save_path = None
        if args.save:
            save_dir = Path(args.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / f"{source_path.stem}_pred{source_path.suffix}")
        
        annotated, pred = predict_single_image(
            weights_path=args.weights,
            image_path=args.source,
            conf_thres=args.conf,
            device=args.device,
            save_path=save_path,
            show=args.show,
        )
        
        # Print results
        print("\nDetections:")
        for cls_name, conf, box in zip(pred['class_names'], pred['confidences'], pred['boxes']):
            x1, y1, x2, y2 = map(int, box)
            print(f"  {cls_name}: {conf:.3f} [{x1}, {y1}, {x2}, {y2}]")
        
        if args.save_txt:
            txt_path = Path(args.save_dir) / f"{source_path.stem}.txt"
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(txt_path, 'w') as f:
                for cls, conf, box in zip(pred['classes'], pred['confidences'], pred['boxes']):
                    x1, y1, x2, y2 = box
                    # Convert to YOLO format (normalized xywh)
                    h, w = pred['image_shape'][:2]
                    x_center = (x1 + x2) / 2 / w
                    y_center = (y1 + y2) / 2 / h
                    width = (x2 - x1) / w
                    height = (y2 - y1) / h
                    
                    if args.save_conf:
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.4f}\n")
                    else:
                        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            print(f"\nSaved predictions to: {txt_path}")
    
    else:
        print(f"Error: Source not found: {args.source}")
        sys.exit(1)
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
