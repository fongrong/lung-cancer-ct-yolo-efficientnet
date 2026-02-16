"""
Preprocessing utilities for Lung Cancer CT Classification.

Includes CLAHE enhancement, augmentation, and DICOM handling.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        image: Input image (RGB or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    if len(image.shape) == 3:
        # Color image - apply to L channel in LAB space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        enhanced = clahe.apply(image)
    
    return enhanced


def add_salt_pepper_noise(
    image: np.ndarray,
    prob: float = 0.02,
) -> np.ndarray:
    """
    Add salt and pepper noise to image.
    
    Args:
        image: Input image
        prob: Probability of noise for each pixel
        
    Returns:
        Noisy image
    """
    noisy = image.copy()
    
    # Generate random noise mask
    random = np.random.random(image.shape[:2])
    
    # Salt (white)
    noisy[random < prob/2] = 255
    
    # Pepper (black)
    noisy[(random >= prob/2) & (random < prob)] = 0
    
    return noisy


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize image with letterboxing to preserve aspect ratio.
    
    Args:
        image: Input image
        new_shape: Target size (height, width)
        color: Padding color
        auto: Auto-adjust padding to be mod of stride
        scale_fill: Stretch to fill
        scaleup: Allow scaling up
        stride: Stride for auto padding
        
    Returns:
        Tuple of (resized_image, scale_ratio, padding)
    """
    shape = image.shape[:2]  # Current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    
    return image, ratio, (dw, dh)


def normalize_ct_window(
    image: np.ndarray,
    window_center: float = -600,
    window_width: float = 1500,
) -> np.ndarray:
    """
    Apply CT window/level adjustment.
    
    Common lung window: center=-600, width=1500
    
    Args:
        image: Input CT image (Hounsfield units)
        window_center: Window center
        window_width: Window width
        
    Returns:
        Normalized image in [0, 255]
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    # Apply windowing
    image = np.clip(image, lower, upper)
    
    # Normalize to [0, 255]
    image = ((image - lower) / window_width * 255).astype(np.uint8)
    
    return image


def augment_image(
    image: np.ndarray,
    labels: np.ndarray,
    p_flip: float = 0.5,
    p_brightness: float = 0.5,
    p_hsv: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply augmentation to image and labels.
    
    Args:
        image: Input image
        labels: Labels in YOLO format [class, x, y, w, h]
        p_flip: Probability of horizontal flip
        p_brightness: Probability of brightness adjustment
        p_hsv: Probability of HSV augmentation
        
    Returns:
        Tuple of (augmented_image, augmented_labels)
    """
    import random
    
    # Horizontal flip
    if random.random() < p_flip:
        image = np.fliplr(image).copy()
        if len(labels) > 0:
            labels[:, 1] = 1 - labels[:, 1]  # Flip x-coordinates
    
    # Brightness/contrast
    if random.random() < p_brightness:
        alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
        beta = -20 + random.random() * 40    # -20 to 20
        image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    
    # HSV augmentation
    if random.random() < p_hsv:
        image = hsv_augment(image)
    
    return image, labels


def hsv_augment(
    image: np.ndarray,
    h_gain: float = 0.015,
    s_gain: float = 0.7,
    v_gain: float = 0.4,
) -> np.ndarray:
    """
    Apply HSV color augmentation.
    
    Args:
        image: Input RGB image
        h_gain: Hue gain range
        s_gain: Saturation gain range
        v_gain: Value gain range
        
    Returns:
        Augmented image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] * r[0], 0, 180)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * r[1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * r[2], 0, 255)
    
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class Preprocessor:
    """
    Image preprocessing pipeline for CT images.
    
    Args:
        img_size: Target image size
        clahe: Whether to apply CLAHE
        clahe_clip_limit: CLAHE clip limit
        clahe_tile_size: CLAHE tile grid size
        normalize: Whether to normalize to [0, 1]
        
    Example:
        >>> preprocessor = Preprocessor(img_size=640, clahe=True)
        >>> image = cv2.imread('ct_scan.png')
        >>> processed = preprocessor(image)
    """
    
    def __init__(
        self,
        img_size: int = 640,
        clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        normalize: bool = True,
    ):
        self.img_size = img_size
        self.clahe = clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.normalize = normalize
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Process image through preprocessing pipeline."""
        # Apply CLAHE
        if self.clahe:
            image = apply_clahe(
                image,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_size
            )
        
        # Resize with letterboxing
        image, _, _ = letterbox(image, new_shape=(self.img_size, self.img_size))
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        return image


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing utilities...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test CLAHE
    enhanced = apply_clahe(test_image)
    print(f"CLAHE: {test_image.shape} -> {enhanced.shape}")
    
    # Test letterbox
    resized, ratio, padding = letterbox(test_image, (640, 640))
    print(f"Letterbox: {test_image.shape} -> {resized.shape}")
    
    # Test salt-pepper noise
    noisy = add_salt_pepper_noise(test_image, prob=0.02)
    print(f"Salt-pepper noise: {test_image.shape} -> {noisy.shape}")
    
    # Test preprocessor
    preprocessor = Preprocessor(img_size=640, clahe=True)
    processed = preprocessor(test_image)
    print(f"Preprocessor: {test_image.shape} -> {processed.shape}, range [{processed.min():.3f}, {processed.max():.3f}]")
    
    print("\nAll tests passed!")
