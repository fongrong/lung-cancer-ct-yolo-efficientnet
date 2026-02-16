"""
Lung Cancer CT Classification Package

A Hybrid Deep Learning Framework Integrating YOLO and EfficientNetV2 
for Automated Lung Cancer Detection and Histological Classification in CT Imaging

Authors:
    Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Institutions:
    National Yunlin University of Science and Technology
    National Taiwan University Hospital

Paper:
    Submitted to Computerized Medical Imaging and Graphics (Elsevier)

License:
    MIT License
"""

__version__ = "1.0.0"
__author__ = "Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang"
__email__ = "fongrong@ntu.edu.tw"
__license__ = "MIT"

# Class names
CLASS_NAMES = [
    "Adenocarcinoma",
    "Small_Cell", 
    "Large_Cell",
    "Squamous_Cell"
]

# Class mapping from original dataset labels
CLASS_MAPPING = {
    'A': 0,  # Adenocarcinoma
    'B': 1,  # Small Cell
    'E': 2,  # Large Cell
    'G': 3,  # Squamous Cell
}

# Reverse mapping
CLASS_MAPPING_INV = {v: k for k, v in CLASS_MAPPING.items()}

# Import main modules
from .dataset import LungCancerDataset, create_dataloaders
from .models import create_model
from .train import train_model
from .evaluate import evaluate_model
from .predict import predict

__all__ = [
    # Constants
    "CLASS_NAMES",
    "CLASS_MAPPING",
    "CLASS_MAPPING_INV",
    # Functions
    "LungCancerDataset",
    "create_dataloaders",
    "create_model",
    "train_model",
    "evaluate_model",
    "predict",
]
