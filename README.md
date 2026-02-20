# Hybrid YOLO-EfficientNet for Lung Cancer CT Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.7937%2FTCIA.2020.NNC2--0461-green)](https://doi.org/10.7937/TCIA.2020.NNC2-0461)

Official implementation of **"A Hybrid Deep Learning Framework Integrating YOLO and EfficientNetV2 for Automated Lung Cancer Detection and Histological Classification in CT Imaging"**

📄 **Paper:** Submitted to *Biomedical Signal Processing and Control* (Elsevier)

## 🎯 Highlights

- ✅ Novel hybrid architecture integrating YOLOv8/v9 with EfficientNetV2-M backbone
- ✅ Achieves **Recall 0.989**, **Precision 0.988**, **mAP@50 0.993**
- ✅ Simultaneous tumor detection and histological classification (4 subtypes)
- ✅ Systematic learning rate optimization (lr₀=0.003 recommended)
- ✅ Training time ~5 hours on RTX 3090

## 📊 Performance Summary

| Model | Precision | Recall | mAP@50 | mAP@50-95 | Training Time |
|-------|-----------|--------|--------|-----------|---------------|
| YOLOv8m | 0.985 | 0.981 | 0.992 | 0.678 | 4.98 hrs |
| YOLOv8m + EffNetV2 | **0.988** | 0.986 | **0.993** | **0.681** | 5.01 hrs |
| YOLOv9m | 0.982 | 0.984 | 0.991 | 0.670 | 5.72 hrs |
| YOLOv9m + EffNetV2 | 0.985 | **0.989** | 0.992 | 0.672 | 5.89 hrs |

*Results with medium learning rate configuration (lr₀=0.003, lrf=0.01)*

## 🗂️ Repository Structure

```
lung-cancer-ct-yolo-efficientnet/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
├── LICENSE                      # MIT License
│
├── configs/
│   ├── data.yaml               # Dataset configuration
│   ├── train_config.yaml       # Training hyperparameters
│   └── model_configs/
│       ├── yolov8m.yaml
│       ├── yolov8m_effnet.yaml
│       ├── yolov9m.yaml
│       └── yolov9m_effnet.yaml
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Data loading & preprocessing
│   ├── models.py               # Model architectures
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Inference script
│   └── export.py               # Model export (ONNX, TensorRT)
│
├── models/
│   ├── __init__.py
│   ├── backbone.py             # EfficientNetV2 backbone
│   ├── neck.py                 # PAN-FPN neck
│   ├── head.py                 # Detection heads
│   ├── yolov8_effnet.py        # YOLOv8 + EfficientNetV2
│   ├── yolov9_effnet.py        # YOLOv9 + EfficientNetV2
│   └── adapters.py             # Feature adaptation modules
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py        # CLAHE & augmentation
│   ├── metrics.py              # Evaluation metrics
│   ├── visualization.py        # Plotting utilities
│   ├── dicom_utils.py          # DICOM handling
│   └── callbacks.py            # Training callbacks
│
├── scripts/
│   ├── prepare_data.sh         # Data preparation
│   ├── download_dataset.py     # Download from TCIA
│   ├── convert_annotations.py  # VOC to YOLO format
│   ├── train_all_models.sh     # Train all variants
│   └── evaluate_models.sh      # Evaluate all models
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_visualization.ipynb
│
└── results/
    └── figures/                # Generated figures
```

## 🔧 Installation

### Requirements
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA ≥ 11.8 (for GPU training)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/fongrong/lung-cancer-ct-yolo-efficientnet.git
cd lung-cancer-ct-yolo-efficientnet

# Create conda environment
conda create -n lungcancer python=3.10
conda activate lungcancer

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## 📁 Dataset

### Lung-PET-CT-Dx Dataset

We use the **Lung-PET-CT-Dx** dataset from The Cancer Imaging Archive (TCIA).

| Subtype | Train | Val | Test | Total | % |
|---------|-------|-----|------|-------|---|
| Adenocarcinoma | 9,685 | 1,384 | 2,767 | 13,836 | 66.5% |
| Squamous Cell | 2,997 | 428 | 856 | 4,281 | 20.6% |
| Small Cell | 1,437 | 205 | 411 | 2,053 | 9.9% |
| Large Cell | 428 | 60 | 125 | 613 | 2.9% |
| **Total** | **14,547** | **2,077** | **4,159** | **20,783** | **100%** |

### Download & Prepare Data

```bash
# Download from TCIA (requires NBIA Data Retriever)
python scripts/download_dataset.py --output_dir data/raw

# Convert DICOM to PNG and prepare annotations
python scripts/convert_annotations.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --img_size 640

# Or use the preparation script
bash scripts/prepare_data.sh
```

## 🚀 Quick Start

### Training

```bash
# Train YOLOv8m + EfficientNetV2-M (recommended)
python src/train.py \
    --model yolov8m_effnet \
    --data configs/data.yaml \
    --epochs 100 \
    --batch-size 16 \
    --lr0 0.003 \
    --device 0

# Train with configuration file
python src/train.py --config configs/train_config.yaml

# Train all models
bash scripts/train_all_models.sh
```

### Evaluation

```bash
# Evaluate single model
python src/evaluate.py \
    --weights runs/train/yolov8m_effnet/weights/best.pt \
    --data configs/data.yaml \
    --device 0

# Evaluate all models and generate comparison
python src/evaluate.py --eval-all --results-dir runs/train
```

### Inference

```bash
# Single image prediction
python src/predict.py \
    --weights best.pt \
    --source path/to/image.png \
    --conf 0.25

# Batch prediction
python src/predict.py \
    --weights best.pt \
    --source path/to/images/ \
    --save-txt \
    --save-conf
```

## ⚙️ Training Configurations

### Learning Rate Settings

| Config | lr₀ | lrf | Best For |
|--------|-----|-----|----------|
| High | 0.01 | 0.1 | Fast convergence |
| **Medium** | **0.003** | **0.01** | **Optimal balance (recommended)** |
| Low | 0.001 | 0.005 | Fine-tuning |

### Model Architectures

```
YOLOv8m + EfficientNetV2-M
├── Backbone: EfficientNetV2-M (ImageNet pretrained)
│   ├── P3: 80×80×80  → Adapter → 80×80×320
│   ├── P4: 40×40×176 → Adapter → 40×40×320
│   └── P5: 20×20×512 → Adapter → 20×20×320
├── Neck: PAN-FPN (bidirectional feature fusion)
└── Head: Anchor-free detection (decoupled cls/loc)
```

## 📈 Results Reproduction

```bash
# Step 1: Prepare data
bash scripts/prepare_data.sh

# Step 2: Train all models (12 configurations)
bash scripts/train_all_models.sh

# Step 3: Evaluate and generate figures
bash scripts/evaluate_models.sh

# Step 4: Generate paper figures
python utils/visualization.py --generate-paper-figures
```

## 📊 Visualization

### Training Curves
![Training Curves](results/figures/training_curves.png)

### Performance Comparison
![Performance](results/figures/performance_comparison.png)

### Confusion Matrix
![Confusion Matrix](results/figures/confusion_matrix.png)

## 🔬 Model Architecture

### Hybrid Integration Strategy

```python
# Feature extraction from EfficientNetV2-M
features = efficientnet_backbone(input_image)  # [P3, P4, P5]

# Channel adaptation to YOLO neck
adapted_features = [adapter(f) for f, adapter in zip(features, adapters)]

# YOLO neck and head
output = yolo_neck_head(adapted_features)
```

### Feature Adapter Module

```python
class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels=320):
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{yeh2026hybrid,
  title={A Hybrid Deep Learning Framework Integrating YOLO and EfficientNetV2 
         for Automated Lung Cancer Detection and Histological Classification 
         in CT Imaging},
  author={Yeh, Jiang-Chou and Shiau, Mu-Kai and Cheng, Bor-Wen and Yang, Feng-Jung},
  journal={Biomedical Signal Processing and Control},
  year={2026},
  publisher={Elsevier}
}
```

## 📧 Contact

- **Bor-Wen Cheng** - chengbw@yuntech.edu.tw (Corresponding Author)
- **Feng-Jung Yang** - fongrong@ntu.edu.tw (Corresponding Author)

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [WongKinYiu](https://github.com/WongKinYiu/yolov9) for YOLOv9
- [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) for Lung-PET-CT-Dx dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for EfficientNetV2

---

**National Yunlin University of Science and Technology** & **National Taiwan University Hospital**
