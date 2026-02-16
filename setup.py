"""
Setup script for Lung Cancer CT Classification package.

A Hybrid Deep Learning Framework Integrating YOLO and EfficientNetV2 
for Automated Lung Cancer Detection and Histological Classification in CT Imaging
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8")
requirements = [
    line.strip() 
    for line in requirements.split("\n") 
    if line.strip() and not line.startswith("#") and not line.startswith("=")
]

setup(
    name="lung-cancer-ct-yolo-efficientnet",
    version="1.0.0",
    author="Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang",
    author_email="fongrong@ntu.edu.tw",
    description="Hybrid YOLO-EfficientNet for Lung Cancer CT Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fongrong/lung-cancer-ct-yolo-efficientnet",
    project_urls={
        "Bug Tracker": "https://github.com/fongrong/lung-cancer-ct-yolo-efficientnet/issues",
        "Documentation": "https://github.com/fongrong/lung-cancer-ct-yolo-efficientnet#readme",
        "Source Code": "https://github.com/fongrong/lung-cancer-ct-yolo-efficientnet",
    },
    packages=find_packages(exclude=["tests", "notebooks", "results"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords=[
        "lung cancer",
        "CT imaging",
        "deep learning",
        "YOLO",
        "EfficientNet",
        "medical imaging",
        "computer-aided diagnosis",
        "object detection",
        "histological classification",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "export": [
            "onnx>=1.14.0",
            "onnxruntime>=1.15.0",
            "tensorrt>=8.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lungcancer-train=src.train:main",
            "lungcancer-eval=src.evaluate:main",
            "lungcancer-predict=src.predict:main",
            "lungcancer-export=src.export:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
