#!/usr/bin/env python3
"""
Download Lung-PET-CT-Dx dataset from The Cancer Imaging Archive.

This script provides instructions and utilities for downloading the dataset.
Note: The NBIA Data Retriever tool is required for downloading TCIA datasets.

Authors: Jiang-Chou Yeh, Mu-Kai Shiau, Bor-Wen Cheng, Feng-Jung Yang

Usage:
    python scripts/download_dataset.py --output_dir data/raw
"""

import argparse
import os
from pathlib import Path


DATASET_INFO = """
================================================================================
LUNG-PET-CT-DX DATASET
================================================================================

Dataset:    Lung-PET-CT-Dx
Source:     The Cancer Imaging Archive (TCIA)
DOI:        10.7937/TCIA.2020.NNC2-0461
License:    CC BY 4.0

Description:
    A large-scale CT and PET/CT dataset for lung cancer diagnosis containing
    251,135 DICOM images from 355 patients with histologically confirmed
    lung cancer, annotated with bounding boxes for four histological subtypes.

Subtypes:
    - Adenocarcinoma (66.5%)
    - Squamous Cell Carcinoma (20.6%)
    - Small Cell Carcinoma (9.9%)
    - Large Cell Carcinoma (2.9%)

================================================================================
DOWNLOAD INSTRUCTIONS
================================================================================

1. Install NBIA Data Retriever:
   - Visit: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
   - Download and install NBIA Data Retriever for your platform

2. Download manifest file:
   - Go to: https://doi.org/10.7937/TCIA.2020.NNC2-0461
   - Click "Download" to get the manifest file (.tcia)

3. Open manifest with NBIA Data Retriever:
   - Launch NBIA Data Retriever
   - Load the downloaded .tcia manifest
   - Select download location
   - Start download (may take several hours)

4. Run data preparation:
   python scripts/convert_annotations.py --input_dir data/raw --output_dir data/processed

================================================================================
"""


def check_nbia_retriever():
    """Check if NBIA Data Retriever is available."""
    import shutil
    
    # Common installation paths
    possible_paths = [
        '/opt/nbia-data-retriever',
        os.path.expanduser('~/NBIA-Data-Retriever'),
        'C:/NBIA-Data-Retriever',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found NBIA Data Retriever at: {path}")
            return True
    
    # Check if in PATH
    if shutil.which('NBIADataRetriever'):
        print("NBIA Data Retriever found in PATH")
        return True
    
    return False


def create_directory_structure(output_dir: Path):
    """Create the expected directory structure."""
    dirs = [
        output_dir / 'raw',
        output_dir / 'processed' / 'images' / 'train',
        output_dir / 'processed' / 'images' / 'val',
        output_dir / 'processed' / 'images' / 'test',
        output_dir / 'processed' / 'labels' / 'train',
        output_dir / 'processed' / 'labels' / 'val',
        output_dir / 'processed' / 'labels' / 'test',
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"Created: {d}")


def main():
    parser = argparse.ArgumentParser(
        description='Download Lung-PET-CT-Dx dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DATASET_INFO
    )
    
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory for dataset')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check prerequisites')
    
    args = parser.parse_args()
    
    print(DATASET_INFO)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    has_nbia = check_nbia_retriever()
    
    if not has_nbia:
        print("\n⚠️  NBIA Data Retriever not found!")
        print("Please install it from:")
        print("https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images")
    else:
        print("✓ NBIA Data Retriever found")
    
    if args.check_only:
        return
    
    # Create directory structure
    output_dir = Path(args.output_dir)
    print(f"\nCreating directory structure in: {output_dir}")
    create_directory_structure(output_dir)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Download the dataset manifest from TCIA")
    print("2. Use NBIA Data Retriever to download images")
    print("3. Place downloaded files in:", output_dir / 'raw')
    print("4. Run: python scripts/convert_annotations.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
