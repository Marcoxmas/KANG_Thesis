#!/usr/bin/env python3
"""
Simple 3D Dataset Preprocessing Script for KANG Framework

This script creates all datasets with 3D geometric features.
Supports both with and without global molecular features.

Usage:
    python preprocess_all_3d_datasets.py              # Basic 3D preprocessing
    python preprocess_all_3d_datasets.py --global     # With global features
"""

import os
import sys
from pathlib import Path

# Import dataset classes
from hiv_dataset import HIVGraphDataset
from qm8_dataset import QM8GraphDataset, QM8MultiTaskDataset
from qm9_dataset import QM9GraphDataset, QM9MultiTaskDataset
from toxcast_dataset import ToxCastGraphDataset, ToxCastMultiTaskDataset, get_available_assays


def create_datasets(use_global_features=False):
    """Create all datasets with 3D features."""
    
    dataset_root = "dataset"
    Path(dataset_root).mkdir(exist_ok=True)
    
    print(f"Creating 3D datasets {'with' if use_global_features else 'without'} global features...")
    print(f"Output directory: {dataset_root}")
    print("=" * 60)
    
    # Check data availability
    data_files = {
        'HIV': 'data/HIV.csv',
        'QM8': 'data/qm8.csv', 
        'QM9': 'data/qm9.csv',
        'ToxCast': 'data/toxcast_data.csv'
    }
    
    available_datasets = []
    for name, path in data_files.items():
        if os.path.exists(path):
            available_datasets.append(name)
            print(f"✓ {name} dataset available")
        else:
            print(f"✗ {name} dataset not found at {path}")
    
    if not available_datasets:
        print("No datasets found! Please ensure data files are in the data/ directory.")
        return
    
    print()
    
    # Create HIV dataset
    if 'HIV' in available_datasets:
        print("Creating HIV dataset...")
        try:
            dataset = HIVGraphDataset(
                root=f"{dataset_root}/HIV",
                use_global_features=use_global_features,
                use_3d_geo=True
            )
            print(f"✓ HIV: {len(dataset)} samples")
        except Exception as e:
            print(f"✗ HIV failed: {e}")
    
    # Create QM8 datasets
    if 'QM8' in available_datasets:
        print("Creating QM8 datasets...")
        qm8_columns = [
                "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2",
                "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0",
                "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"
            ]
        # Single task
        try:
            for col in qm8_columns:
                try:
                    dataset = QM8GraphDataset(
                        root=f"{dataset_root}/QM8_{col}",
                        target_column=col,
                        use_global_features=use_global_features,
                        use_3d_geo=True
                    )
                    print(f"✓ QM8 {col}: {len(dataset)} samples")
                except Exception as e:
                    print(f"✗ QM8 {col} failed: {e}")
            print(f"✓ QM8 E1-CC2: {len(dataset)} samples")
        except Exception as e:
            print(f"✗ QM8 E1-CC2 failed: {e}")
        
        # Multi-task
        try:
            # Create target hash like in graph_regression.py
            target_str = " ".join(sorted(qm8_columns))
            target_hash = str(sum(ord(c) for c in target_str) % 10**8)
            dataset = QM8MultiTaskDataset(
                root=f"{dataset_root}/QM8_multitask_{target_hash}",
                target_columns=qm8_columns,
                use_global_features=use_global_features,
                use_3d_geo=True
            )
            print(f"✓ QM8 multitask ({len(qm8_columns)} targets): {len(dataset)} samples")
        except Exception as e:
            print(f"✗ QM8 multitask failed: {e}")
    
    # Create QM9 datasets
    if 'QM9' in available_datasets:
        print("Creating QM9 datasets...")
        qm9_columns = [
            "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0", "u298", "h298", "g298", "cv"
        ]
        # Single task
        for col in qm9_columns:
            try:
                dataset = QM9GraphDataset(
                    root=f"{dataset_root}/QM9_{col}",
                    target_column=col,
                    use_global_features=use_global_features,
                    use_3d_geo=True
                )
                print(f"✓ QM9 {col}: {len(dataset)} samples")
            except Exception as e:
                print(f"✗ QM9 {col} failed: {e}")

        # Multi-task
        try:
            # Create target hash like in graph_regression.py
            target_str = " ".join(sorted(qm9_columns))
            target_hash = str(sum(ord(c) for c in target_str) % 10**8)
            dataset = QM9MultiTaskDataset(
                root=f"{dataset_root}/QM9_multitask_{target_hash}",
                target_columns=qm9_columns,
                use_global_features=use_global_features,
                use_3d_geo=True
            )
            print(f"✓ QM9 multitask ({len(qm9_columns)} targets): {len(dataset)} samples")
        except Exception as e:
            print(f"✗ QM9 multitask failed: {e}")
    
    # Create ToxCast dataset
    if 'ToxCast' in available_datasets:
        print("Creating ToxCast dataset...")
        toxcast_assays = [
            "TOX21_AhR_LUC_Agonist",
            "TOX21_Aromatase_Inhibition",
            "TOX21_AutoFluor_HEK293_Cell_blue",
            "TOX21_p53_BLA_p3_ch1",
            "TOX21_p53_BLA_p4_ratio"
        ]
        # Single Task
        for col in toxcast_assays:
            try:
                dataset = ToxCastGraphDataset(
                    root=f"{dataset_root}/TOXCAST_{col}",
                    target_column=col,
                    use_global_features=use_global_features,
                    use_3d_geo=True
                )
                print(f"✓ ToxCast {col}: {len(dataset)} samples")
            except Exception as e:
                print(f"✗ ToxCast {col} failed: {e}")
        # Multi-task
        try:
            # Create assay hash like in graph_classification.py
            assay_str = " ".join(sorted(toxcast_assays))
            assay_hash = str(sum(ord(c) for c in assay_str) % 10**8)
            dataset = ToxCastMultiTaskDataset(
                root=f"{dataset_root}/TOXCAST_multitask_{assay_hash}",
                target_columns=toxcast_assays,
                use_global_features=use_global_features,
                use_3d_geo=True
            )
            print(f"✓ ToxCast multitask ({len(toxcast_assays)} targets): {len(dataset)} samples")
        except Exception as e:
            print(f"✗ ToxCast failed: {e}")
    
    print()
    print("=" * 60)
    print("Dataset creation completed!")
    print(f"All datasets saved in: {dataset_root}/")


def main():
    """Main function."""
    use_global = False
    
    # Simple argument parsing
    if len(sys.argv) > 1:
        if '--global' in sys.argv or '--global-features' in sys.argv:
            use_global = True
            
            # Check if global features are available
            try:
                from src.global_features import get_global_extractor
                extractor = get_global_extractor()
                if extractor is None:
                    print("Global features requested but descriptastorus not available.")
                    print("Install with: pip install descriptastorus")
                    sys.exit(1)
            except ImportError:
                print("Global features requested but descriptastorus not available.")
                print("Install with: pip install descriptastorus")
                sys.exit(1)
    
    create_datasets(use_global_features=use_global)


if __name__ == "__main__":
    main()
