import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import shutil
from rdkit import RDLogger
from smiles_to_graph import smiles_to_data
from src.global_features import get_global_extractor, get_global_feature_dim

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')

class QM8GraphDataset(InMemoryDataset):

    def __init__(self, root, target_column='E1-CC2', use_global_features=False, use_3d_geo=False, transform=None, pre_transform=None):
        self.csv_file = "data/qm8.csv"
        self.target_column = target_column
        self.use_global_features = use_global_features
        self.use_3d_geo = use_3d_geo

        # Define available target columns for QM8
        self.available_targets = [
            'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
            'E1-PBE0.1', 'E2-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1',
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
        ]

        if target_column not in self.available_targets:
            raise ValueError(f"target_column '{target_column}' not available. "
                           f"Available targets: {self.available_targets}")
        
        # Initialize global feature extractor if needed
        if self.use_global_features:
            self.global_extractor = get_global_extractor()
            if self.global_extractor is None:
                raise ImportError(
                    "Global features requested but descriptastorus not available. "
                    "Please install descriptastorus: pip install descriptastorus"
                )
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def len(self):
        return len(self.slices['x']) - 1

    def get(self, idx):
        return super().get(idx)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        global_features_count = 0
        
        print(f"Processing QM8 dataset with target column: {self.target_column}")
        print(f"Total molecules to process: {len(df)}")
        print(f"Global features: {'enabled' if self.use_global_features else 'disabled'}")
        print(f"3D geometric features: {'enabled' if self.use_3d_geo else 'disabled'}")
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(df)} molecules...")
                
            smiles = row['smiles']
            target_value = row[self.target_column]
            
            if pd.isnull(target_value):
                invalid_count += 1
                continue
                
            try:
                # For QM8 with 3D features, pass the row index to match SDF file
                mol_index = idx if self.use_3d_geo else None
                data = smiles_to_data(smiles, labels=float(target_value), use_3d_geo=self.use_3d_geo, 
                                    dataset_type='QM8', mol_index=mol_index)
                if data is not None and hasattr(data, 'edge_index'):
                    # Extract global features if requested
                    if self.use_global_features:
                        global_features = self.global_extractor.extract_features(smiles)
                        if global_features is not None:
                            data.global_features = torch.tensor(global_features, dtype=torch.float32)
                            global_features_count += 1
                        else:
                            # Skip molecules where global features cannot be extracted
                            if idx % 10000 == 0:  # Only print occasionally to avoid spam
                                print(f"Skipped molecule due to failed global feature extraction: {smiles}")
                            invalid_count += 1
                            continue
                    
                    data_list.append(data)
                    valid_count += 1
                else:
                    print(f"Excluded invalid graph for SMILES: {smiles}")
                    invalid_count += 1
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                invalid_count += 1
        
        print(f"Processed {valid_count} valid molecules, {invalid_count} invalid/missing")
        if self.use_global_features:
            print(f"Successfully extracted global features for {global_features_count} molecules")
        
        if not data_list:
            raise ValueError("No valid data found! Check your data.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        raw_csv_path = self.raw_paths[0]
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        
        main_csv_path = os.path.join(os.path.dirname(os.path.dirname(self.root)), self.csv_file)
        
        if not os.path.exists(raw_csv_path) and os.path.exists(main_csv_path):
            print(f"Copying {main_csv_path} to {raw_csv_path}")
            shutil.copy2(main_csv_path, raw_csv_path)
        elif not os.path.exists(main_csv_path):
            raise FileNotFoundError(f"Main CSV file not found at {main_csv_path}")
        elif os.path.exists(raw_csv_path):
            print(f"CSV file already exists at {raw_csv_path}")

    @property
    def raw_file_names(self):
        return ['qm8.csv']

    @property
    def processed_file_names(self):
        suffix = ""
        if self.use_global_features:
            suffix += "_with_global_features"
        if self.use_3d_geo:
            suffix += "_with_3d_geo"
        return [f'data_{self.target_column}{suffix}.pt']

    @property
    def num_classes(self):
        return 1
    
    @property 
    def num_global_features(self):
        """
        Returns the number of global features if enabled.
        """
        return get_global_feature_dim() if self.use_global_features else 0
    
    def has_global_features(self):
        """
        Returns whether this dataset includes global features.
        """
        return self.use_global_features

    def get_target_statistics(self):
        if not hasattr(self, '_target_stats'):
            df = pd.read_csv(self.raw_paths[0])
            target_values = df[self.target_column].dropna()
            
            self._target_stats = {
                'mean': float(target_values.mean()),
                'std': float(target_values.std()),
                'min': float(target_values.min()),
                'max': float(target_values.max()),
                'count': len(target_values)
            }
        
        return self._target_stats

    def print_dataset_info(self):
        stats = self.get_target_statistics()
        print(f"\nQM8 Dataset Information:")
        print(f"Target column: {self.target_column}")
        print(f"Number of samples: {len(self)}")
        print(f"Target statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Count: {stats['count']}")
        
        sample = self.get(0)
        print(f"Node features: {sample.x.shape}")
        print(f"Edge features: {sample.edge_attr.shape}")


class QM8MultiTaskDataset(InMemoryDataset):
    def __init__(self, root, target_columns, use_global_features=False, use_3d_geo=False, transform=None, pre_transform=None):
        """
        Multi-task QM8 dataset for graph neural networks.
        
        Args:
            root (str): Root directory where the dataset should be saved.
            target_columns (list): List of target column names for multitask learning.
            use_global_features (bool): Whether to include global molecular features.
            use_3d_geo (bool): Whether to include 3D geometric edge features.
            transform: Optional transform to be applied to each data object.
            pre_transform: Optional pre-transform to be applied before saving.
        """
        self.target_columns = target_columns if isinstance(target_columns, list) else [target_columns]
        self.use_global_features = use_global_features
        self.use_3d_geo = use_3d_geo
        self.csv_file = "data/qm8.csv"
        
        # Define available target columns for QM8
        self.available_targets = [
            'E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
            'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
            'E1-PBE0.1', 'E2-PBE0.1', 'f1-PBE0.1', 'f2-PBE0.1',
            'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'
        ]
        
        # Validate target columns
        invalid_targets = [col for col in self.target_columns if col not in self.available_targets]
        if invalid_targets:
            raise ValueError(f"Invalid target columns: {invalid_targets}. "
                           f"Available targets: {self.available_targets}")
        
        # Initialize global feature extractor if needed
        if self.use_global_features:
            self.global_extractor = get_global_extractor()
            if self.global_extractor is None:
                raise ImportError(
                    "Global features requested but descriptastorus not available. "
                    "Please install descriptastorus: pip install descriptastorus"
                )
        
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["qm8.csv"]

    @property
    def processed_file_names(self):
        # Create a filename that represents all target columns
        targets_str = "_".join(self.target_columns)
        suffix = ""
        if self.use_global_features:
            suffix += "_with_global_features"
        if self.use_3d_geo:
            suffix += "_with_3d_geo"
        return [f"data_{targets_str}{suffix}.pt"]

    def download(self):
        # Copy the main CSV file to the raw directory if it doesn't exist
        raw_csv_path = self.raw_paths[0]
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        
        main_csv_path = os.path.join(os.path.dirname(os.path.dirname(self.root)), self.csv_file)
        
        if not os.path.exists(raw_csv_path) and os.path.exists(main_csv_path):
            print(f"Copying {main_csv_path} to {raw_csv_path}")
            shutil.copy2(main_csv_path, raw_csv_path)
        elif not os.path.exists(main_csv_path):
            raise FileNotFoundError(f"Main CSV file not found at {main_csv_path}")
        elif os.path.exists(raw_csv_path):
            print(f"CSV file already exists at {raw_csv_path}")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        global_features_count = 0
        
        print(f"Processing QM8 multi-task dataset with targets: {self.target_columns}")
        print(f"Total molecules to process: {len(df)}")
        print(f"Global features: {'enabled' if self.use_global_features else 'disabled'}")
        print(f"3D geometric features: {'enabled' if self.use_3d_geo else 'disabled'}")
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"Processed {idx}/{len(df)} molecules...")
                
            smiles = row['smiles']
            
            # Extract labels for all target columns
            labels = []
            valid_labels = True
            
            for col in self.target_columns:
                label = row[col]
                if pd.isnull(label):
                    valid_labels = False
                    break
                labels.append(float(label))
            
            # Skip this molecule if any label is missing
            if not valid_labels:
                invalid_count += 1
                continue
                
            try:
                data = smiles_to_data(smiles, labels=labels, use_3d_geo=self.use_3d_geo, dataset_type='QM8')
                if data is not None and hasattr(data, 'edge_index'):
                    # Extract global features if requested
                    if self.use_global_features:
                        global_features = self.global_extractor.extract_features(smiles)
                        if global_features is not None:
                            data.global_features = torch.tensor(global_features, dtype=torch.float32)
                            global_features_count += 1
                        else:
                            # Skip molecules where global features cannot be extracted
                            if idx % 10000 == 0:  # Only print occasionally to avoid spam
                                print(f"Skipped molecule due to failed global feature extraction: {smiles}")
                            invalid_count += 1
                            continue
                    
                    data_list.append(data)
                    valid_count += 1
                else:
                    print(f"Excluded invalid graph for SMILES: {smiles}")
                    invalid_count += 1
            except Exception as e:
                print(f"Error processing SMILES '{smiles}': {e}")
                invalid_count += 1
        
        print(f"Processed {valid_count} valid molecules, {invalid_count} invalid/missing")
        print(f"Target columns: {self.target_columns}")
        print(f"Number of tasks: {len(self.target_columns)}")
        if self.use_global_features:
            print(f"Successfully extracted global features for {global_features_count} molecules")
        
        if not data_list:
            raise ValueError("No valid data found! Check your target columns and data.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_target_columns(self):
        """Return the list of target columns used in this dataset."""
        return self.target_columns
    
    def get_num_tasks(self):
        """Return the number of tasks (target columns)."""
        return len(self.target_columns)
    
    @property 
    def num_global_features(self):
        """
        Returns the number of global features if enabled.
        """
        return get_global_feature_dim() if self.use_global_features else 0
    
    def has_global_features(self):
        """
        Returns whether this dataset includes global features.
        """
        return self.use_global_features

    def print_dataset_info(self):
        print(f"QM8 Multi-Task Dataset Info:")
        print(f"Number of molecules: {len(self)}")
        print(f"Target columns: {self.target_columns}")
        print(f"Number of tasks: {self.get_num_tasks()}")
        print(f"Number of node features: {self.num_node_features}")
        print(f"Number of edge features: {self.num_edge_features}")
        print(f"Global features: {'enabled' if self.use_global_features else 'disabled'}")
        if self.use_global_features:
            print(f"Number of global features: {self.num_global_features}")


def create_qm8_multitask_dataset(target_columns, dataset_root="dataset", name=None, use_global_features=False, use_3d_geo=False):
    """
    Utility function to create a multitask QM8 dataset.
    
    Args:
        target_columns (list): List of target column names.
        dataset_root (str): Root directory for datasets.
        name (str): Name for the multitask dataset directory. If None, auto-generated.
        use_global_features (bool): Whether to include global molecular features.
        use_3d_geo (bool): Whether to include 3D geometric edge features.
    
    Returns:
        QM8MultiTaskDataset: The created multitask dataset.
    """
    if name is None:
        # Create a short hash for target columns
        target_str = " ".join(sorted(target_columns))
        target_hash = str(sum(ord(c) for c in target_str) % 10**8)
        name = f"QM8_multitask_{target_hash}"
    root = os.path.join(dataset_root, name)
    return QM8MultiTaskDataset(root=root, target_columns=target_columns, use_global_features=use_global_features, use_3d_geo=use_3d_geo)


if __name__ == "__main__":
    dataset = QM8GraphDataset(root='./dataset/QM8')
    dataset.print_dataset_info()
    
    print("\n" + "="*50)
    dataset_homo = QM8GraphDataset(root='./dataset/QM8_HOMO', target_column='homo')
    dataset_homo.print_dataset_info()
    
    print(f"\nExample molecule (first sample):")
    sample = dataset.get(0)
    print(f"Number of atoms: {sample.x.shape[0]}")
    print(f"Number of edges: {sample.edge_index.shape[1]}")
    print(f"Target value ({dataset.target_column}): {sample.y.item():.4f}")
    
    # Test multi-task dataset
    print("\n" + "="*60)
    print("Testing QM8 Multi-Task Dataset:")
    multitask_targets = ['E1-CC2', 'E2-CC2', 'f1-CC2']
    multitask_dataset = create_qm8_multitask_dataset(multitask_targets)
    multitask_dataset.print_dataset_info()
    
    sample_mt = multitask_dataset.get(0)
    print(f"\nMulti-task sample:")
    print(f"Number of atoms: {sample_mt.x.shape[0]}")
    print(f"Target values shape: {sample_mt.y.shape}")
    print(f"Target values: {sample_mt.y}")
