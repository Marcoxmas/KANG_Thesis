import torch
from torch_geometric.data import InMemoryDataset
import pandas as pd
import numpy as np
import os
import shutil
from smiles_to_graph import smiles_to_data
from src.global_features import get_global_extractor, get_global_feature_dim

class ToxCastMultiTaskDataset(InMemoryDataset):
    def __init__(self, root, target_columns, use_global_features=False, transform=None, pre_transform=None):
        """
        Multi-task ToxCast dataset for graph neural networks.
        
        Args:
            root (str): Root directory where the dataset should be saved.
            target_columns (list): List of target column names for multitask learning.
            use_global_features (bool): Whether to include global molecular features.
            transform: Optional transform to be applied to each data object.
            pre_transform: Optional pre-transform to be applied before saving.
        """
        self.target_columns = target_columns if isinstance(target_columns, list) else [target_columns]
        self.use_global_features = use_global_features
        
        # Simple path like other datasets
        self.main_csv_path = "data/toxcast_data.csv"
        
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
        return ["toxcast_data.csv"]

    @property
    def processed_file_names(self):
        # Create a filename that represents all target columns
        targets_str = "_".join(self.target_columns)
        suffix = "_with_global_features" if self.use_global_features else ""
        return [f"data_{targets_str}{suffix}.pt"]

    def download(self):
        # Copy the main CSV file to the raw directory if it doesn't exist
        raw_csv_path = self.raw_paths[0]
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        
        main_csv_path = os.path.join(os.path.dirname(os.path.dirname(self.root)), self.main_csv_path)
        
        if not os.path.exists(raw_csv_path) and os.path.exists(main_csv_path):
            print(f"Copying {main_csv_path} to {raw_csv_path}")
            shutil.copy2(main_csv_path, raw_csv_path)
        elif not os.path.exists(main_csv_path):
            raise FileNotFoundError(f"Main CSV file not found at {main_csv_path}")
        elif os.path.exists(raw_csv_path):
            print(f"CSV file already exists at {raw_csv_path}")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        
        # Verify all target columns exist
        missing_columns = [col for col in self.target_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing target columns: {missing_columns}")
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        global_features_count = 0
        
        print(f"Processing ToxCast dataset with targets: {self.target_columns}")
        print(f"Global features: {'enabled' if self.use_global_features else 'disabled'}")
        
        for idx, row in df.iterrows():
            smiles = row["smiles"]
            
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
                data = smiles_to_data(smiles, labels=labels)
                if data is not None and hasattr(data, 'edge_index'):
                    # Extract global features if requested
                    if self.use_global_features:
                        global_features = self.global_extractor.extract_features(smiles)
                        if global_features is not None:
                            data.global_features = torch.tensor(global_features, dtype=torch.float32)
                            global_features_count += 1
                        else:
                            # Skip molecules where global features cannot be extracted
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
    
    @property
    def processed_file_names(self):
        """
        Returns the list of processed file names that must be present in the processed directory.
        """
        suffix = "_with_global_features" if self.use_global_features else ""
        return [f'data_{self.target_column}{suffix}.pt']


# Backward compatibility: keep the original single-task class
class ToxCastGraphDataset(InMemoryDataset):
    def __init__(self, root, target_column, use_global_features=False, transform=None, pre_transform=None):
        self.target_column = target_column
        self.target_columns = [target_column]  # Ensure target_columns is initialized as a list
        self.use_global_features = use_global_features
        
        # Simple path like other datasets
        self.main_csv_path = "data/toxcast_data.csv"
        
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
        return ["toxcast_data.csv"]

    @property
    def processed_file_names(self):
        suffix = "_with_global_features" if self.use_global_features else ""
        return [f"data_{self.target_column}{suffix}.pt"]

    def download(self):
        # Copy the main CSV file to the raw directory if it doesn't exist
        raw_csv_path = self.raw_paths[0]
        os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
        
        main_csv_path = os.path.join(os.path.dirname(os.path.dirname(self.root)), self.main_csv_path)
        
        if not os.path.exists(raw_csv_path) and os.path.exists(main_csv_path):
            print(f"Copying {main_csv_path} to {raw_csv_path}")
            shutil.copy2(main_csv_path, raw_csv_path)
        elif not os.path.exists(main_csv_path):
            raise FileNotFoundError(f"Main CSV file not found at {main_csv_path}")
        elif os.path.exists(raw_csv_path):
            print(f"CSV file already exists at {raw_csv_path}")

    def process(self):
        df = pd.read_csv(self.raw_paths[0])

        # Verify all target columns exist
        missing_columns = [col for col in self.target_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing target columns: {missing_columns}")

        data_list = []
        valid_count = 0
        invalid_count = 0
        global_features_count = 0
        
        print(f"Processing ToxCast dataset with target: {self.target_column}")
        print(f"Global features: {'enabled' if self.use_global_features else 'disabled'}")

        for idx, row in df.iterrows():
            smiles = row["smiles"]

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
                data = smiles_to_data(smiles, labels=labels)
                if data is not None and hasattr(data, 'edge_index'):
                    # Extract global features if requested
                    if self.use_global_features:
                        global_features = self.global_extractor.extract_features(smiles)
                        if global_features is not None:
                            data.global_features = torch.tensor(global_features, dtype=torch.float32)
                            global_features_count += 1
                        else:
                            # Skip molecules where global features cannot be extracted
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


def create_multitask_dataset(target_columns, dataset_root="dataset", name="TOXCAST_multitask", use_global_features=False):
    """
    Utility function to create a multitask ToxCast dataset.
    
    Args:
        target_columns (list): List of target column names.
        dataset_root (str): Root directory for datasets.
        name (str): Name for the multitask dataset directory.
        use_global_features (bool): Whether to include global molecular features.
    
    Returns:
        ToxCastMultiTaskDataset: The created multitask dataset.
    """
    root = os.path.join(dataset_root, name)
    return ToxCastMultiTaskDataset(root=root, target_columns=target_columns, use_global_features=use_global_features)


def get_available_assays(csv_path="data/toxcast_data.csv"):
    """
    Get list of available assay columns from the ToxCast CSV file.
    
    Args:
        csv_path (str): Path to the ToxCast CSV file.
    
    Returns:
        list: List of column names that could be used as targets.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return []
    
    df = pd.read_csv(csv_path)
    # Filter out SMILES column and identify potential target columns
    potential_targets = [col for col in df.columns if col != "smiles"]
    
    print(f"Found {len(potential_targets)} potential target columns")
    print("Sample columns:")
    for i, col in enumerate(potential_targets[:10]):
        non_null_count = df[col].notna().sum()
        print(f"  {col}: {non_null_count} non-null values")
    
    if len(potential_targets) > 10:
        print(f"  ... and {len(potential_targets) - 10} more columns")
    
    return potential_targets