import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import os
import shutil
from smiles_to_graph import smiles_to_data
from src.global_features import get_global_extractor, get_global_feature_dim

class HIVGraphDataset(InMemoryDataset):
    def __init__(self, root, use_global_features=False, transform=None, pre_transform=None):
        self.csv_file = "data/HIV.csv"
        self.use_global_features = use_global_features
        
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
        # Use the parent class method which properly handles slicing
        return super().get(idx)

    def process(self):
        """
        Processes the raw data into PyTorch Geometric Data objects and saves them.
        """
        df = pd.read_csv(self.raw_paths[0])
        
        data_list = []
        valid_count = 0
        invalid_count = 0
        global_features_count = 0
        
        print(f"Processing HIV dataset with {'global features enabled' if self.use_global_features else 'global features disabled'}")
        
        for idx, row in df.iterrows():
            smiles = row['smiles']
            label = row['HIV_active']  # Use HIV_active column as the label
            
            # Skip if label is missing
            if pd.isnull(label):
                invalid_count += 1
                continue
                
            try:
                data = smiles_to_data(smiles, labels=int(label))
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
        if self.use_global_features:
            print(f"Successfully extracted global features for {global_features_count} molecules")
        
        if not data_list:
            raise ValueError("No valid data found! Check your data.")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def download(self):
        """
        Downloads the dataset if it does not exist.
        """
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

    @property
    def raw_file_names(self):
        """
        Returns the list of raw file names that must be present in the raw directory.
        """
        return ['HIV.csv']

    @property
    def processed_file_names(self):
        """
        Returns the list of processed file names that must be present in the processed directory.
        """
        suffix = "_with_global_features" if self.use_global_features else ""
        return [f'data{suffix}.pt']

    @property
    def num_classes(self):
        """
        Returns the number of classes in the dataset.
        """
        return 2  # Binary classification for HIV activity
    
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

# Example usage
if __name__ == "__main__":
    # Test without global features (default)
    print("=== Testing without global features ===")
    dataset = HIVGraphDataset(root="dataset/HIV")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of global features: {dataset.num_global_features}")
    print(f"Has global features: {dataset.has_global_features()}")
    
    for i in range(min(2, len(dataset))):
        data = dataset[i]
        print(f"Sample {i}: {data}")
        print(f"  Has global_features attr: {hasattr(data, 'global_features')}")
    
    # Test with global features
    print("\n=== Testing with global features ===")
    try:
        dataset_with_global = HIVGraphDataset(root="dataset/HIV_global", use_global_features=True)
        print(f"Dataset size: {len(dataset_with_global)}")
        print(f"Number of classes: {dataset_with_global.num_classes}")
        print(f"Number of node features: {dataset_with_global.num_node_features}")
        print(f"Number of global features: {dataset_with_global.num_global_features}")
        print(f"Has global features: {dataset_with_global.has_global_features()}")
        
        for i in range(min(2, len(dataset_with_global))):
            data = dataset_with_global[i]
            print(f"Sample {i}: {data}")
            if hasattr(data, 'global_features'):
                print(f"  Global features shape: {data.global_features.shape}")
                print(f"  Global features range: [{data.global_features.min():.4f}, {data.global_features.max():.4f}]")
    except ImportError as e:
        print(f"Could not test global features: {e}")
