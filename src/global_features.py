"""
Global Molecular Features Extraction using RDKit descriptors via descriptastorus.

This module provides functionality to extract 200 normalized molecular descriptors
from SMILES strings, following the approach used in the D-MPNN (Chemprop) paper.

The descriptors are pre-normalized using empirical CDFs fitted on a large collection
of molecules, ensuring all features are in the [0, 1] range and distribution-aware.
"""

import numpy as np
import torch
from typing import Optional, Union, List
import warnings

try:
    from descriptastorus.descriptors import rdNormalizedDescriptors
    DESCRIPTASTORUS_AVAILABLE = True
except ImportError:
    DESCRIPTASTORUS_AVAILABLE = False
    rdNormalizedDescriptors = None


class GlobalFeatureExtractor:
    """
    Extracts normalized global molecular features from SMILES strings using descriptastorus.
    
    This class provides a clean interface to extract the same 200 RDKit descriptors
    used in the D-MPNN paper, with pre-computed CDF normalization.
    """
    
    def __init__(self):
        """Initialize the feature extractor."""
        if not DESCRIPTASTORUS_AVAILABLE:
            raise ImportError(
                "descriptastorus is not installed. Please install it using:\n"
                "pip install descriptastorus\n"
                "or add it to your requirements.txt file."
            )
        
        # Initialize the RDKit normalized descriptor generator
        self.generator = rdNormalizedDescriptors.RDKit2DNormalized()
        self.feature_names = None
        self._initialize_feature_names()
    
    def _initialize_feature_names(self):
        """Initialize feature names for debugging and analysis."""
        try:
            # Get feature names from the generator
            self.feature_names = self.generator.columns
        except Exception as e:
            warnings.warn(f"Could not retrieve feature names: {e}")
            self.feature_names = [f"descriptor_{i}" for i in range(200)]
    
    def extract_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extract global molecular features from a SMILES string.
        
        Args:
            smiles (str): SMILES string representing the molecule
            
        Returns:
            Optional[np.ndarray]: Array of 200 normalized features in [0, 1] range,
                                 or None if extraction failed
        """
        try:
            # Process the SMILES string - returns [smiles, feature1, feature2, ..., feature200]
            result = self.generator.process(smiles)
            
            if result is None or len(result) < 201:
                return None
            
            # Extract features (skip the first element which is the SMILES string)
            features = np.array(result[1:], dtype=np.float32)
            
            # Validate features
            if len(features) != 200:
                warnings.warn(f"Expected 200 features, got {len(features)}")
                return None
            
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return None
            
            return features
            
        except Exception as e:
            warnings.warn(f"Failed to extract features for SMILES '{smiles}': {e}")
            return None
    
    def extract_features_batch(self, smiles_list: List[str]) -> List[Optional[np.ndarray]]:
        """
        Extract global molecular features for a batch of SMILES strings.
        
        Args:
            smiles_list (List[str]): List of SMILES strings
            
        Returns:
            List[Optional[np.ndarray]]: List of feature arrays, with None for failed extractions
        """
        results = []
        for smiles in smiles_list:
            features = self.extract_features(smiles)
            results.append(features)
        return results
    
    def get_feature_dim(self) -> int:
        """
        Get the dimensionality of the global features.
        
        Returns:
            int: Number of global features (should be 200)
        """
        return 200
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the global features.
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names.copy() if self.feature_names else []


def extract_global_features(smiles: str) -> Optional[torch.Tensor]:
    """
    Convenience function to extract global features and return as PyTorch tensor.
    
    Args:
        smiles (str): SMILES string representing the molecule
        
    Returns:
        Optional[torch.Tensor]: Tensor of shape (200,) with normalized features,
                               or None if extraction failed
    """
    if not DESCRIPTASTORUS_AVAILABLE:
        warnings.warn("descriptastorus not available, returning None")
        return None
    
    extractor = GlobalFeatureExtractor()
    features = extractor.extract_features(smiles)
    
    if features is not None:
        return torch.tensor(features, dtype=torch.float32)
    return None


def get_global_feature_dim() -> int:
    """
    Get the dimensionality of global features.
    
    Returns:
        int: Number of global features (200)
    """
    return 200


# Global instance for reuse
_global_extractor = None

def get_global_extractor() -> Optional[GlobalFeatureExtractor]:
    """
    Get a shared global feature extractor instance.
    
    Returns:
        Optional[GlobalFeatureExtractor]: Shared extractor instance, or None if not available
    """
    global _global_extractor
    
    if not DESCRIPTASTORUS_AVAILABLE:
        return None
    
    if _global_extractor is None:
        try:
            _global_extractor = GlobalFeatureExtractor()
        except Exception:
            return None
    
    return _global_extractor
