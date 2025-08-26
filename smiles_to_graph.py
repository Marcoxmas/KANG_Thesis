import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit import RDLogger
import warnings
from geometric_features_3d import get_3d_coordinates, create_3d_edge_features

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')


def one_hot(value, choices):
    encoding = [0] * len(choices)
    if value in choices:
        encoding[choices.index(value)] = 1
    return encoding


def atom_features(atom):
    """Returns a rich atom feature vector."""
    features = []
    # 1. Atomic number (one-hot + unknown pad)
    atom_type_list = list(range(1, 101))  # Atomic numbers from 1 to 100
    features += one_hot(atom.GetAtomicNum(), atom_type_list)

    # 2. Degree (# bonds, one-hot + pad, 0-5)
    features += one_hot(atom.GetDegree(), list(range(6)))

    # 3. Formal charge (one-hot + pad, includes [-2, -1, 0, +1, +2])
    formal_charge_list = [-2, -1, 0, 1, 2]
    features += one_hot(atom.GetFormalCharge(), formal_charge_list)

    # 4. Chirality (one-hot + pad, based on RDKit ChiralType enum 0-3)
    chirality_list = list(range(4))  # 0-3 range
    features += one_hot(int(atom.GetChiralTag()), chirality_list)

    # 5. Num Hs (one-hot + pad, 0-4 bonded Hs)
    features += one_hot(atom.GetTotalNumHs(includeNeighbors=True), list(range(5)))

    # 6. Hybridization (one-hot + pad, SP, SP2, SP3, etc.)
    hybridization_list = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features += one_hot(atom.GetHybridization(), hybridization_list)

    # 7. Aromaticity (binary, a.GetIsAromatic() → 0 or 1)
    features.append(int(atom.GetIsAromatic()))

    # 8. Mass (scaled float, 0.01 * a.GetMass() — real-valued feature)
    atomic_mass = 0.01 * atom.GetMass()
    features.append(atomic_mass)

    feat_tensor = torch.tensor(features, dtype=torch.float)
    return feat_tensor


def bond_features(bond):
    """Returns a rich bond feature vector."""
    features = []
    
    # 1. Nullity (binary, 1 bit) - set to 1 if bond is None
    if bond is None:
        features.append(1)
        # For None bonds, pad with zeros for all other features
        features += [0] * 4  # Bond type (4 bits)
        features.append(0)   # Conjugated (1 bit)
        features.append(0)   # In ring (1 bit)
        features += [0] * 7  # Stereo (7 bits)
        return torch.tensor(features, dtype=torch.float)
    else:
        features.append(0)
    
    # 2. Bond type (one-hot, 4 bits) - SINGLE, DOUBLE, TRIPLE, AROMATIC
    bond_type_list = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    features += one_hot(bond.GetBondType(), bond_type_list)
    
    # 3. Conjugated (binary, 1 bit) - b.GetIsConjugated()
    features.append(int(bond.GetIsConjugated()))
    
    # 4. In ring (binary, 1 bit) - b.IsInRing()
    features.append(int(bond.IsInRing()))
    
    # 5. Stereo (one-hot + pad, 7 bits) - stereo codes 0-5 + "unknown"
    stereo_list = list(range(6))  # 0-5 range + unknown pad
    stereo_value = int(bond.GetStereo())
    features += one_hot(stereo_value, stereo_list)
    
    return torch.tensor(features, dtype=torch.float)


def bond_to_edge_index(mol, smiles=None):
    """Returns edge_index and edge_attr for a molecule."""
    edge_index = [[], []]
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # Add both directions (undirected graph)
        edge_index[0] += [i, j]
        edge_index[1] += [j, i]
        edge_attr += [bf, bf]
    if not edge_attr:
        # Handle molecules with no bonds (single atoms)
        if smiles:
            print(f"Molecule with no bonds: {smiles}")
        return None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr)
    return edge_index, edge_attr


def smiles_to_data(smiles: str, labels=None, include_hydrogens: bool = True, 
                   use_3d_geo: bool = False, cutoff: float = 4.0, num_rbf: int = 16, 
                   n_fourier: int = 2, dataset_type=None, mol_index=None, max_k_for_angles: int = 4) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles (str): SMILES string.
        labels (float, list, torch.Tensor, or None): Single label, list of labels, or tensor for multitask learning.
        include_hydrogens (bool): Whether to add explicit hydrogens to the molecule.
        use_3d_geo (bool): Whether to use 3D geometric edge features.
        cutoff (float): Distance cutoff for 3D edges.
        num_rbf (int): Number of RBF basis functions.
        n_fourier (int): Number of Fourier components for angle features.
        dataset_type (str): Dataset type (used for cache differentiation).
        mol_index (int): Molecule index (used for cache differentiation, but no longer for SDF lookup).
        
    Returns:
        torch_geometric.data.Data: Graph object or None if parsing fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hydrogens:
        mol = Chem.AddHs(mol)

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()])

    if use_3d_geo:
        # Get 3D coordinates using the new direct RDKit approach
        pos = get_3d_coordinates(smiles, dataset_type=dataset_type, seed=42, 
                               include_hydrogens=include_hydrogens, mol_index=mol_index)
        
        use_fake_3d = False
        if pos is None:
            print(f"Failed to obtain 3D coordinates for SMILES: {smiles}, using fake 3D mode")
            # Create fake coordinates at origin to maintain consistent dimensions
            pos = torch.zeros(mol.GetNumAtoms(), 3).numpy()
            use_fake_3d = True
        elif pos.shape[0] != sum(1 for atom in mol.GetAtoms()):
            print(f"Coordinate count mismatch for {smiles}: got {pos.shape[0]} coords for {mol.GetNumAtoms()} atoms (including hydrogens: {include_hydrogens}), using fake 3D mode")
            # Create fake coordinates at origin to maintain consistent dimensions
            pos = torch.zeros(mol.GetNumAtoms(), 3).numpy()
            use_fake_3d = True

    if use_3d_geo:
        # Get bond-based edge features for reference
        bond_edge_index, bond_edge_attr = bond_to_edge_index(mol, smiles=smiles)
        
        if use_fake_3d:
            # Don't waste computation - directly create features with zero RBF and angle parts
            if bond_edge_index is None or bond_edge_attr is None:
                # Handle molecules with no bonds
                if mol.GetNumAtoms() == 1:
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    # Create features: zero RBF + zero bond + zero angle
                    edge_attr = torch.zeros(1, num_rbf + 13 + 2 * n_fourier)
                else:
                    print(f"Excluded molecule with no bonds for SMILES: {smiles}")
                    return None
            else:
                edge_index = bond_edge_index
                n_edges = edge_index.shape[1]
                # Create features: zero RBF + real bond + zero angle
                zero_rbf = torch.zeros(n_edges, num_rbf)
                zero_angle = torch.zeros(n_edges, 2 * n_fourier)
                edge_attr = torch.cat([zero_rbf, bond_edge_attr, zero_angle], dim=1)
        else:
            # Use real 3D geometric edge features with full computation
            edge_index, edge_attr = create_3d_edge_features(
                pos, bond_edge_index, bond_edge_attr, cutoff, num_rbf, n_fourier, max_k_for_angles
            )
        
        if edge_index is None or edge_attr is None or edge_index.numel() == 0 or x.numel() == 0:
            print(f"Excluded invalid or empty 3D graph for SMILES: {smiles}")
            return None
        
        # Add 3D coordinates to data (real or fake)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=torch.tensor(pos, dtype=torch.float32))
    else:
        # Use traditional 2D bond-based features
        edge_index, edge_attr = bond_to_edge_index(mol, smiles=smiles)
        
        # Handle molecules with no bonds (e.g., single atoms)
        if edge_index is None or edge_attr is None:
            if mol.GetNumAtoms() == 1:
                # Single atom: create self-loop
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.zeros(1, 13)  # Default bond feature size
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            else:
                print(f"Excluded molecule with no bonds for SMILES: {smiles}")
                return None
        elif edge_index.numel() == 0 or x.numel() == 0:
            print(f"Excluded invalid or empty graph for SMILES: {smiles}")
            return None
        else:
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    # Add labels
    if labels is not None:
        if isinstance(labels, (list, tuple)):
            data.y = torch.tensor(labels, dtype=torch.float)
        elif isinstance(labels, torch.Tensor):
            data.y = labels.float()
        else:
            # Single label (backward compatibility)
            data.y = torch.tensor([labels], dtype=torch.float)

    return data


def print_graph_info(data: Data):
    """Print comprehensive information about a graph data object."""
    print("Graph Information:")
    print(f" - Number of nodes: {data.num_nodes}")
    print(f" - Node features shape: {data.x.shape}")
    print(f" - Edge index shape: {data.edge_index.shape}")
    print(f" - Number of edges: {data.num_edges}")
    print(f" - Edge attributes shape: {data.edge_attr.shape}")
    if hasattr(data, 'pos') and data.pos is not None:
        print(f" - 3D coordinates shape: {data.pos.shape}")
        print(f" - Using 3D geometric features")
    else:
        print(" - Using 2D bond-based features")
    if hasattr(data, 'y'):
        print(f" - Label: {data.y}")
    else:
        print(" - No label assigned.")
    print(f" - Edge feature breakdown:")
    if hasattr(data, 'pos') and data.pos is not None:
        # 3D case: RBF + bond + angle features
        edge_dim = data.edge_attr.shape[1]
        bond_dim = 13
        print(f"   - Total edge features: {edge_dim}")
        print(f"   - RBF features: first {data.num_rbf} dimensions (default)")
        print(f"   - Bond features: next {bond_dim} dimensions")
        print(f"   - Angle features: last {2*data.n_fourier} dimensions (2*n_fourier, default)")
    else:
        # 2D case: bond features only
        print(f"   - Bond features: {data.edge_attr.shape[1]} dimensions")
