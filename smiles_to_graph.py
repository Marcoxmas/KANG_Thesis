import torch
from torch_geometric.data import Data
from rdkit import Chem


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
        print(f"Skipped molecule with no bonds: {smiles}")
        return None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.stack(edge_attr)
    return edge_index, edge_attr


def smiles_to_data(smiles: str, labels=None, include_hydrogens: bool = False) -> Data:
    """
    Converts a SMILES string to a PyTorch Geometric Data object.
    
    Args:
        smiles (str): SMILES string.
        labels (float, list, torch.Tensor, or None): Single label, list of labels, or tensor for multitask learning.
        include_hydrogens (bool): Whether to add explicit hydrogens to the molecule.
        
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

    # Edge index and edge attributes
    edge_index, edge_attr = bond_to_edge_index(mol, smiles=smiles)
    if edge_index is None or edge_attr is None or edge_index.numel() == 0 or x.numel() == 0:
        print(f"Excluded invalid or empty graph for SMILES: {smiles}")
        return None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
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
    print("Graph Information:")
    print(f" - Number of nodes: {data.num_nodes}")
    print(f" - Node features:\n{data.x}")
    print(f" - Edge index:\n{data.edge_index}")
    print(f" - Number of edges: {data.num_edges}")
    print(f" - Edge attributes:\n{data.edge_attr}")
    if hasattr(data, 'y'):
        print(f" - Label: {data.y}")
    else:
        print(" - No label assigned.")