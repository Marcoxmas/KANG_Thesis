"""
3D Geometric Features Module for KANG Framework

This module provides functionality to compute 3D geometric edge features including:
- Radial Basis Function (RBF) distance encoding
- Angle summary features using Fourier encoding  
- Torsion (dihedral) angle features using Fourier encoding

The edge attributes are constructed as:
edge_attr[j→i] = [ RBF(d_ji)  ||  bond_bits_or_zero  ||  angle_summary(j,i)  ||  torsion_summary(j,i) ]
"""

import random
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger
import requests
import time
from urllib.parse import quote
import pickle
import hashlib
import pandas as pd
from torch_cluster import radius_graph

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import AllChem
import numpy as np
import os
import warnings
from pathlib import Path


class RadialBasisFunction(nn.Module):
    """
    Radial Basis Function for distance encoding in 3D molecular graphs.
    """
    def __init__(
        self,
        grid_min: float = -1.,
        grid_max: float = 1.,
        num_grids: int = 8,
        denominator: float = None,  # larger denominators lead to smoother basis
        linspace: bool = False,
        trainable_grid: bool = True
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids

        if linspace:
            grid = torch.linspace(grid_min, grid_max, num_grids)
        else:
            # Sample control points from a Gaussian distribution with mean 0 and std 1
            grid = torch.randn(num_grids)  # Sample from N(0, 1)
            
            # Scale and shift the sampled points to fit within [grid_min, grid_max]
            grid = torch.clamp(grid, grid_min, grid_max)
            
            # Sort the grid to ensure it's in ascending order
            grid, _ = torch.sort(grid)

        self.grid = torch.nn.Parameter(grid, requires_grad=trainable_grid)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


# Global cache directory path (computed once)
_CACHE_DIR = None

def _get_cache_dir():
    """Get or create the cache directory for 3D coordinates."""
    global _CACHE_DIR
    if _CACHE_DIR is None:
        data_dir = get_data_path()
        cache_dir = data_dir / "3d_coords_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        _CACHE_DIR = cache_dir
    return _CACHE_DIR

def get_data_path():
    """Get the path to the data directory relative to the current module."""
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data"
    return data_dir


def get_cache_key(smiles, include_hydrogens, seed):
    # Use MD5 hash to avoid illegal filename characters
    base = f"{smiles}_{include_hydrogens}_{seed}"
    safe_hash = hashlib.md5(base.encode("utf-8")).hexdigest()
    return safe_hash


def load_from_cache(smiles, include_hydrogens=True, seed=42):
    """Load 3D coordinates from cache if available (optimized for speed)."""
    try:
        cache_dir = _get_cache_dir()
        cache_key = get_cache_key(smiles, include_hydrogens, seed)
        cache_file = cache_dir / f"{cache_key}.pkl"
        # print(f"Looking for cache file: {cache_file}")
        if cache_file.exists():
            # print(f"Cache file found: {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                # Handle both old format (dict) and new format (direct numpy array)
                if isinstance(cached_data, dict):
                    return cached_data['coordinates']
                else:
                    return cached_data  # Direct numpy array
        # print(f"Cache file not found: {cache_file}")
    except Exception as e:
        print(f"Error loading from cache: {e}")
    
    return None


def save_to_cache(smiles, coordinates, include_hydrogens=True, seed=42):
    """Save 3D coordinates to cache (optimized for speed)."""
    try:
        cache_dir = _get_cache_dir()
        cache_key = get_cache_key(smiles, include_hydrogens, seed)
        cache_file = cache_dir / f"{cache_key}.pkl"
        
        # Save coordinates directly (no dictionary overhead)
        with open(cache_file, 'wb') as f:
            pickle.dump(coordinates, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"Saved 3D coordinates to cache for {smiles}")
    except Exception as e:
        print(f"Error saving to cache: {e}")


def generate_3d_conformer(smiles, seed=42, include_hydrogens=True):
    """
    Generate a 3D conformer using ETKDGv3 + MMFF optimization.
    This is the legacy single-attempt version, kept for backward compatibility.
    
    Args:
        smiles (str): SMILES string
        seed (int): Random seed for determinism
        include_hydrogens (bool): Whether to add explicit hydrogens
        
    Returns:
        numpy.ndarray or None: 3D coordinates if successful, None otherwise
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        if include_hydrogens:
            mol = Chem.AddHs(mol)
        
        # Handle single atom case
        if mol.GetNumAtoms() == 1:
            # Return origin coordinates for single atom
            return np.array([[0.0, 0.0, 0.0]])
        
        # ETKDGv3 conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        
        result = AllChem.EmbedMolecule(mol, params)
        if result != 0:
            return None
        
        # MMFF optimization
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            # If MMFF fails, continue with unoptimized coordinates
            pass
        
        conf = mol.GetConformer()
        pos = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return pos
        
    except Exception as e:
        warnings.warn(f"3D conformer generation failed for SMILES {smiles}: {e}")
        return None


def compute_distances(pos):
    """
    Compute pairwise distances between atoms.
    
    Args:
        pos (numpy.ndarray): 3D coordinates of shape (n_atoms, 3)
        
    Returns:
        numpy.ndarray: Distance matrix of shape (n_atoms, n_atoms)
    """
    pos = np.array(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances

# Legacy function
def compute_angle_features(pos, edge_index, cutoff, n_fourier=4):
    """
    Compute angle summary features for directed edges.
    
    For each directed edge j→i, collect incoming edges to j: those with t==j (k→j), excluding k==i.
    For each k, compute the angle at center j:
    - u_kj = unit_vector(j - k)
    - u_ji = unit_vector(i - j)  
    - a_kji = arccos(clamp(dot(u_kj, u_ji), -1+1e-7, 1-1e-7))
    
    Encode with Fourier bank for n=1..n_fourier:
    - f(a_kji) = [sin(n*a_kji), cos(n*a_kji)]_n

    Aggregate = mean over k (if no k, use zeros).
    
    Args:
        pos (numpy.ndarray): 3D coordinates of shape (n_atoms, 3)
        edge_index (torch.Tensor): Edge indices of shape (2, n_edges)
        cutoff (float): Distance cutoff for edges
        n_fourier (int): Number of Fourier components
        
    Returns:
        torch.Tensor: Angle features of shape (n_edges, 2*n_fourier)
    """
    n_edges = edge_index.shape[1]
    angle_features = torch.zeros(n_edges, 2 * n_fourier)
    
    if len(pos) < 3:
        # Need at least 3 atoms to compute angles
        return angle_features
    
    distances = compute_distances(pos)
    
    # Create adjacency information for efficient lookup
    adj_list = {}
    for i in range(len(pos)):
        adj_list[i] = []
    
    # Build adjacency list from distance-based edges
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and distances[i, j] <= cutoff:
                adj_list[i].append(j)
    
    for edge_idx in range(n_edges):
        j, i = edge_index[0, edge_idx].item(), edge_index[1, edge_idx].item()
        
        # Find all neighbors k of j, excluding i
        neighbors_k = [k for k in adj_list[j] if k != i]
        
        if len(neighbors_k) == 0:
            # No neighbors, features remain zeros
            continue
        
        angles = []
        for k in neighbors_k:
            # Compute vectors
            u_kj = pos[j] - pos[k]
            u_ji = pos[i] - pos[j]
            
            # Normalize vectors
            u_kj_norm = np.linalg.norm(u_kj)
            u_ji_norm = np.linalg.norm(u_ji)
            
            if u_kj_norm > 1e-7 and u_ji_norm > 1e-7:
                u_kj = u_kj / u_kj_norm
                u_ji = u_ji / u_ji_norm
                
                # Compute angle
                dot_product = np.dot(u_kj, u_ji)
                dot_product = np.clip(dot_product, -1 + 1e-7, 1 - 1e-7)
                angle = np.arccos(dot_product)
                angles.append(angle)
        
        if len(angles) > 0:
            # Compute Fourier features and average
            fourier_features = []
            for angle in angles:
                fourier_feat = []
                for n in range(1, n_fourier + 1):
                    fourier_feat.extend([np.sin(n * angle), np.cos(n * angle)])
                fourier_features.append(fourier_feat)
            
            # Average over all angles
            mean_fourier = np.mean(fourier_features, axis=0)
            angle_features[edge_idx] = torch.tensor(mean_fourier, dtype=torch.float32)
    
    return angle_features


def compute_angle_features_from_graph(pos_t: torch.Tensor,
                                      edge_index: torch.Tensor,
                                      n_fourier: int = 2,
                                      max_k_for_angles: int = 8,
                                      debug_timing: bool = False):
    """
    Vectorized, center-batched angle features.
    For each center j, compute u_kj once for all its in-neighbors k,
    then reuse it for all outgoing edges j->i in one matmul.
    """
    import time
    t0 = time.perf_counter()

    E = edge_index.shape[1]
    feat_dim = 2 * n_fourier
    out = torch.zeros(E, feat_dim, dtype=torch.float32)
    if n_fourier <= 0 or pos_t.size(0) < 3 or E == 0:
        if debug_timing:
            print(f"[ANG-BATCH] trivial_return={time.perf_counter()-t0:.4f}s")
        return out

    src, dst = edge_index[0], edge_index[1]  # edge j->i: src=j, dst=i
    N = pos_t.size(0)

    # Map center j -> list of edges e (edges j->i)
    j_to_edges = [[] for _ in range(N)]
    for e in range(E):
        j = src[e].item()
        j_to_edges[j].append(e)

    # In-neighbors of each j (k->j)
    in_neighbors = [[] for _ in range(N)]
    for e in range(E):
        k = src[e].item()
        j = dst[e].item()
        in_neighbors[j].append(k)

    t1 = time.perf_counter()

    # Precompute u_ji per edge once
    u_ji = pos_t[dst] - pos_t[src]  # [E, 3]
    norms_ji = torch.linalg.norm(u_ji, dim=1)
    good_edge = norms_ji > 1e-7
    u_ji[good_edge] = u_ji[good_edge] / norms_ji[good_edge].unsqueeze(1)

    t2 = time.perf_counter()

    total_pairs = 0
    # Process per center j
    for j in range(N):
        edges_e = j_to_edges[j]
        if not edges_e:
            continue
        neigh = in_neighbors[j]
        if not neigh:
            continue

        neigh_t = torch.tensor(neigh, dtype=torch.long)
        # u_kj for all neighbors once
        kj = pos_t[j].unsqueeze(0) - pos_t[neigh_t]     # [K, 3]
        kj_norm = torch.linalg.norm(kj, dim=1)
        valid_k = kj_norm > 1e-7
        if not torch.any(valid_k):
            continue
        kj = kj[valid_k] / kj_norm[valid_k].unsqueeze(1)  # [Kvalid, 3]
        valid_neigh = neigh_t[valid_k]                    # [Kvalid]

        # Optional neighbor cap for speed
        if kj.size(0) > max_k_for_angles:
            sel = torch.randperm(kj.size(0))[:max_k_for_angles]
            kj = kj[sel]
            valid_neigh = valid_neigh[sel]

        e_tensor = torch.tensor(edges_e, dtype=torch.long)
        u_ji_block = u_ji[e_tensor]  # [Ej, 3]
        mask_good = good_edge[e_tensor]
        if not torch.any(mask_good):
            continue

        # Dot products for all (k, edges) in one shot
        # kj: [K,3], u_ji_block.T: [3,Ej] -> dots: [K,Ej]
        dots = kj @ u_ji_block.T
        dots = torch.clamp(dots, -1.0 + 1e-7, 1.0 - 1e-7)

        # Mask k==i for each column
        i_idx = dst[e_tensor]                                # [Ej]
        eq = (valid_neigh.unsqueeze(1) == i_idx.unsqueeze(0))  # [K,Ej]
        dots = dots.masked_fill(eq, float('nan'))

        angles = torch.arccos(dots)  # [K,Ej] with NaNs masked

        # Build [sin(n*angles), cos(n*angles)] without Python loops over k
        # angles: [K,Ej], ns: [n], -> angles[...,None]*ns: [K,Ej,n]
        ns = torch.arange(1, n_fourier + 1, device=angles.device, dtype=angles.dtype)
        na = angles.unsqueeze(-1) * ns  # [K,Ej,n]
        s = torch.sin(na)               # [K,Ej,n]
        c = torch.cos(na)               # [K,Ej,n]

        # nanmean over K (dim=0)
        valid_s = ~torch.isnan(s)
        valid_c = ~torch.isnan(c)
        sum_s = torch.where(valid_s, s, 0.0).sum(dim=0)          # [Ej,n]
        sum_c = torch.where(valid_c, c, 0.0).sum(dim=0)          # [Ej,n]
        cnt_s = valid_s.sum(dim=0).clamp_min(1)                  # [Ej,n]
        cnt_c = valid_c.sum(dim=0).clamp_min(1)                  # [Ej,n]
        mean_s = sum_s / cnt_s
        mean_c = sum_c / cnt_c

        # Write results back to out for these edges
        # Interleave s,c along last dim: [Ej, 2n]
        inter = torch.empty((mean_s.size(0), 2 * n_fourier), dtype=mean_s.dtype)
        inter[:, 0::2] = mean_s
        inter[:, 1::2] = mean_c
        out[e_tensor] = inter

        total_pairs += torch.isfinite(angles).sum().item()

    t3 = time.perf_counter()
    if debug_timing:
        print(f"[ANG-BATCH] build_in={t1-t0:.4f}s pre_uji={t2-t1:.4f}s "
              f"by_center={t3-t2:.4f}s pairs={total_pairs}")
    return out


def compute_torsion_features_from_graph(pos_t: torch.Tensor,
                                        edge_index: torch.Tensor,
                                        n_fourier: int = 2,
                                        max_k_for_torsions: int = 8,
                                        debug_timing: bool = False):
    """
    Optimized vectorized torsion angle (dihedral angle) features for edges.
    For each edge j->i, find all possible torsion paths k->j->i->l
    and compute dihedral angles using the four atoms k,j,i,l.
    
    This version is optimized for performance by:
    1. Minimizing CPU-GPU synchronization points
    2. Using vectorized operations where possible
    3. Early termination for degenerate cases
    
    Args:
        pos_t (torch.Tensor): 3D coordinates [N, 3]
        edge_index (torch.Tensor): Edge indices [2, E] 
        n_fourier (int): Number of Fourier components
        max_k_for_torsions (int): Maximum number of torsions per edge for performance
        debug_timing (bool): Print timing information
        
    Returns:
        torch.Tensor: Torsion features [E, 2*n_fourier]
    """
    import time
    t0 = time.perf_counter()
    
    E = edge_index.shape[1]
    feat_dim = 2 * n_fourier
    out = torch.zeros(E, feat_dim, dtype=torch.float32, device=pos_t.device)
    
    if n_fourier <= 0 or pos_t.size(0) < 4 or E == 0:
        if debug_timing:
            print(f"[TORSION-BATCH] trivial_return={time.perf_counter()-t0:.4f}s")
        return out
    
    # Early exit for very small molecules (< 4 atoms can't have proper torsions)
    N = pos_t.size(0)
    if N < 4:
        if debug_timing:
            print(f"[TORSION-BATCH] too_few_atoms={time.perf_counter()-t0:.4f}s")
        return out
    
    src, dst = edge_index[0], edge_index[1]  # edge j->i: src=j, dst=i
    
    # Convert to CPU numpy for adjacency building (more efficient for sparse operations)
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    
    # Build adjacency lists more efficiently
    in_neighbors = [[] for _ in range(N)]
    out_neighbors = [[] for _ in range(N)]
    
    for e in range(E):
        j, i = src_np[e], dst_np[e]
        out_neighbors[j].append(i)
        in_neighbors[i].append(j)
    
    t1 = time.perf_counter()
    
    total_torsions = 0
    processed_edges = 0
    
    # Process edges in batches to reduce computation
    batch_size = min(32, E)  # Process edges in smaller batches
    
    for batch_start in range(0, E, batch_size):
        batch_end = min(batch_start + batch_size, E)
        
        for e in range(batch_start, batch_end):
            j, i = src_np[e], dst_np[e]
            
            # Find k nodes connected to j (excluding i)
            k_candidates = [k for k in in_neighbors[j] if k != i]
            # Find l nodes connected to i (excluding j)  
            l_candidates = [l for l in out_neighbors[i] if l != j]
            
            if not k_candidates or not l_candidates:
                continue
            
            # Limit candidates for performance - use deterministic selection instead of random
            max_k = max_k_for_torsions // 2
            if len(k_candidates) > max_k:
                k_candidates = k_candidates[:max_k]  # Take first N instead of random sample
            if len(l_candidates) > max_k:
                l_candidates = l_candidates[:max_k]
            
            # Early skip if too few candidates
            if len(k_candidates) == 0 or len(l_candidates) == 0:
                continue
                
            # Get positions for the central bond
            pos_j = pos_t[j]  # [3]
            pos_i = pos_t[i]  # [3]
            v2 = pos_i - pos_j  # central bond j->i
            v2_norm = torch.linalg.norm(v2)
            
            # Skip if central bond is degenerate
            if v2_norm < 1e-7:
                continue
                
            v2_unit = v2 / v2_norm
            
            # Collect all valid torsion angles for this edge
            valid_torsions = []
            
            # Vectorize the inner computation where possible
            for k in k_candidates:
                pos_k = pos_t[k]
                v1 = pos_j - pos_k  # k->j
                v1_norm = torch.linalg.norm(v1)
                
                if v1_norm < 1e-7:
                    continue
                
                for l in l_candidates:
                    pos_l = pos_t[l]
                    v3 = pos_l - pos_i  # i->l
                    v3_norm = torch.linalg.norm(v3)
                    
                    if v3_norm < 1e-7:
                        continue
                    
                    # Compute normal vectors to the planes using cross products
                    n1 = torch.cross(v1, v2)  # normal to plane k-j-i
                    n2 = torch.cross(v2, v3)  # normal to plane j-i-l
                    
                    # Check for degenerate cases (colinear points)
                    n1_norm = torch.linalg.norm(n1)
                    n2_norm = torch.linalg.norm(n2)
                    
                    if n1_norm > 1e-7 and n2_norm > 1e-7:
                        n1_unit = n1 / n1_norm
                        n2_unit = n2 / n2_norm
                        
                        # Compute dihedral angle
                        cos_angle = torch.dot(n1_unit, n2_unit)
                        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
                        
                        # Determine sign of dihedral angle using the central bond direction
                        cross_product = torch.cross(n1_unit, n2_unit)
                        sign = torch.sign(torch.dot(cross_product, v2_unit))
                        dihedral = sign * torch.arccos(cos_angle)
                        
                        valid_torsions.append(dihedral)
            
            if len(valid_torsions) > 0:
                # Convert to tensor for vectorized Fourier computation (no .item() calls)
                torsions_t = torch.stack(valid_torsions)
                
                # Vectorized Fourier feature computation
                fourier_features = torch.zeros(feat_dim, device=pos_t.device)
                
                for n in range(1, n_fourier + 1):
                    sin_features = torch.sin(n * torsions_t)
                    cos_features = torch.cos(n * torsions_t)
                    
                    # Store in interleaved format: [sin1, cos1, sin2, cos2, ...]
                    idx = (n - 1) * 2
                    fourier_features[idx] = sin_features.mean()
                    fourier_features[idx + 1] = cos_features.mean()
                
                out[e] = fourier_features
                
            total_torsions += len(valid_torsions)
            processed_edges += 1
    
    t2 = time.perf_counter()
    if debug_timing:
        print(f"[TORSION-BATCH] build_adj={t1-t0:.4f}s compute_torsions={t2-t1:.4f}s "
              f"processed_edges={processed_edges}/{E} total_torsions={total_torsions}")
    
    return out



def create_3d_edge_features(pos, bond_edge_index, bond_edge_attr, cutoff=4.0, num_rbf=16, n_fourier=2, max_k_for_angles=4, include_torsions=True, max_k_for_torsions=4):
    """
    Create 3D geometric edge features combining distance RBF, bond features, angle summary, and torsion angles.
    
    Args:
        pos (numpy.ndarray): 3D coordinates
        bond_edge_index (torch.Tensor): Bond-based edge indices
        bond_edge_attr (torch.Tensor): Bond features
        cutoff (float): Distance cutoff
        num_rbf (int): Number of RBF basis functions
        n_fourier (int): Number of Fourier components for angles and torsions
        max_k_for_angles (int): Maximum neighbors for angle computation
        include_torsions (bool): Whether to include torsion angle features
        max_k_for_torsions (int): Maximum torsions per edge for performance
        
    Returns:
        tuple: (new_edge_index, new_edge_attr)
    """
    n_atoms = len(pos)
    
    # Handle single atom case
    if n_atoms == 1:
        # Single atom: create self-loop with zero features
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        # Create zero features: RBF + bond + angle + torsion
        bond_dim = bond_edge_attr.shape[1] if bond_edge_attr is not None else 13
        torsion_dim = 2 * n_fourier if include_torsions else 0
        total_dim = num_rbf + bond_dim + 2 * n_fourier + torsion_dim
        edge_attr = torch.zeros(1, total_dim)
        
        return edge_index, edge_attr
    
    # radius_graph expects a torch.float tensor [N, 3]
    pos_t = torch.as_tensor(pos, dtype=torch.float32)
    t0 = time.perf_counter()
    edge_index = radius_graph(
        pos_t, r=cutoff, loop=False, max_num_neighbors=32
    )
    t1 = time.perf_counter()
    # edge_index is shape [2, E], directed (both i->j and j->i present)

    if edge_index.numel() == 0:
        # Fall back to bond-only graph (keep old behavior)
        if bond_edge_index is not None and bond_edge_attr is not None:
            n_edges = bond_edge_index.shape[1]
            torsion_dim = 2 * n_fourier if include_torsions else 0
            padding_size = num_rbf + 2 * n_fourier + torsion_dim
            padding = torch.zeros(n_edges, padding_size)
            new_edge_attr = torch.cat([padding, bond_edge_attr], dim=1)
            return bond_edge_index, new_edge_attr
        else:
            return None, None
        
    # --- Distances for each edge (vectorized) ---
    src = edge_index[0]  # j
    dst = edge_index[1]  # i
    vec = pos_t[dst] - pos_t[src]           # [E, 3]
    edge_distances = torch.linalg.norm(vec, dim=1)  # [E]
    t2 = time.perf_counter()
    
    # Create RBF
    rbf = RadialBasisFunction(
        grid_min=0.0,
        grid_max=cutoff,
        num_grids=num_rbf,
        denominator=(cutoff / (num_rbf - 1)) * 1.2,
        linspace=True,
        trainable_grid=False
    )

    with torch.no_grad():
        rbf_features = rbf(edge_distances)  # [E, num_rbf]
    t3 = time.perf_counter()
    
    # --- Bond features aligned to edges ---
    # Build dict for quick lookup of real bond features; zeros otherwise.
    bond_dim = 13
    bond_features_all = torch.zeros(edge_index.shape[1], bond_dim)

    bond_features_dict = {}
    if bond_edge_index is not None and bond_edge_attr is not None:
        bond_dim = bond_edge_attr.shape[1]
        bond_features_all = torch.zeros(edge_index.shape[1], bond_dim)
        for k in range(bond_edge_index.shape[1]):
            i_k = bond_edge_index[0, k].item()
            j_k = bond_edge_index[1, k].item()
            bond_features_dict[(i_k, j_k)] = bond_edge_attr[k]

        # Fill for each new edge
        # (This loop is cheap: E over dict lookups, no N^2.)
        for e in range(edge_index.shape[1]):
            j = src[e].item()
            i = dst[e].item()
            if (j, i) in bond_features_dict:
                bond_features_all[e] = bond_features_dict[(j, i)]
    t4 = time.perf_counter()

    # --- Angle features using the same radius graph ---
    angle_features = compute_angle_features_from_graph(
        pos_t, edge_index, n_fourier=n_fourier, max_k_for_angles=max_k_for_angles
    )  # [E, 2*n_fourier]
    t5 = time.perf_counter()

    # --- Torsion features (optional) ---
    if include_torsions:
        torsion_features = compute_torsion_features_from_graph(
            pos_t, edge_index, n_fourier=n_fourier, max_k_for_torsions=max_k_for_torsions
        )  # [E, 2*n_fourier]
        t6 = time.perf_counter()
    else:
        torsion_features = torch.zeros(edge_index.shape[1], 0)
        t6 = t5

    if False:
        torsion_time = f"torsions={t6-t5:.4f}s " if include_torsions else ""
        print(f"[DEBUG] radius_graph={t1-t0:.4f}s "
                f"dist={t2-t1:.4f}s RBF={t3-t2:.4f}s "
                f"bond_map={t4-t3:.4f}s angles={t5-t4:.4f}s "
                f"{torsion_time}"
                f"E={edge_index.shape[1]}")

    # --- Concatenate blocks ---
    if include_torsions:
        new_edge_attr = torch.cat([rbf_features, bond_features_all, angle_features, torsion_features], dim=1)
    else:
        new_edge_attr = torch.cat([rbf_features, bond_features_all, angle_features], dim=1)
    
    return edge_index, new_edge_attr


def get_3d_coordinates_from_pubchem(smiles, timeout=10, max_retries=2, include_hydrogens=True):
    """
    Retrieve 3D coordinates from PubChem API using SMILES.
    
    Args:
        smiles (str): SMILES string
        timeout (int): Request timeout in seconds
        max_retries (int): Maximum number of retry attempts
        include_hydrogens (bool): Whether to preserve explicit hydrogens
        
    Returns:
        numpy.ndarray or None: 3D coordinates if found, None otherwise
    """
    try:
        # Step 1: Get CID from SMILES
        search_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{}/cids/JSON".format(quote(smiles, safe=''))
        # print(f"URL: {search_url}")

        for attempt in range(max_retries):
            try:
                # print(f"Searching PubChem for {smiles} (attempt {attempt + 1}/{max_retries})...")
                response = requests.get(search_url, timeout=timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                        cids = data['IdentifierList']['CID']
                        if cids:
                            cid = cids[0]  # Use first CID
                            
                            # Step 2: Get 3D SDF from CID
                            sdf_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF?record_type=3d"
                            sdf_response = requests.get(sdf_url, timeout=timeout)
                            
                            if sdf_response.status_code == 200:
                                sdf_content = sdf_response.text
                                
                                # Parse SDF content with RDKit
                                mol = Chem.MolFromMolBlock(sdf_content, removeHs=not include_hydrogens)
                                if mol is not None and mol.GetNumConformers() > 0:
                                    # If we want hydrogens but the molecule doesn't have them, add them
                                    if include_hydrogens and mol.GetNumAtoms() <= 10:  # Only for small molecules
                                        if not any(atom.GetSymbol() == 'H' for atom in mol.GetAtoms()):
                                            mol = Chem.AddHs(mol)
                                            # Try to generate coordinates for the new hydrogens
                                            try:
                                                AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                                                AllChem.MMFFOptimizeMolecule(mol)
                                            except:
                                                pass  # Use original coordinates
                                    
                                    conf = mol.GetConformer()
                                    pos = np.array([conf.GetAtomPosition(j) for j in range(mol.GetNumAtoms())])
                                    return pos
                                else:
                                    print(f"Invalid SDF content from PubChem for CID {cid}")
                            elif sdf_response.status_code == 404 or sdf_response.status_code == 400:
                                print(f"3D SDF not found in PubChem for CID {cid}")
                                break
                            else:
                                print(f"Failed to get SDF from PubChem (status: {sdf_response.status_code})")
                elif response.status_code == 404 or response.status_code == 400:
                    print(f"Compound not found in PubChem")
                    break  # Don't retry for 404 or 400
                else:
                    print(f"PubChem search failed (status: {response.status_code})")
                    
            except requests.exceptions.Timeout:
                print(f"PubChem request timeout (attempt {attempt + 1})")
            except requests.exceptions.RequestException as e:
                print(f"PubChem request error: {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
        # time.sleep(0.3)            
    except Exception as e:
        print(f"Error in PubChem lookup: {e}")
    
    return None


def get_3d_coordinates(smiles, dataset_type=None, seed=42, include_hydrogens=True, mol_index=None):
    """
    Get 3D coordinates for a molecule using RDKit direct generation, PubChem API as fallback.
    Uses caching to avoid repeated processing for the same molecules.
    
    Note: SDF file lookup for QM8/QM9 has been disabled in favor of direct RDKit generation
    for better performance (100x faster) while maintaining the same success rate.
    
    Args:
        smiles (str): SMILES string
        dataset_type (str): Dataset type - used for cache key differentiation
        seed (int): Random seed for conformer generation
        include_hydrogens (bool): Whether to add/preserve explicit hydrogens
        mol_index (int): Molecule index - used for cache key differentiation
        
    Returns:
        numpy.ndarray or None: 3D coordinates if successful, None otherwise
    """
    # Strategy 0: Check cache first
    # Cache key only depends on SMILES, include_hydrogens, and seed - not dataset_type or mol_index
    # since the same SMILES with same parameters should yield identical 3D coordinates
    pos = load_from_cache(smiles, include_hydrogens, seed)
    if pos is not None:
        return pos
    
    # Strategy 1: Try direct RDKit generation first (fastest and most reliable)
    pos = generate_3d_conformer(smiles, seed=seed, include_hydrogens=include_hydrogens)
    if pos is not None:
        save_to_cache(smiles, pos, include_hydrogens, seed)
        return pos
    
    # Strategy 2: Try PubChem API as fallback (mainly for HIV/TOXCAST datasets)
    print(f"Direct RDKit generation failed, trying PubChem API for {smiles}...")
    pos = get_3d_coordinates_from_pubchem(smiles, include_hydrogens=include_hydrogens)
    if pos is not None:
        print(f"Successfully retrieved 3D coordinates from PubChem")
        save_to_cache(smiles, pos, include_hydrogens, seed)
        return pos
    
    # If both methods fail, return None
    print(f"All methods failed for {smiles}")
    return pos


def test_torsion_features():
    """Test function to verify torsion feature computation works correctly."""
    print("Testing torsion feature computation...")
    
    # Create a simple 4-atom molecule (butane-like chain)
    pos = np.array([
        [0.0, 0.0, 0.0],    # atom 0
        [1.0, 0.0, 0.0],    # atom 1  
        [2.0, 1.0, 0.0],    # atom 2
        [3.0, 1.0, 0.0]     # atom 3
    ])
    
    # Create bond edges (0-1-2-3 chain)
    bond_edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],  # source nodes
        [1, 0, 2, 1, 3, 2]   # target nodes  
    ], dtype=torch.long)
    
    # Create dummy bond features
    bond_edge_attr = torch.ones(6, 13)  # 6 edges, 13 bond features each
    
    # Test with torsion features enabled
    edge_index, edge_attr = create_3d_edge_features(
        pos, bond_edge_index, bond_edge_attr, 
        cutoff=4.0, n_fourier=2, include_torsions=True
    )
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    print(f"Expected edge attr dim: RBF(16) + Bond(13) + Angles(4) + Torsions(4) = 37")
    print(f"Actual edge attr dim: {edge_attr.shape[1]}")
    
    # Test with torsions disabled
    edge_index_no_torsion, edge_attr_no_torsion = create_3d_edge_features(
        pos, bond_edge_index, bond_edge_attr, 
        cutoff=4.0, n_fourier=2, include_torsions=False
    )
    
    print(f"Without torsions - Edge attr dim: {edge_attr_no_torsion.shape[1]}")
    print("Torsion feature test completed successfully!")


if __name__ == "__main__":
    # Run test if script is executed directly
    test_torsion_features()
