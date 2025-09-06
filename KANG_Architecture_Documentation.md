# KANG Architecture: Comprehensive Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Single Task Setup](#single-task-setup)
4. [Global Features Integration](#global-features-integration)
5. [Multi-Task Learning](#multi-task-learning)
6. [3D Geometric Features](#3d-geometric-features)
7. [Molecular Graph Construction](#molecular-graph-construction)
8. [Technical Advantages](#technical-advantages)

## Overview

**KANG (Kolmogorov-Arnold Network for Graphs)** is a novel graph neural network architecture that replaces traditional linear transformations with Kolmogorov-Arnold Networks (KANs). The architecture is built on PyTorch Geometric and designed for molecular property prediction tasks, offering superior function approximation capabilities through learnable univariate functions.

## Core Architecture

### Key Components

#### 1. **KAND (KAN Dense Layer)**
The foundational building block that replaces standard linear layers with learnable univariate functions.

**Features:**
- **Radial Basis Functions (RBF)**: Uses Gaussian RBF with trainable or fixed grid points
- **Spline-based Functions**: Implements learnable B-spline functions for smooth approximations
- **Base + Spline Architecture**: Combines a base activation (SiLU) with spline transformations
- **Mathematical Form**: 
  ```
  f(x) = SplineLinear(RBF(LayerNorm(x))) + BaseLinear(SiLU(x))
  ```

**Key Parameters:**
- `grid_min`, `grid_max`: Define the function domain
- `num_grids`: Number of control points for spline approximation
- `trainable_grid`: Whether grid points are learnable
- `use_base_update`: Enable/disable base linear transformation

#### 2. **KANGConv (KAN Graph Convolution)**
Graph convolution layer implementing learnable message functions using KAND layers.

**Features:**
- **Message Passing**: Implements learnable message functions using KAND layers
- **Edge-Aware**: Incorporates edge attributes through concatenation `[x_j, edge_attr]`
- **Lazy Initialization**: Dynamically adapts to edge attribute dimensions
- **Residual Connections**: Includes skip connections for better gradient flow
- **Aggregation**: Supports mean, sum, and max aggregation schemes

**Message Function:**
```python
# Content message from [x_j, edge_attr]
z = torch.cat([x_j, edge_attr], dim=-1)
m = self.msg_kand(z)  # Learnable message transformation
```

#### 3. **Graph Construction Pipeline**
Sophisticated molecular representation combining multiple information sources.

## Single Task Setup

The base KANG model supports both **classification** and **regression** tasks with identical encoder architectures but different output processing.

### Classification (KANG.py)
- **Output**: Log-softmax activated predictions for binary/multi-class tasks
- **Architecture**: Encoder → Global Pooling → Optional Global Features → KAN/MLP Classifier
- **Loss**: Typically used with NLL loss or cross-entropy
- **Final Activation**: `F.log_softmax(x, dim=1)`

### Regression (KANG_regression.py)
- **Output**: Direct continuous value predictions
- **Architecture**: Same as classification but without final activation
- **Loss**: Typically used with MSE or MAE loss
- **Final Output**: Raw predictions without activation

**Common Architecture Flow:**
```
Input Graph → KANGConv Layers → LayerNorm → Dropout → Global Pooling → Global Features (optional) → Output Layer
```

## Global Features Integration

The system incorporates **200 normalized RDKit molecular descriptors** through a sophisticated global features module.

### Features Source
- **descriptastorus**: Uses pre-normalized RDKit descriptors following D-MPNN approach
- **Normalization**: Features are normalized to [0,1] range using empirical CDFs
- **Dimensionality**: Exactly 200 features per molecule
- **Categories**: Includes topological, electronic, and physicochemical descriptors

### Feature Types
- **Topological**: Molecular connectivity and shape descriptors
- **Electronic**: Charge distribution and electronic properties  
- **Physicochemical**: Solubility, lipophilicity, and drug-like properties
- **Geometric**: 2D molecular shape and size descriptors

### Integration Mechanism
```python
# During forward pass:
if self.use_global_features and global_features is not None:
    batch_size = x.size(0)
    global_features = global_features.view(batch_size, -1)
    # Concatenate after graph pooling
    x = torch.cat([x, global_features], dim=1)
```

**Benefits:**
- **Complementary Information**: Provides molecular context beyond graph structure
- **Robust Representations**: Combines local (graph) and global (molecular) information
- **Proven Effectiveness**: Based on successful D-MPNN methodology

## Multi-Task Learning

The architecture supports sophisticated multi-task learning through four specialized variants, offering different trade-offs between performance and efficiency.

### Multi-Head Approach

#### 1. **KANG_MultiTask.py** (Classification Tasks)
- **Architecture**: Shared encoder + Task-specific KAN heads
- **Output Shape**: `[batch_size, num_tasks, 2]` for binary classification
- **Task Heads**: Each task has its own KAND/KANLinear output layer
- **Activation**: Log-softmax per task for independent classification

#### 2. **KANG_MultiTask_Regression.py** (Regression Tasks)
- **Architecture**: Shared encoder + Task-specific regression heads
- **Output Shape**: `[batch_size, num_tasks]`
- **Task Heads**: Each task outputs a single continuous value
- **Loss**: Can handle different loss functions per task

### Single Head Approach

#### 3. **KANG_MultiTask_SingleHead.py** (Classification)
- **Key Difference**: Uses a **single large output layer** instead of multiple task-specific heads
- **Architecture**: Shared Encoder → Global Pooling → Optional Global Features → **Single KAN Head**
- **Output Computation**: 
  ```python
  # Single head outputs: [batch_size, num_tasks * 2]
  out = self.head(x)
  # Reshape to: [batch_size, num_tasks, 2] 
  out = out.view(-1, self.num_tasks, 2)
  # Apply log_softmax per task
  out = F.log_softmax(out, dim=2)
  ```

#### 4. **KANG_MultiTask_Regression_SingleHead.py** (Regression)  
- **Architecture**: Shared Encoder → Global Pooling → Optional Global Features → **Single KAN Head**
- **Output**: Direct `[batch_size, num_tasks]` tensor
- **Simpler Design**: No reshaping needed for regression tasks

### Architectural Comparison: Multi-Head vs Single Head

| Aspect | **Multi-Head Approach** | **Single Head Approach** |
|--------|------------------------|--------------------------|
| **Output Layer** | `num_tasks` separate KAN/MLP layers | 1 large KAN/MLP layer |
| **Parameters** | More parameters (separate weights per task) | Fewer parameters (shared final transformation) |
| **Task Independence** | High (each task has dedicated weights) | Lower (tasks share final layer weights) |
| **Computational Cost** | Higher (multiple forward passes through heads) | Lower (single forward pass) |
| **Memory Usage** | Higher (multiple head parameters) | Lower (single head parameters) |
| **Task Interactions** | Limited to shared encoder | Extended to final layer |

### Mathematical Formulation

**Multi-Head Approach:**
```
For each task i ∈ {1, ..., num_tasks}:
    y_i = Head_i(Encoder(x, edge_index, edge_attr))
```

**Single Head Approach:**
```
y = Head(Encoder(x, edge_index, edge_attr))
where Head outputs: [y_1, y_2, ..., y_num_tasks]
```

### Multi-Task Benefits
- **Parameter Sharing**: Common molecular representation across tasks
- **Transfer Learning**: Knowledge transfer between related tasks
- **Efficiency**: Single forward pass for multiple predictions
- **Regularization**: Implicit regularization through shared representations

### Use Case Recommendations

**Choose Single Head When:**
- Tasks are **highly related** (e.g., predicting different quantum mechanical properties)
- **Parameter efficiency** is crucial
- You have **limited computational resources**
- Tasks have **similar complexity** and **output requirements**
- You want **stronger regularization** through forced parameter sharing

**Choose Multi-Head When:**
- Tasks are **diverse** or **weakly related** 
- You need **maximum task-specific performance**
- **Computational resources are abundant**
- Tasks have **different complexities** or **output requirements**
- You want **maximum flexibility** for task-specific adaptations

## 3D Geometric Features

The system incorporates sophisticated 3D molecular geometry through an advanced geometric features module, going beyond traditional 2D molecular graphs.

### 3D Coordinate Generation
- **Primary Method**: RDKit ETKDGv3 conformer generation with MMFF optimization
- **Fallback**: PubChem API for molecules where RDKit fails
- **Caching**: Intelligent caching system to avoid recomputation
- **Deterministic**: Seeded generation for reproducible results

### 3D Edge Features Architecture
The edge attributes are constructed as a rich concatenation of complementary geometric information:

```
edge_attr[j→i] = [RBF(d_ji) || bond_bits || angle_summary(j,i) || torsion_summary(j,i)]
```

#### Components:

##### 1. **RBF Distance Encoding** (16 dimensions)
- **Purpose**: Encode inter-atomic distances using smooth basis functions
- **Implementation**: Gaussian radial basis functions over inter-atomic distances
- **Cutoff**: Distance-based edge generation (default 4.0 Å)
- **Grid**: Learnable or fixed grid points for basis functions
- **Formula**: `RBF(d) = exp(-((d - μ_i) / σ)²)` for each basis function i

##### 2. **Bond Features** (13 dimensions)
- **Content**: Traditional 2D bond information preserved in 3D context
- **Includes**: Bond type, conjugation, ring membership, stereo information
- **Handling**: Zero-padded for distance-based edges without chemical bonds
- **Compatibility**: Maintains 2D molecular graph information

##### 3. **Angle Summary Features** (2×n_fourier dimensions, default 4)
- **Purpose**: Capture local angular geometry around each atom
- **Method**: For each edge j→i, considers all neighbors k of j
- **Computation**: Computes angles k-j-i using vectorized operations
- **Encoding**: Fourier encoding `[sin(n×angle), cos(n×angle)]` for n=1..n_fourier
- **Aggregation**: Mean over all valid angle triplets for rotation invariance
- **Performance**: Optimized vectorized implementation for speed

##### 4. **Torsion Features** (2×n_fourier dimensions, optional)
- **Purpose**: Capture conformational information through dihedral angles
- **Method**: Dihedral angles for 4-atom paths k-j-i-l
- **Computation**: Vectorized computation using cross products
- **Encoding**: Same Fourier encoding as angles
- **Trade-off**: Computationally expensive, often disabled for speed
- **Use Cases**: Critical for conformationally sensitive properties

### Performance Optimizations
- **Vectorized Operations**: Batch processing of geometric calculations
- **Early Termination**: Skip expensive computations for small molecules
- **Neighbor Limits**: Cap the number of neighbors considered for angles/torsions
- **CPU-GPU Optimization**: Strategic movement between devices
- **Fake 3D Mode**: Fallback with zero geometric features when 3D fails

### 3D vs 2D Comparison

| Feature | **2D Mode** | **3D Mode** |
|---------|-------------|-------------|
| **Edge Features** | 13D bond features | 37+D geometric features |
| **Spatial Information** | Topological only | Real 3D coordinates |
| **Computational Cost** | Low | Higher |
| **Information Content** | Chemical bonds | Bonds + geometry + conformation |
| **Applicability** | Universal | Requires 3D coordinates |

## Molecular Graph Construction

The `smiles_to_graph.py` module handles the complete molecular representation pipeline.

### Node Features (Rich Atom Representation)
Comprehensive atomic feature vector incorporating chemical and structural information:

- **Atomic Number**: One-hot encoding (1-100) for element identity
- **Degree**: Number of bonds (0-5) for connectivity information
- **Formal Charge**: Charge state (-2 to +2) for electronic properties
- **Chirality**: Stereochemistry information for spatial arrangement
- **Hydrogen Count**: Implicit hydrogens (0-4) for hydrogen bonding
- **Hybridization**: SP, SP2, SP3, SP3D, SP3D2 for orbital information
- **Aromaticity**: Binary aromatic flag for π-electron systems
- **Atomic Mass**: Scaled continuous value for physical properties

**Total Node Feature Dimension**: ~120 features per atom

### Edge Features
**2D Mode**: Traditional bond features (13 dimensions)
- Nullity, bond type, conjugation, ring membership, stereo information

**3D Mode**: Extended geometric features (37+ dimensions)
- RBF distance encoding + bond features + angle features + optional torsion features

### Flexibility and Robustness
- **Hydrogen Handling**: Optional explicit hydrogen addition for detailed modeling
- **Coordinate Systems**: Supports both 2D and 3D representations seamlessly
- **Multi-label Support**: Handles single and multi-task labels efficiently
- **Error Handling**: Robust fallback mechanisms for invalid molecules
- **Caching**: Intelligent caching for 3D coordinate generation

## Technical Advantages

### 1. **Learnable Functions vs Fixed Activations**
- **Traditional GNNs**: Use fixed activations (ReLU, GELU, Tanh)
- **KANG**: Learns optimal activation functions per neuron
- **Benefit**: Better function approximation capabilities and adaptive representations

### 2. **3D Geometric Awareness**
- **Traditional**: Limited to 2D molecular graphs
- **KANG**: Incorporates real 3D structure information
- **Benefit**: Captures spatial relationships and conformational effects

### 3. **Multi-Modal Integration**
- **Graph Structure**: Chemical connectivity and bonding
- **3D Geometry**: Spatial coordinates and geometric features
- **Global Descriptors**: Molecular-level physicochemical properties
- **Benefit**: Complementary information sources for robust predictions

### 4. **Scalable Multi-Task Learning**
- **Shared Representations**: Efficient parameter sharing across tasks
- **Task-Specific Adaptation**: Flexible head architectures for different requirements
- **Architectural Flexibility**: Both multi-head and single-head variants
- **Benefit**: Supports both classification and regression tasks efficiently

### 5. **Advanced Function Approximation**
- **KAN Theory**: Based on Kolmogorov-Arnold representation theorem
- **Universal Approximation**: Can theoretically approximate any continuous function
- **Adaptive Learning**: Learns problem-specific activation functions
- **Benefit**: Superior expressiveness compared to traditional neural networks

## Implementation Details

### Key Dependencies
- **PyTorch Geometric**: Graph neural network framework
- **RDKit**: Molecular informatics and 3D generation
- **descriptastorus**: Normalized molecular descriptors
- **torch_cluster**: Efficient graph operations

### Model Configuration
```python
# Example KANG configuration
model = KANG(
    in_channels=120,          # Node feature dimension
    hidden_channels=256,      # Hidden representation size
    out_channels=2,           # Output classes/values
    num_layers=3,             # Number of graph convolution layers
    grid_min=-2.0,           # KAN function domain
    grid_max=2.0,
    num_grids=8,             # Number of basis functions
    dropout=0.1,             # Regularization
    kan=True,                # Use KAN layers
    use_global_features=True, # Include global descriptors
    use_self_loops=True      # Add self-connections
)
```

### Training Considerations
- **Learning Rate**: Typically requires lower learning rates due to spline complexity
- **Regularization**: Built-in regularization through function smoothness
- **Memory Usage**: Higher memory requirements due to RBF computations
- **Convergence**: May require more epochs but achieves better final performance

## Conclusion

The KANG architecture represents a significant advancement in molecular property prediction, combining:

1. **Theoretical Foundation**: Kolmogorov-Arnold representation theorem
2. **Practical Innovation**: Learnable activation functions in graph neural networks
3. **Multi-Modal Integration**: Graph structure + 3D geometry + global features
4. **Architectural Flexibility**: Multiple variants for different use cases
5. **Performance Optimization**: Efficient implementations for real-world applications

This comprehensive approach enables superior molecular property prediction across diverse chemical spaces and task types, making KANG a powerful tool for computational chemistry and drug discovery applications.
