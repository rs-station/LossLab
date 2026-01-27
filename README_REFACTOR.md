# LossLab: Modular Coordinate Refinement Library

LossLab is a modular library for coordinate refinement against experimental data (cryo-EM maps, X-ray density).

## Quick Start

```python
from LossLab import RealSpaceLoss, RefinementEngine, RefinementConfig

# 1. Setup (one section)
loss_fn = RealSpaceLoss(target_map, pdb_obj, loss_type="l2")
config = RefinementConfig(num_iterations=400, output_dir="./output")
engine = RefinementEngine(config, loss_fn, structure_factor_calc)

# 2. Run refinement (one line!)
results = engine.run(reference_coords, prediction_callback, features)

# 3. Access results
print(f"Best loss: {results['best_loss']:.6f}")
```

```bash
cd LossLab
pip install -e .
```

## Structure

### Core Modules

```
LossLab/
├── losses/
│   ├── base.py              # Base loss class
│   └── realspace.py         # Real-space losses (CC, L2, Sinkhorn)
├── refinement/
│   ├── engine.py            # Main refinement engine
│   ├── config.py            # Configuration dataclass
│   ├── metrics.py           # Metrics tracking
│   ├── checkpoint.py        # Checkpoint management
│   └── utils.py             # Utility functions
└── utils/
    ├── decorators.py        # Reusable decorators
    ├── geometry.py          # Geometry operations (Kabsch, RMSD, etc.)
    └── map_utils.py         # Map operations (masking, smoothing, etc.)
```

### Key Components

#### 1. Loss Functions (`LossLab.losses`)

All loss functions inherit from `BaseLoss`:

```python
from LossLab.losses import RealSpaceLoss

loss_fn = RealSpaceLoss(
    target_map=ccp4_map,
    pdb_obj=pdb_parser,
    device="cuda:0",
    loss_type="l2",  # Options: "cc", "l2", "sinkhorn"
    mask_center=np.array([10.0, 20.0, 15.0]),
    mask_radius=15.0,
)

# Compute loss
loss = loss_fn.compute(coordinates, structure_factor_calc)
```

**Supported loss types:**
- `"cc"`: Negative correlation coefficient
- `"l2"`: L2 loss on normalized maps
- `"sinkhorn"`: Optimal transport (Sinkhorn divergence)

#### 2. Refinement Engine (`LossLab.refinement.RefinementEngine`)

Handles the complete refinement workflow:

```python
from LossLab.refinement import RefinementEngine

engine = RefinementEngine(
    config=refinement_config,
    loss_function=loss_fn,
    structure_factor_calculator=sfc,
    rigid_body_refine_fn=optional_rbr_function,
)

results = engine.run(
    reference_coordinates=ref_coords,
    prediction_callback=af_forward_pass,
    feature_dict=features,
    save_pdb_callback=optional_save_function,
)
```

**Engine handles:**
- Multiple runs with automatic tracking
- Early stopping
- Metrics logging (CSV + NPZ)
- Checkpoint management
- Progress bars
- Memory management
- Best model selection

#### 3. Configuration (`LossLab.refinement.RefinementConfig`)

All parameters in one dataclass:

```python
from LossLab.refinement import RefinementConfig

config = RefinementConfig(
    # Optimization
    num_iterations=400,
    num_runs=5,
    learning_rate_additive=1e-3,
    learning_rate_multiplicative=1e-3,
    
    # Loss
    loss_type="l2",
    
    # Output
    output_dir="./output",
    run_note="my_refinement",
    save_every_n_iterations=50,
    
    # Early stopping
    early_stopping_patience=150,
    early_stopping_min_delta=0.0001,
)

# Save for reproducibility
config.to_yaml("config.yaml")

# Load later
config = RefinementConfig.from_yaml("config.yaml")
```

#### 4. Utilities

**Decorators** (`LossLab.utils.decorators`):
```python
from LossLab.utils.decorators import gpu_memory_tracked, timed, cached_property

@gpu_memory_tracked  # Logs GPU memory usage
@timed              # Logs execution time
def my_function():
    ...

class MyClass:
    @cached_property  # Computed once and cached
    def expensive_property(self):
        return compute_something()
```

**Geometry** (`LossLab.utils.geometry`):
```python
from LossLab.utils.geometry import kabsch_align, compute_rmsd

# Align coordinates
aligned = kabsch_align(moving_coords, reference_coords)

# Compute RMSD
rmsd = compute_rmsd(coords1, coords2)
```

**Map utilities** (`LossLab.utils.map_utils`):
```python
from LossLab.utils.map_utils import normalize_map, create_spherical_mask, gaussian_smooth_3d

# Normalize map
normalized = normalize_map(map_tensor, mask, method="zscore")

# Create spherical mask
mask = create_spherical_mask(grid_shape, center, radius, voxel_size, device)

# Smooth map
smoothed = gaussian_smooth_3d(map_tensor, sigma_angstrom=4.0, voxel_size)
```

## Comparison with og ROCKET prototype

### Before (ROCKET)

```python
# ~300+ lines 
for n in range(num_runs):
    run_id = number_to_letter(n)
    # Setup features, bias, optimizer, tracking, progress bar...
    for iteration in range(num_iterations):
        # Reset features
        # AF forward pass
        # Extract coordinates
        # Kabsch alignment
        # Save preRBR PDB
        # Rigid body refinement
        # Compute loss
        # Track metrics
        # Check if best
        # Save periodically
        # Update progress bar
        # Backward pass
        # Check early stopping
        # Step optimizer
        # Clear cache
        # Log to file
    # Save run results
# Save best overall results
# Write summary
```

### After (LossLab)

```python
# ~50 lines total
loss_fn = RealSpaceLoss(target_map, pdb_obj, loss_type="l2")
config = RefinementConfig(num_iterations=400, output_dir="./output")
engine = RefinementEngine(config, loss_fn, sfc)

results = engine.run(reference_coords, predict_callback, features)

print(f"Best loss: {results['best_loss']:.6f}")
```
