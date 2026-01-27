# LossLab Examples

This directory contains example scripts demonstrating the clean LossLab API for coordinate refinement.

## Quick Start

### Minimal Example

```python
from LossLab import RealSpaceLoss, RefinementEngine, RefinementConfig

# 1. Setup
loss_fn = RealSpaceLoss(target_map, pdb_obj, device="cuda:0", loss_type="l2")
config = RefinementConfig(num_iterations=400, output_dir="./output")
engine = RefinementEngine(config, loss_fn, structure_factor_calc)

# 2. Run (one line!)
results = engine.run(reference_coords, prediction_callback, features)

# 3. Done!
print(f"Best loss: {results['best_loss']:.6f}")
```

## Files

### `simple_refinement.py`
Complete working example showing the full refinement workflow with LossLab's clean API.

**Key features:**
- Minimal boilerplate
- Clear separation of concerns
- Easy to understand and modify

**Usage:**
```bash
python simple_refinement.py
```

### `comparison_old_vs_new.py`
Side-by-side comparison of the old ROCKET approach vs the new LossLab approach.

**Highlights:**
- **83% code reduction** (300+ lines → 50 lines)
- Improved clarity and maintainability
- Better testability
- More reusable components

## Key Concepts

### 1. Modular Design

LossLab separates functionality into clear modules:

```python
from LossLab.losses import RealSpaceLoss          # Loss computation
from LossLab.refinement import RefinementEngine   # Training loop
from LossLab.refinement import RefinementConfig   # Configuration
from LossLab.refinement import MetricsTracker     # Logging
from LossLab.refinement import CheckpointManager  # Checkpointing
from LossLab.utils import kabsch_align            # Geometry utilities
```

### 2. Configuration-Driven

All parameters in one place:

```python
config = RefinementConfig(
    num_iterations=400,
    num_runs=5,
    learning_rate_additive=1e-3,
    learning_rate_multiplicative=1e-3,
    loss_type="l2",
    output_dir="./output",
    early_stopping_patience=150,
    save_every_n_iterations=50,
)

# Save for reproducibility
config.to_yaml("config.yaml")

# Load later
config = RefinementConfig.from_yaml("config.yaml")
```

### 3. Decorator-Based Utilities

Common patterns handled by decorators:

```python
from LossLab.utils.decorators import gpu_memory_tracked, timed, cached_property

@gpu_memory_tracked
@timed
def my_expensive_function():
    # Automatically logs time and memory usage
    ...

class MyClass:
    @cached_property
    def expensive_property(self):
        # Computed once and cached
        return compute_something_expensive()
```

### 4. Clean Loss API

Easy to create custom loss functions:

```python
from LossLab.losses import BaseLoss

class MyCustomLoss(BaseLoss):
    def compute(self, coordinates, **kwargs):
        # Your loss logic here
        return loss_value
```

### 5. Flexible Refinement Engine

The engine handles:
- Multiple runs automatically
- Early stopping
- Metrics tracking
- Checkpoint management
- Progress bars
- Memory management

All you provide:
1. Configuration
2. Loss function
3. Prediction callback

## Comparison: Before vs After

### Before (ROCKET):

```python
# 300+ lines of boilerplate...
for n in range(num_runs):
    run_id = number_to_letter(n)
    # Load features
    # Initialize bias
    # Setup tracking
    # Create progress bar
    for iteration in range(num_iterations):
        # Reset features
        # AF forward pass
        # Extract coordinates
        # Kabsch alignment
        # Save preRBR PDB
        # Rigid body refinement
        # Compute loss
        # Track metrics
        # Check best
        # Save periodically
        # Update progress
        # Backward pass
        # Early stopping
        # Step optimizer
        # Clear cache
        # Log metrics
    # Save run results
# Save best results
# Write summary
```

### After (LossLab):

```python
# ~50 lines total
loss_fn = RealSpaceLoss(target_map, pdb_obj, loss_type="l2")
config = RefinementConfig(num_iterations=400, output_dir="./output")
engine = RefinementEngine(config, loss_fn, sfc)

results = engine.run(reference_coords, predict_callback, features)

print(f"Best loss: {results['best_loss']:.6f}")
```

## Advanced Usage

### Custom Rigid Body Refinement

```python
def my_rigid_body_refine(coords, loss_fn, sfc, **kwargs):
    # Your custom RBR logic
    return refined_coords, loss_history

engine = RefinementEngine(
    config,
    loss_fn,
    sfc,
    rigid_body_refine_fn=my_rigid_body_refine,
)
```

### Custom Save Callback

```python
def save_pdb_with_metadata(coords, path):
    # Your custom PDB saving logic
    ...

results = engine.run(
    reference_coords,
    predict_callback,
    features,
    save_pdb_callback=save_pdb_with_metadata,
)
```

### Multiple Loss Types

```python
# Try different losses easily
for loss_type in ["l2", "cc", "sinkhorn"]:
    loss_fn = RealSpaceLoss(target_map, pdb_obj, loss_type=loss_type)
    config = RefinementConfig(loss_type=loss_type, output_dir=f"./output_{loss_type}")
    engine = RefinementEngine(config, loss_fn, sfc)
    results = engine.run(reference_coords, predict_callback, features)
```

## Testing

Each component can be tested independently:

```python
# Test loss function
loss_fn = RealSpaceLoss(target_map, pdb_obj)
loss = loss_fn.compute(test_coords, sfc)
assert loss.item() < 1e6

# Test metrics tracker
tracker = MetricsTracker("./test_output", "A")
tracker.log(iteration=0, loss=1.23, plddt=0.85)
tracker.save()

# Test checkpoint manager
manager = CheckpointManager("./test_output")
manager.save_checkpoint(0, "A", 1.23, msa_bias=test_tensor)
loaded = manager.load_checkpoint("A", 0, ["msa_bias"])
```

## Tips

1. **Start simple**: Use default configuration first
2. **Profile memory**: Use `gpu_memory_tracked` decorator
3. **Save configs**: Always save your `RefinementConfig` for reproducibility
4. **Monitor metrics**: Check the CSV logs for debugging
5. **Use early stopping**: Saves time on converged runs

## Support

For questions or issues, please refer to the main LossLab documentation or open an issue on GitHub.
