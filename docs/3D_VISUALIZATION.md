# 3D Structure Visualization in Weights & Biases

LossLab automatically logs interactive 3D protein structures to your Weights & Biases dashboard during refinement.

## Features

### 1. **Real-time Structure Logging**
Each iteration can stream a live 3D frame to W&B so you can watch refinement progress as it happens.

### 2. **Trajectory Animation**
At the end of refinement, an animated 3D viewer is logged (HTML + 3Dmol.js) so you can play the trajectory like a movie and
- Rotate structures with mouse drag
- Zoom with scroll wheel
- Step through frames or play/pause
- Adjust animation speed

## Setup

Simply enable W&B logging in your refinement config:

```python
from LossLab import RefinementConfig, RefinementEngine

config = RefinementConfig(
    # ... other parameters ...
    
    # Enable W&B logging
    use_wandb=True,
    wandb_entity="your-team",
    wandb_project="protein-refinement",
    wandb_name="experiment_name",
    
    # Enable trajectory saving
    save_best_pdb=True,
    save_trajectory_pdb=True,
    save_trajectory_interval=1,
)

engine = RefinementEngine(
    config=config,
    loss_function=loss_fn,
    structure_factor_calculator=sfc,
    pdb_template="your_structure.pdb",  # Important!
)
```

## What Gets Logged

### During Refinement
- **Live frames**: Each iteration can log a live structure
- **Step tracking**: Each structure is associated with its iteration number
- **Loss values**: Captions show the loss value for each structure

### At Completion
- **Final best model**: The overall best structure with its loss value
- **Trajectory animation**: Embedded HTML player (3Dmol.js)

## Viewing in W&B

1. **Go to your W&B run page**
2. **Navigate to the "Media" panel** or scroll through logged data
3. **Look for**:
    - `trajectory_*_live`: Live streaming frame
    - `trajectory_animation_*`: Animated viewer
4. **Click to interact** with any 3D structure

## Tips

### Limit Trajectory Frames
For long refinements, you may want to limit logged frames to avoid UI slowdown.
The animation builder uses `max_frames` (default 100).

### Compare Structures
Use W&B's comparison features to view multiple structures side-by-side across different runs.

### Performance
Live frames add a small overhead per iteration. If you want faster runs, increase
`save_trajectory_interval` or disable live streaming by turning off W&B.

## Troubleshooting

### No 3D structures appearing?
- Ensure `pdb_template` is provided to `RefinementEngine`
- Check that `save_best_pdb=True` in config
- Verify W&B is properly initialized (check for W&B URL in logs)

### Atom count mismatch warnings?
Make sure the template PDB contains the same atoms as your coordinates. If mdtraj
loads fewer atoms, it could be due to having altlocs in your input PDB.

## Example Output

In your W&B dashboard, you'll see:
- Live frames during refinement
- An animated 3D viewer at completion
- Metrics (loss, RMSD, etc.) correlated with structure changes


## Requirements

- `wandb` package installed: `pip install wandb`
- Valid W&B account and login
- PDB template file provided to engine

## If W&B Is Disabled

If `use_wandb=False`, LossLab does not stream live frames or create the
embedded animation viewer. You will still get local outputs when trajectory
saving is enabled:

- `trajectory/*_refinement_trajectory.pdb` (multi-model PDB)
- `trajectory/best_*.pdb` (best model snapshots)

You can open these locally in PyMOL or other viewers.
