# Weights & Biases Setup Guide

How to install and use Weights & Biases (W&B) for experiment tracking with LossLab.

## Installation

```bash
pip install wandb
```

## Setup Options

### Option 1: Cloud Logging (Recommended for Collaboration)

1. **Create Account**
   ```bash
   # Visit https://wandb.ai/signup
   # Sign up (free for academics)
   ```

2. **Login**
   ```bash
   wandb login
   # Follow prompts to paste your API key from https://wandb.ai/authorize
   ```

3. **Use in LossLab**
   ```python
   config = RefinementConfig(
       # ... other config ...
       use_wandb=True,
       wandb_project="my-project",
       wandb_name="experiment-1",
       wandb_tags=["protein-refinement", "test"],
      save_best_pdb=True,
      save_trajectory_pdb=True,
      save_trajectory_interval=1,
   )
   ```

4. **View Results**
   - Visit https://wandb.ai/your-username/my-project
   - Web UI with interactive plots
   - Compare multiple runs
   - Share 

### Option 2: Local Logging (No Account Needed)

1. **Enable Offline Mode**
   ```bash
   # Set environment variable
   export WANDB_MODE=offline
   
   # Or in Python
   import os
   os.environ["WANDB_MODE"] = "offline"
   ```

2. **Use in LossLab** (same as above)
   ```python
   config = RefinementConfig(
       use_wandb=True,
       wandb_project="my-project",
       wandb_name="experiment-1",
   )
   ```

3. **View Results Locally**
   
   **Method A: Local Server**
   ```bash
   # Install local server
   pip install wandb[local]
   
   # Start local server
   wandb local
   # Opens browser at http://localhost:8080
   ```
   
   **Method B: Export and Upload Later**
   ```bash
   # Your runs are saved in ./wandb/ directory
   # Each run has a folder like wandb/offline-run-20260127_120000-abcd1234/
   
   # To sync later (if you create an account):
   wandb sync ./wandb/offline-run-20260127_120000-abcd1234/
   ```
   
   **Method C: View Files Directly**
   ```bash
   # Metrics are saved as JSON in ./wandb/
   ls wandb/offline-run-*/files/
   
   # View metrics
   cat wandb/offline-run-*/files/wandb-history.jsonl | jq
   
   # Or convert to CSV
   python -c "
   import json, pandas as pd
   with open('wandb/offline-run-*/files/wandb-history.jsonl') as f:
       data = [json.loads(line) for line in f]
   df = pd.DataFrame(data)
   df.to_csv('metrics.csv', index=False)
   "
   ```

### Option 3: Disable W&B (Default)

```python
config = RefinementConfig(
    # ... other config ...
    use_wandb=False,  # Default
)
```

Metrics are still saved to CSV and NPZ files in the output directory.
Trajectory PDBs are still written locally if `save_trajectory_pdb=True`, but
no live frames or animations are logged to W&B.

## What Gets Logged

When `use_wandb=True`, LossLab automatically logs:

1. **Configuration**
   - All refinement parameters
   - Loss type, learning rates, etc.

2. **Metrics (Every Iteration)**
   - Loss value
   - Mean confidence (pLDDT)
   - Correlation coefficient
   - B-factor statistics
   - GPU memory usage
   - Time per iteration

3. **Artifacts**
   - Best PDB model
   - Configuration YAML file
   - Trajectory PDB (multi-model)
   - Animated 3D viewer (HTML)
   - Live per-iteration frames (optional)

## Example Usage

### Basic Usage

```python
from LossLab import RefinementConfig, RefinementEngine, RealSpaceLoss

config = RefinementConfig(
    num_iterations=100,
    num_runs=3,
    use_wandb=True,
    wandb_project="protein-refinement",
    wandb_name="target_run1",
    wandb_tags=["cryo-em", "test"],
    wandb_notes="Testing new loss function",
)

engine = RefinementEngine(
    config=config,
    loss_function=loss_fn,
    structure_factor_calculator=sfc,
      pdb_template="input.pdb",  # Required for trajectory + 3D logs
)

results = engine.run(
    reference_coordinates=ref_coords,
    prediction_callback=predictor,
    optimizer=optimizer,
)
```

### Advanced: Manual Logging

```python
from LossLab.refinement.wandb_logger import WandbLogger

# Create custom logger
wandb_logger = WandbLogger(
    project="my-project",
    name="custom-run",
    config={"lr": 0.001, "batch_size": 32},
    tags=["experiment", "v2"],
)

# Log custom metrics
wandb_logger.log({
    "custom_metric": 0.95,
    "another_value": 123,
}, step=iteration)

# Log PDFs
wandb_logger.log_pdb("final_model.pdb", name="final")

# Log coordinates
wandb_logger.log_coordinates(coords_tensor, name="best_coords")

# Finish
wandb_logger.finish()
```


