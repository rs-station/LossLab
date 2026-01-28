"""Example showing trajectory saving and W&B logging."""

import gemmi
import numpy as np
import torch
from SFC_Torch import PDBParser
import SFC_Torch as sfc

from LossLab import RealSpaceLoss, RefinementConfig, RefinementEngine


def main():
    """Refinement with automatic trajectory and W&B logging."""
    
    # ========== 1. SETUP ==========
    
    # Load map and structure
    target_map = gemmi.read_ccp4_map("./masked_x395.ccp4")
    target_map.setup(0.0)
    
    input_pdb = PDBParser("./x395_no_altB.pdb")
    input_pdb.set_spacegroup("P 1")
    input_pdb.set_unitcell(target_map.grid.unit_cell)
    
    # Structure factor calculator
    structure_factor_calc = sfc.SFcalculator(
        input_pdb,
        dmin=1.8,
        mode="xray",
        device="cuda:3",
    )
    structure_factor_calc.inspect_data()
    structure_factor_calc.gridsize = [
        target_map.grid.nu,
        target_map.grid.nv,
        target_map.grid.nw,
    ]
    
    # Loss function
    loss_fn = RealSpaceLoss(
        target_map=target_map,
        pdb_obj=input_pdb,
        device="cuda:3",
        loss_type="l2",
        mask_center=np.array([-0.097, 44.482, 29.168]),
        mask_radius=18.0,
    )
    
    # ========== 2. CONFIGURE WITH TRACKING ==========
    
    config = RefinementConfig(
        num_iterations=3,
        num_runs=1,
        learning_rate_additive=1e-3,
        learning_rate_multiplicative=1e-3,
        loss_type="l2",
        output_dir="./outputA",
        run_note="_",
        
        # Trajectory saving
        save_best_pdb=True,              # Save best PDB
        save_trajectory_pdb=True,         # Save full trajectory
        save_trajectory_interval=1,      # Save every iteration
        
        # W&B logging (optional)
        # Automatically logs 3D interactive structures to W&B dashboard!
        use_wandb=True,
        wandb_entity="af840-columbia-university",  # Your team/org
        wandb_project="protein-viz_fixed",
        wandb_name="vis-tests",
        wandb_tags=["cryo-em", "x395", "test"],
        wandb_notes="Testing trajectory and W&B integration with 3D visualization",
    )
    
    # ========== 3. CREATE ENGINE WITH PDB TEMPLATE ==========
    
    # Pass the file path for trajectory saving (mdtraj will handle it)
    engine = RefinementEngine(
        config=config,
        loss_function=loss_fn,
        structure_factor_calculator=structure_factor_calc,
        pdb_template="./x395_no_altB.pdb",
    )
    
    # Reference coordinates
    reference_coords = torch.tensor(
        input_pdb.atom_pos,
        device="cuda:3",
        dtype=torch.float32,
    )
    
    # ========== 4. DEFINE PREDICTOR ==========
    
    class DummyPredictor:
        """Dummy predictor for demonstration."""
        
        def __init__(self, coords):
            self.coords = coords
        
        def __call__(self):
            # Add small random noise
            noisy_coords = self.coords + torch.randn_like(self.coords) * 0.1
            # Return [N, 4]: xyz + confidence
            confidence = torch.ones(len(noisy_coords), device=self.coords.device) * 0.9
            return torch.cat([noisy_coords, confidence.unsqueeze(-1)], dim=-1)
    
    predictor = DummyPredictor(reference_coords)
    
    # ========== 5. RUN REFINEMENT ==========
    
    results = engine.run(
        reference_coordinates=reference_coords,
        prediction_callback=predictor,
        optimizer=None,  # No optimization for dummy example
    )
    
    # ========== 6. RESULTS ==========
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best loss: {results['loss']:.6f}")
    print(f"Best run: {results['run_id']}")
    print(f"Best iteration: {results['iteration']}")
    print(f"\nOutput directory: {config.output_dir / config.run_note}")
    print("\nSaved files:")
    print("  - trajectory/best.pdb          (best model)")
    print("  - trajectory/A_trajectory.pdb  (run A trajectory)")
    print("  - trajectory/A_0000.pdb        (individual snapshots)")
    print("  - A_metrics.npz                (metrics)")
    print("  - config.yaml                  (configuration)")
    
    if config.use_wandb:
        print("\n View results at: https://wandb.ai/")


if __name__ == "__main__":
    main()
