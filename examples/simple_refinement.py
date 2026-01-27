"""Simple example script demonstrating the clean LossLab API.

This script shows how the refactored code reduces the main refinement
loop to just a few lines of code.
"""

import glob
import pickle

import gemmi
import numpy as np
import rocket
import SFC_Torch as sfc
import torch
from openfold.config import model_config
from rocket import coordinates as rk_coordinates
from rocket import refinement_utils as rkrf_utils
from rocket import utils as rk_utils
from SFC_Torch import PDBParser

# Import LossLab components
from LossLab import RealSpaceLoss, RefinementConfig, RefinementEngine


def main():
    """Main refinement script."""

    # ========== 1. INITIALIZATION (One section) ==========

    # Load your target map and structure
    target_map = gemmi.read_ccp4_map("./masked_x395.ccp4")
    target_map.setup(0.0)

    input_pdb = PDBParser("./x395-pred-aligned.pdb")
    input_pdb.set_spacegroup("P 1")
    input_pdb.set_unitcell(target_map.grid.unit_cell)

    # Setup structure factor calculator
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

    # Create loss function
    loss_fn = RealSpaceLoss(
        target_map=target_map,
        pdb_obj=input_pdb,
        device="cuda:3",
        loss_type="l2",
        mask_center=np.array([-0.097, 44.482, 29.168]),
        mask_radius=18.0,
    )

    # Configure refinement
    config = RefinementConfig(
        num_iterations=30,
        num_runs=1,
        learning_rate_additive=1e-3,
        learning_rate_multiplicative=1e-3,
        loss_type="l2",
        output_dir="./output",
        run_note="jan26",
        save_every_n_iterations=50,
        early_stopping_patience=150,
    )

    # Create refinement engine
    engine = RefinementEngine(
        config=config,
        loss_function=loss_fn,
        structure_factor_calculator=structure_factor_calc,
    )

    # Reference coordinates
    reference_coords = torch.tensor(
        input_pdb.atom_pos,
        device="cuda:3",
        dtype=torch.float32,
    )

    # ========== 2. MODEL SETUP ==========

    # Load AlphaFold model (exactly like pandda_realspacerefine_x395.py for now)
    device = "cuda:3"
    preset = "model_1_ptm"

    # Initialize model
    af_model = rocket.MSABiasAFv3(model_config(preset, train=True), preset).to(device)
    af_model.freeze()

    # Load processed features from pickle (exactly like pandda_realspacerefine_x395.py)
    pickle_path = "./"
    pickle_files = glob.glob(f"{pickle_path}/*processed_features.pickle")

    with open(pickle_files[0], "rb") as file:
        processed_features = pickle.load(file)

    # Move to device
    device_processed_features = rk_utils.move_tensors_to_device(
        processed_features, device=device
    )

    # Save initial features for resetting - MUST DETACH like ROCKET!
    features_at_it_start = device_processed_features["msa_feat"].detach().clone()

    # Initialize bias and optimizer using ROCKET's init_bias
    device_processed_features, optimizer, _ = rkrf_utils.init_bias(
        device_processed_features=device_processed_features,
        bias_version=3,  # MSABiasAFv3
        device=device,
        lr_a=config.learning_rate_additive,
        lr_m=config.learning_rate_multiplicative,
        weight_decay=None,
        starting_bias="./best_msa_bias_A_135.pt",
        starting_weights="./best_feat_weights_A_135.pt",
    )

    # Define prediction callback
    class AlphaFoldPredictor:
        """Prediction callback that returns [N, 4]: xyz + confidence."""

        def __init__(self, model, features, features_backup, pdb_obj):
            self.model = model
            self.features = features
            self.features_backup = features_backup
            self.pdb_obj = pdb_obj
            self.prevs = None

        def __call__(self):
            """Return [N, 4] tensor: [x, y, z, confidence]."""
            # Reset features to starting point - MUST DETACH like ROCKET!
            self.features["msa_feat"] = self.features_backup.detach().clone()

            # Handle recycling internally
            if self.prevs is None:
                # First call: initialize with recycling, no bias
                outputs, self.prevs = self.model(
                    self.features,
                    [None, None, None],
                    num_iters=3,
                    bias=False,
                )
                # Detach prevs to avoid backprop through them
                self.prevs = [p.detach() for p in self.prevs]
                deep_copied_prevs = [p.clone().detach() for p in self.prevs]
                outputs, _ = self.model(
                    self.features,
                    deep_copied_prevs,
                    num_iters=1,
                    bias=True,
                )
            else:
                # Subsequent calls: use cached prevs, single iteration with bias
                deep_copied_prevs = [p.clone().detach() for p in self.prevs]
                outputs, _ = self.model(
                    self.features,
                    deep_copied_prevs,
                    num_iters=1,
                    bias=True,
                )

            # Extract all atoms using ROCKET's utility
            xyz_orth_sfc, plddts = rk_coordinates.extract_allatoms(
                outputs,
                self.features,
                self.pdb_obj.cra_name,
            )
            # Return [N_atoms, 4]: xyz + confidence
            return torch.cat([xyz_orth_sfc, plddts.unsqueeze(-1)], dim=-1)

    # Create predictor instance
    predictor = AlphaFoldPredictor(
        af_model,
        device_processed_features,
        features_at_it_start,
        input_pdb,
    )

    # ========== 3. TRAINING LOOP (One line!) ==========

    results = engine.run(
        reference_coordinates=reference_coords,
        prediction_callback=predictor,
        optimizer=optimizer,
    )

    # ========== 4. LOGGING/RESULTS (One line) ==========

    print("\nRefinement complete!")
    print(f"Best loss: {results['loss']:.6f}")
    print(f"Best run: {results['run_id']}")
    print(f"Best iteration: {results['iteration']}")
    print(f"Output directory: {config.output_dir / config.run_note}")


if __name__ == "__main__":
    main()
