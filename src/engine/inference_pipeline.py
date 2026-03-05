import os
import torch
import pandas as pd
from pathlib import Path

from src.config import get_config
from src.paths import PathoScreenPaths
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset
from src.utils import ModelCalibrator, InferenceEngine

def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PathoScreen] Inference P{args.pathway_id} on {device}")

    # Safely get CLI arguments with fallbacks
    output_root = getattr(args, 'output_root', 'output')
    batch_size = getattr(args, 'batch_size', 32)
    threshold = getattr(args, 'threshold', 0.5)
    allow_missing_cell = getattr(args, 'allow_missing_cell', False)

    # 1. Setup Paths
    paths = PathoScreenPaths(Path(output_root), args.pathway_id)
    paths.ensure()

    # Determine Checkpoint (Support Path B: Pretrained)
    if getattr(args, 'use_pretrained', False):
        ckpt_path = f"checkpoints/P{args.pathway_id}.pt" # Updated to match your actual checkpoints dir structure
        print(f"Using official pretrained model: {ckpt_path}")
    elif getattr(args, 'checkpoint', None):
        ckpt_path = args.checkpoint
    else:
        # Default to trained best
        ckpt_path = str(paths.checkpoint_best())
        
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 2. Load Model
    config = get_config(args.pathway_id)
    model = PathoScreen(config, device).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded weights: {ckpt_path}")

    # 3. Load Data
    strict_cell = not allow_missing_cell

    dataset = PathoScreenDataset(
        csv_path=args.input_csv,
        mode="inference",
        emb_pkl=args.emb_pkl,
        emb_dir=getattr(args, 'emb_dir', None),
        strict_cell=strict_cell,
        smiles_cache_path=getattr(args, 'smiles_cache', None)
    )

    # 4. Run Inference
    engine = InferenceEngine(model, device)
    raw_probs = engine.predict(dataset, batch_size=batch_size)

    # 5. Calibration
    final_probs = raw_probs
    
    # Determine Calibrator Path
    calib_path = getattr(args, 'calibrator_path', None)
    if not calib_path:
        # Try to use the pre-trained isotonic calibrator if use_pretrained is flagged
        if getattr(args, 'use_pretrained', False):
            default_calib = f"checkpoints/P{args.pathway_id}_isotonic.pkl"
        else:
            default_calib = paths.calibrator_path()
            
        if os.path.exists(default_calib):
            calib_path = str(default_calib)

    if calib_path and os.path.exists(calib_path):
        calibrator = ModelCalibrator.load(calib_path)
        if calibrator:
            print(f"Applying calibration: {calib_path}")
            final_probs = calibrator.predict(raw_probs)
        else:
            print("⚠️ Failed to load calibrator. Using raw probabilities.")
    else:
        print("ℹ️ No calibrator found/provided. Using raw probabilities.")

    # 6. Save Results
    df = pd.read_csv(args.input_csv)
    if len(df) != len(final_probs):
        print(f"⚠️ Warning: Input rows ({len(df)}) != Output scores ({len(final_probs)}).")

    score_col = f"P{args.pathway_id}_score"
    label_col = f"P{args.pathway_id}_label"
    
    df[score_col] = final_probs
    df[label_col] = (final_probs >= threshold).astype(int)

    out_path = getattr(args, 'output_csv', None) or str(paths.predictions_dir / f"P{args.pathway_id}_candidates_pred.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")