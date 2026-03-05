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

    # 1. Setup Paths
    paths = PathoScreenPaths(Path(args.output_root), args.pathway_id)
    paths.ensure()

    # Determine Checkpoint (Support Path B: Pretrained)
    if hasattr(args, 'use_pretrained') and args.use_pretrained:
        ckpt_path = f"checkpoints/P{args.pathway_id}_final.pt" # Assuming standard location
        print(f"Using official pretrained model: {ckpt_path}")
    elif args.checkpoint:
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
    # Handle strictness logic
    strict_cell = not args.allow_missing_cell
    if args.strict_cell: strict_cell = True

    dataset = PathoScreenDataset(
        csv_path=args.input_csv,
        mode="inference",
        emb_pkl=args.emb_pkl,
        emb_dir=args.emb_dir,
        strict_cell=strict_cell,
        smiles_cache_path=args.smiles_cache
    )

    # 4. Run Inference
    engine = InferenceEngine(model, device)
    # Note: InferenceEngine.predict returns raw probabilities (softmax)
    # We need to ensure InferenceEngine in utils.py returns what we expect.
    # Based on context utils.py, it returns softmax[:, 1].
    raw_probs = engine.predict(dataset, batch_size=args.batch_size)

    # 5. Calibration
    final_probs = raw_probs
    
    # Determine Calibrator Path
    # If user didn't specify, try to find one in the output dir
    calib_path = args.calibrator_path
    if not calib_path:
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
    # Align lengths if dataset skipped invalid SMILES
    if len(df) != len(final_probs):
        print(f"⚠️ Warning: Input rows ({len(df)}) != Output scores ({len(final_probs)}).")
        # In a real scenario, we should handle alignment better (e.g. by returning IDs from dataset)
        # For now, we assume dataset order is preserved for valid entries.
        # Truncate df to match (assuming tail was cut) or raise error.
        # Ideally, Dataset should return indices.
        pass

    score_col = f"P{args.pathway_id}_score"
    label_col = f"P{args.pathway_id}_label"
    
    df[score_col] = final_probs
    df[label_col] = (final_probs >= args.threshold).astype(int)

    out_path = args.output_csv or str(paths.predictions_dir / "candidates_pred.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved predictions: {out_path}")