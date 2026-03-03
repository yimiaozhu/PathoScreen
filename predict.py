import os
import argparse
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import get_config
from src.paths import PathoScreenPaths
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset, GraphCollator
from src.utils import ModelCalibrator


@torch.no_grad()
def predict_softmax_positive(model, dataset, device, batch_size=32, num_workers=0):
    """
    Run forward pass and extract softmax probabilities for the positive class (label=1).
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=GraphCollator(device),
        pin_memory=False,
    )

    all_scores = []
    for mol, adj, cell, _label, atom_lens in loader:
        out = model(mol, adj, cell, atom_lens)

        logits = out[0] if isinstance(out, (tuple, list)) else out
        if not torch.is_tensor(logits):
            raise TypeError(f"Expected logits to be torch.Tensor, got {type(logits)}")

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected logits shape (B,2) for softmax, got {tuple(logits.shape)}")

        probs_pos = torch.softmax(logits, dim=1)[:, 1]
        all_scores.append(probs_pos.detach().cpu())

    return torch.cat(all_scores, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description="PathoScreen Inference (softmax P(class=1))")
    parser.add_argument("--pathway_id", type=int, required=True)
    parser.add_argument("--input_csv", type=str, required=True, help="Candidates CSV with columns: SMILES,cell_id")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path. Default: output/P{pathway_id}/predictions/candidates_pred.csv")

    parser.add_argument("--use_pretrained", action="store_true", help="Flag to use official pre-trained weights from ./checkpoints/ directory.")
    parser.add_argument("--output_root", type=str, default="output", help="Root dir for artifacts (output/P{pathway_id}/...)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Override checkpoint path. Default: output/P{pathway_id}/checkpoints/best.pt")
    parser.add_argument("--calibrator_path", type=str, default=None, help="Override calibrator path (.pkl). Default: output/P{pathway_id}/calibration/isotonic.pkl")
    
    
    parser.add_argument("--emb_pkl", type=str, default=None, help="Path to input_cell_matrix.pkl (dict cell_id -> (978,512)).")
    parser.add_argument("--emb_dir", type=str, default=None, help="Directory containing {cell_id}_scFoundation_input_gene_emb.npy")

    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold applied after calibration (default=0.5).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    
    parser.add_argument("--strict_cell", action="store_true", help="Error if a cell_id embedding is missing (default if using emb_dir).")
    parser.add_argument("--allow_missing_cell", action="store_true", help="If missing cell_id embedding, fall back to zeros (not recommended).")
    args = parser.parse_args()

    strict_cell = not args.allow_missing_cell
    if args.strict_cell:
        strict_cell = True

    if (args.emb_pkl is None) == (args.emb_dir is None):
        raise ValueError("You must provide exactly ONE of --emb_pkl or --emb_dir.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    paths = PathoScreenPaths(Path(args.output_root), args.pathway_id)
    paths.ensure()

    if args.use_pretrained:
        default_ckpt = f"checkpoints/P{args.pathway_id}_final.pt"
        default_calib = f"checkpoints/calibrator_P{args.pathway_id}.pkl"
    else:
        default_ckpt = str(paths.checkpoint_best())
        default_calib = str(paths.calibrator_path())
        if not os.path.exists(default_ckpt) and os.path.exists(str(paths.checkpoint_last())):
            default_ckpt = str(paths.checkpoint_last())

    ckpt_path = args.checkpoint_path if args.checkpoint_path else default_ckpt
    calib_path = args.calibrator_path if args.calibrator_path else default_calib

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n"
                                f"Tip: If you want to use the official weights, add the '--use_pretrained' flag.")

    # Dataset
    dataset = PathoScreenDataset(
        csv_path=args.input_csv,
        mode="inference",
        emb_pkl=args.emb_pkl,
        emb_dir=args.emb_dir,
        strict_cell=strict_cell,
    )

    # Model
    config = get_config(args.pathway_id)
    model = PathoScreen(config, device).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f" Loaded model checkpoint: {ckpt_path}")

    # Inference
    raw_probs = predict_softmax_positive(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Calibration
    final_probs = raw_probs
    calib_path = args.calibrator_path or str(paths.calibrator_path())
    if calib_path and os.path.exists(calib_path):
        calibrator = ModelCalibrator.load(calib_path)
        if calibrator is not None:
            print(f"Applying calibration from {calib_path}")
            final_probs = calibrator.predict(raw_probs)
        else:
            print(" Calibrator could not be loaded. Using RAW probabilities.")
    else:
        print(" No calibrator found. Using RAW probabilities.")


    # 5save
    df = pd.read_csv(args.input_csv)
    if len(df) != len(final_probs):
        print(f"⚠️ Warning: Input rows ({len(df)}) != Output scores ({len(final_probs)}). "
              f"This usually means invalid SMILES were skipped.")
    score_col = f"P{args.pathway_id}_score"
    label_col = f"P{args.pathway_id}_label"
    df[score_col] = final_probs
    df[label_col] = (final_probs >= args.threshold).astype(int)

    out_path = args.output_csv or str(paths.predictions_dir / "candidates_pred.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f" Saved prediction to: {out_path}")


if __name__ == "__main__":
    main()
