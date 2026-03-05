import argparse
import sys
from src.config import PATHWAY_NAMES

def main():
    parser = argparse.ArgumentParser(prog="pathoscreen", description="PathoScreen: MASLD Perturbation Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # ==========================================
    # --- 1. Train Command ---
    # ==========================================
    train_parser = subparsers.add_parser("train", help="Train a model for a specific pathway")
    train_parser.add_argument("--pathway_id", type=int, required=True, choices=list(PATHWAY_NAMES.keys()))
    train_parser.add_argument("--config", type=str, default="configs/default_train.yaml")
    train_parser.add_argument("--output_root", type=str, default="output")
    train_parser.add_argument("--gpu", type=int, default=0)
    train_parser.add_argument("--smiles_cache", type=str, default=None, help="Path to pre-computed SMILES graph cache")

    # ==========================================
    # --- 2. Predict Command ---
    # ==========================================
    pred_parser = subparsers.add_parser("predict", help="Inference on new candidates")
    pred_parser.add_argument("--pathway_id", type=int, required=True, choices=list(PATHWAY_NAMES.keys()))
    pred_parser.add_argument("--input_csv", type=str, required=True)
    pred_parser.add_argument("--emb_pkl", type=str, required=True, help="Path to cell embeddings (.pkl)")
    pred_parser.add_argument("--emb_dir", type=str, default=None)
    pred_parser.add_argument("--checkpoint", type=str, default=None)
    pred_parser.add_argument("--output_csv", type=str, default=None)
    pred_parser.add_argument("--output_root", type=str, default="output")
    pred_parser.add_argument("--smiles_cache", type=str, default=None)
    pred_parser.add_argument("--use_pretrained", action="store_true")
    pred_parser.add_argument("--threshold", type=float, default=0.5)
    pred_parser.add_argument("--batch_size", type=int, default=32)
    pred_parser.add_argument("--calibrator_path", type=str, default=None)
    pred_parser.add_argument("--allow_missing_cell", action="store_true")

    # ==========================================
    # --- 3. Calibrate Command ---
    # ==========================================
    calib_parser = subparsers.add_parser("calibrate", help="Train isotonic calibrators")
    calib_parser.add_argument("--pathway_ids", nargs="+", type=int, default=[0,1,2,3,4,5,6])
    calib_parser.add_argument("--test_data_root", type=str, required=True)
    calib_parser.add_argument("--emb_path", type=str, required=True)
    calib_parser.add_argument("--output_root", type=str, default="output")

    # ==========================================
    # --- 4. Build Cache Command ---
    # ==========================================
    cache_parser = subparsers.add_parser("build-cache", help="Pre-compute SMILES graph cache")
    cache_parser.add_argument("--input_csv", type=str, default=None, help="Candidates CSV file")
    cache_parser.add_argument("--pathways_root", type=str, default=None, help="Root directory containing pathway datasets")
    cache_parser.add_argument("--output_path", type=str, required=True, help="Output path for the .pkl cache")

    # ==========================================
    # --- 5. Rank Command ---
    # ==========================================
    rank_parser = subparsers.add_parser("rank", help="Rank screened candidates across all pathways")
    rank_parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing P0-P6 prediction CSVs")
    rank_parser.add_argument("--brier_json", type=str, required=True, help="Path to brier_scores.json from calibration")
    rank_parser.add_argument("--output_file", type=str, required=True, help="Output ranked CSV")

    args = parser.parse_args()

    # Route commands to their respective core logic in src/
    if args.command == "train":
        from src.engine.train_pipeline import run_training
        run_training(args)
    elif args.command == "predict":
        from src.engine.inference_pipeline import run_inference
        run_inference(args)
    elif args.command == "calibrate":
        from src.engine.calibration_pipeline import run_calibration
        run_calibration(args)
    elif args.command == "build-cache":
        from src.data.cache import build_smiles_cache 
        build_smiles_cache(args)
    elif args.command == "rank":
        from src.utils import generate_final_ranking
        generate_final_ranking(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()