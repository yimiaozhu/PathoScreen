import argparse
import sys
from src.config import PATHWAY_NAMES

def main():
    parser = argparse.ArgumentParser(prog="pathoscreen", description="PathoScreen: MASLD Perturbation Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Train Command ---
    train_parser = subparsers.add_parser("train", help="Train a model for a specific pathway")
    train_parser.add_argument("--pathway_id", type=int, required=True, choices=list(PATHWAY_NAMES.keys()))
    train_parser.add_argument("--config", type=str, default="configs/default_train.yaml")
    train_parser.add_argument("--output_root", type=str, default="output")
    train_parser.add_argument("--gpu", type=int, default=0)
    train_parser.add_argument("--smiles_cache", type=str, default=None, help="Path to pre-computed SMILES graph pickle")

    # --- Predict Command ---
    pred_parser = subparsers.add_parser("predict", help="Inference on new candidates")
    pred_parser.add_argument("--pathway_id", type=int, required=True)
    pred_parser.add_argument("--input_csv", type=str, required=True)
    pred_parser.add_argument("--emb_pkl", type=str, required=True, help="Path to cell embeddings (.pkl)")
    pred_parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt file")
    pred_parser.add_argument("--output_csv", type=str, default=None)
    pred_parser.add_argument("--smiles_cache", type=str, default=None, help="Path to pre-computed SMILES graph pickle")
    pred_parser.add_argument("--use_pretrained", action="store_true", help="Use official checkpoints")

    # --- Calibrate Command ---
    calib_parser = subparsers.add_parser("calibrate", help="Train isotonic calibrators")
    calib_parser.add_argument("--pathway_ids", nargs="+", type=int, default=[0,1,2,3,4,5,6])
    calib_parser.add_argument("--test_data_root", type=str, required=True)
    calib_parser.add_argument("--emb_path", type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        from src.engine.train_pipeline import run_training
        run_training(args)
    elif args.command == "predict":
        from src.engine.inference_pipeline import run_inference
        run_inference(args)
    elif args.command == "calibrate":
        from src.engine.calibration_pipeline import run_calibration
        run_calibration(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
