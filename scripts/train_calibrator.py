import os
import json
import torch
import numpy as np
import argparse
from src.config import get_config
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset
from src.utils import InferenceEngine
from src.utils import ModelCalibrator

def main():
    parser = argparse.ArgumentParser(description="Train Isotonic Calibrators (Step 1)")
    parser.add_argument("--pathway_ids", nargs="+", type=int, default=[0,1,2,3,4,5,6])
    parser.add_argument("--test_data_root", type=str, required=True, help="Root dir containing P0/test.csv, etc.")
    parser.add_argument("--emb_path", type=str, required=True, help="Path to input_cell_matrix.pkl")
    parser.add_argument("--output_dir", type=str, default="calibration_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brier_scores = {}

    for pid in args.pathway_ids:
        print(f"\n Processing Pathway P{pid}...")
        
        csv_path = os.path.join(args.test_data_root, f"P{pid}", "test.csv")
        dataset = PathoScreenDataset(csv_path, args.emb_path, mode='train') 
        
        config = get_config(pid)
        model = PathoScreen(config, device).to(device)
        ckpt_path = f"checkpoints/P{pid}.pt"
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            print(f"⚠️ Checkpoint not found: {ckpt_path}, skipping.")
            continue

        engine = InferenceEngine(model, device)
        raw_probs = engine.predict(dataset)
        labels = dataset.get_labels()

        calibrator = ModelCalibrator(pid)
        bs = calibrator.fit(raw_probs, labels)
        
        calibrator.save(args.output_dir)
        brier_scores[pid] = float(bs)
        print(f"✅ P{pid} Calibrated. Brier Score: {bs:.4f}")

    with open(os.path.join(args.output_dir, "brier_scores.json"), "w") as f:
        json.dump(brier_scores, f, indent=4)
    print(f"\n Brier scores saved to {args.output_dir}/brier_scores.json")

if __name__ == "__main__":
    main()