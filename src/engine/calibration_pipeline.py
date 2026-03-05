import os
import json
import torch
from pathlib import Path

from src.config import get_config
from src.paths import PathoScreenPaths
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset
from src.utils import InferenceEngine, ModelCalibrator


def run_calibration(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brier_scores = {}
    
    output_root = getattr(args, 'output_root', 'output')
    os.makedirs(output_root, exist_ok=True)
    
    for pid in args.pathway_ids:
        print(f"\n[Calibrate] Processing Pathway P{pid}...")
        paths = PathoScreenPaths(Path("output"), pid) 
        
        # Construct test path
        csv_path = os.path.join(args.test_data_root, f"P{pid}", "test.csv")
        if not os.path.exists(csv_path):
            print(f"Test data not found: {csv_path}. Skipping.")
            continue
            
        dataset = PathoScreenDataset(csv_path, mode='train', emb_pkl=args.emb_path)
        
        # Load Model
        config = get_config(pid)
        model = PathoScreen(config, device).to(device)
        
        ckpt_path = paths.checkpoint_best()
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        # Predict Raw
        engine = InferenceEngine(model, device)
        raw_probs = engine.predict(dataset)
        labels = dataset.get_labels()
        
        # Fit Calibrator
        calibrator = ModelCalibrator(pid)
        bs = calibrator.fit(raw_probs, labels)
        
        save_path = calibrator.save(paths.calibration_dir)
        brier_scores[pid] = float(bs)
        print(f"✅ Saved calibrator to {save_path} (Brier={bs:.4f})")
        
    # Save summary
    summary_path = os.path.join(output_root, "brier_scores.json")
    with open(summary_path, "w") as f:
        json.dump(brier_scores, f, indent=4)
    print(f"\n✅ All calibration finished. Summary saved to {summary_path}")