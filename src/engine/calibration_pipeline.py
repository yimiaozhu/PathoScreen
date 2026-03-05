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
    
    # Output dir logic
    # If args.output_root is generic, we save per pathway.
    # But calibration usually saves to P{id}/calibration/isotonic.pkl
    
    for pid in args.pathway_ids:
        print(f"\n[Calibrate] Processing Pathway P{pid}...")
        paths = PathoScreenPaths(Path("output"), pid) # Assuming default output root or passed via args if we added it
        
        # Construct test path
        csv_path = os.path.join(args.test_data_root, f"P{pid}", "test.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ Test data not found: {csv_path}. Skipping.")
            continue
            
        dataset = PathoScreenDataset(csv_path, args.emb_path, mode='train') # mode='train' to load labels
        
        # Load Model
        config = get_config(pid)
        model = PathoScreen(config, device).to(device)
        
        ckpt_path = paths.checkpoint_best()
        if not os.path.exists(ckpt_path):
            print(f"⚠️ Checkpoint not found: {ckpt_path}. Skipping.")
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
    with open("output/brier_scores.json", "w") as f:
        json.dump(brier_scores, f, indent=4)