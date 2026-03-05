import os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
from .data.dataset import GraphCollator


# Chemical utils
def canonicalize_smiles(smiles, isomeric):
    """SMILES standardization"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)
    except:
        pass
    return None


# Calibration utils
class ModelCalibrator:
    """Wrapper for Isotonic Regression calibration"""
    def __init__(self, pathway_id):
        self.pid = pathway_id
        self.iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        self.brier_score = None

    def fit(self, probs, labels):
        self.iso_reg.fit(probs, labels)
        calib_probs = self.iso_reg.predict(probs)
        self.brier_score = brier_score_loss(labels, calib_probs)
        return self.brier_score

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self.iso_reg.predict(probs)

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"P{self.pid}_isotonic.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        
        if isinstance(obj, IsotonicRegression):
            instance = cls(pathway_id=None)
            instance.calibrator = obj
            instance.brier_score = None
            return instance

        return obj


# Inference engine
class InferenceEngine:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, dataset, batch_size=32):
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=GraphCollator(self.device),
            shuffle=False,
            num_workers=0
        )
        
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", leave=False):
                mol, adj, cell, _, atom_nums = batch
                
                logits, _ = self.model(mol, adj, cell, atom_nums)
                
                probs = torch.softmax(logits, dim=1)[:, 1]
                all_scores.extend(probs.cpu().numpy())
                
        return np.array(all_scores)

# Ranking engine
def generate_final_ranking(args):
    import glob
    import json
    import pandas as pd
    
    print(f"Generating final candidate ranking from predictions in: {args.pred_dir}")
    
    OFFICIAL_BRIER_SCORES = {
        "0": 0.0475,  # Fatty acid metabolism
        "1": 0.0424,  # Liver insulin signaling
        "2": 0.0813,  # Antioxidant defense
        "3": 0.0545,  # Anti-apoptosis
        "4": 0.0297,  # Mitochondrial function
        "5": 0.0529,  # Anti-inflammation
        "6": 0.0448   # TGF-beta signaling
    }

    brier_scores = {}

    if getattr(args, 'brier_json', None) and os.path.exists(args.brier_json):
        with open(args.brier_json, 'r') as f:
            brier_scores = json.load(f)
        print(f"Loaded CUSTOM Brier scores from {args.brier_json}")
    else:
        print("Using OFFICIAL pre-trained Brier scores for ranking calculation.")
        brier_scores = OFFICIAL_BRIER_SCORES
        
    pred_files = glob.glob(os.path.join(args.pred_dir, "**", "*_pred.csv"), recursive=True)
    if not pred_files:
        print("No prediction CSV files found in the specified directory.")
        return

    df_merged = None
    for pf in pred_files:
        df = pd.read_csv(pf)
        if df_merged is None:
            df_merged = df[['SMILES', 'cell_id']].copy()
            
        score_cols = [c for c in df.columns if c.endswith('_score')]
        label_cols = [c for c in df.columns if c.endswith('_label')]
        
        if not score_cols or not label_cols:
            continue
            
        score_col = score_cols[0]
        label_col = label_cols[0]
        
        df_merged[score_col] = df[score_col]
        df_merged[label_col] = df[label_col]

    all_score_cols = [c for c in df_merged.columns if c.endswith('_score')]
    all_label_cols = [c for c in df_merged.columns if c.endswith('_label')]
    
    # 1. Vote count
    df_merged['Vote_Count'] = df_merged[all_label_cols].sum(axis=1)
    
    # 2. Pathway Consensus Score (PCS) 
    pcs_num = 0.0
    pcs_den = 0.0
    
    for score_col in all_score_cols:
        pid = score_col.split('_')[0].replace('P', '')
        bs = brier_scores.get(pid, 1.0) 
        weight = 1.0 / (bs + 1e-6) 
        pcs_num += df_merged[score_col] * weight
        pcs_den += weight
            
    df_merged['PCS'] = pcs_num / pcs_den if pcs_den > 0 else 0
    
    # 3. Vote_Count → PCS (descending)
    df_merged = df_merged.sort_values(by=['Vote_Count', 'PCS'], ascending=[False, False])
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_merged.to_csv(args.output_file, index=False)
    print(f"Final ranking saved to {args.output_file}")