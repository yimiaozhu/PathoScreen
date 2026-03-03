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
        path = os.path.join(output_dir, f"calibrator_P{self.pid}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(calibrator_path):
        if not os.path.exists(calibrator_path):
            return None
        with open(calibrator_path, "rb") as f:
            return pickle.load(f)


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