import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

GraphValue = Tuple[np.ndarray, np.ndarray]  # (atom_feat[a,34], adj[a,a])

@dataclass
class SmilesGraphCache:
    path: str
    meta_path: Optional[str] = None
    _data: Optional[Dict[str, GraphValue]] = None

    def load(self):
        if self._data is not None:
            return self._data
        if not os.path.exists(self.path):
            self._data = {}
            return self._data
        with open(self.path, "rb") as f:
            self._data = pickle.load(f)
        return self._data

    def get(self, key):
        return self.load().get(key)

    def set(self, key, value):
        self.load()[key] = value

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump(self.load(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_meta(self, meta):
        meta_path = self.meta_path or (self.path.replace(".pkl", ".meta.json"))
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def build_smiles_cache(args):
    from src.data.dataset import mol_to_graph_features
    print(f"Building SMILES graph cache at: {args.output_path}")
    cache = SmilesGraphCache(args.output_path)
    smiles_set = set()

    if getattr(args, 'input_csv', None) and os.path.exists(args.input_csv):
        df = pd.read_csv(args.input_csv)
        if 'SMILES' in df.columns:
            smiles_set.update(df['SMILES'].dropna().astype(str).tolist())
            
    if getattr(args, 'pathways_root', None) and os.path.exists(args.pathways_root):
        for root, _, files in os.walk(args.pathways_root):
            for f in files:
                if f.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, f))
                    if 'SMILES' in df.columns:
                        smiles_set.update(df['SMILES'].dropna().astype(str).tolist())
                        
    if not smiles_set:
        print("No SMILES found. Please check input paths.")
        return

    print(f"Found {len(smiles_set)} unique SMILES. Converting...")
    success_count = 0
    loaded_cache = cache.load()
    
    for smiles in tqdm(smiles_set, desc="Caching Graphs"):
        if smiles not in loaded_cache:
            feat, adj = mol_to_graph_features(smiles)
            if feat is not None:
                cache.set(smiles, (feat, adj))
                success_count += 1
                
    cache.save()
    print(f"Successfully cached {success_count} new SMILES graphs. Total in cache: {len(cache.load())}")