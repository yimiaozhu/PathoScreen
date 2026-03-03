import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np

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
