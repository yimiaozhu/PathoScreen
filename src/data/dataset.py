import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem


NUM_ATOM_FEAT = 34
CELL_EMB_SHAPE = (978, 512)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def get_atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
    
    degree = [0, 1, 2, 3, 4, 5, 6]
    
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + \
              [atom.GetIsAromatic()]
    
    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                       [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False, atom.HasProp('_ChiralityPossible')]
    
    return np.array(results, dtype=np.float32)


def get_adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + 2.0 * np.eye(adjacency.shape[0])


def mol_to_graph_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        mol = None
    
    if mol is None:
        return None, None

    Chem.AssignStereochemistry(mol, force=True, cleanIt=True) 

    num_atoms = mol.GetNumAtoms()
    atom_feat = np.zeros((num_atoms, NUM_ATOM_FEAT), dtype=np.float32)
    
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx()] = get_atom_features(atom)
        
    adj_matrix = get_adjacent_matrix(mol)
    return atom_feat, adj_matrix


class PathoScreenDataset(Dataset):
    def __init__(self, csv_path, mode='train', emb_pkl=None, emb_dir=None, strict_cell=True):
        self.mode = mode
        self.strict_cell = strict_cell

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        self.data = pd.read_csv(csv_path)

        required_cols = ['SMILES', 'cell_id', 'label'] if mode == 'train' else ['SMILES', 'cell_id']
        for col in required_cols:
            if col not in self.data.columns:
                found = False
                for c in self.data.columns:
                    if c.lower() == col.lower():
                        self.data.rename(columns={c: col}, inplace=True)
                        found = True
                        break
                if not found:
                    raise ValueError(f"Missing required column '{col}' in {csv_path}")

        if (emb_pkl is None) == (emb_dir is None):
            raise ValueError("You must provide exactly ONE of emb_pkl or emb_dir.")

        self.emb_pkl = emb_pkl
        self.emb_dir = emb_dir
        self.cell_map = None

        if self.emb_pkl is not None:
            if not os.path.exists(self.emb_pkl):
                raise FileNotFoundError(f"Embedding pkl not found: {self.emb_pkl}")
            print(f"[{mode.upper()}] Loading cell embedding map (pkl) from {self.emb_pkl}...")
            with open(self.emb_pkl, 'rb') as f:
                self.cell_map = pickle.load(f)
            if not isinstance(self.cell_map, dict):
                raise TypeError(f"Expected emb_pkl to load a dict, got: {type(self.cell_map)}")
            print(f"[{mode.upper()}] Loaded {len(self.cell_map)} unique cell embeddings from pkl.")

        if self.emb_dir is not None:
            if not os.path.isdir(self.emb_dir):
                raise FileNotFoundError(f"Embedding directory not found: {self.emb_dir}")
            print(f"[{mode.upper()}] Using cell embedding directory: {self.emb_dir}")

    def get_labels(self):
        if 'label' in self.data.columns:
            return self.data['label'].values.astype(int)
        return np.zeros(len(self.data), dtype=int)

    def __len__(self):
        return len(self.data)

    def _resolve_cell_key(self, cell_id):
        cid = str(cell_id).strip()
        return [cid, cid.upper(), cid.lower()]

    def _load_cell_emb_from_pkl(self, cell_id):
        assert self.cell_map is not None
        for key in self._resolve_cell_key(cell_id):
            if key in self.cell_map:
                emb = self.cell_map[key]
                emb = np.asarray(emb, dtype=np.float32)
                return emb
        raise KeyError(f"cell_id '{cell_id}' not found in embedding pkl keys.")

    def _load_cell_emb_from_dir(self, cell_id):
        for key in self._resolve_cell_key(cell_id):
            fp = os.path.join(self.emb_dir, f"{key}_scFoundation_input_gene_emb.npy")
            if os.path.exists(fp):
                emb = np.load(fp)
                if emb.ndim == 3 and emb.shape[0] == 1:
                    emb = emb[0]
                emb = np.asarray(emb, dtype=np.float32)
                return emb
        raise FileNotFoundError(
            f"cell_id '{cell_id}' embedding file not found in dir '{self.emb_dir}'. "
            f"Expected one of: {cell_id}_scFoundation_input_gene_emb.npy (case variants)."
        )

    def _get_cell_emb(self, cell_id):
        try:
            if self.emb_pkl is not None:
                emb = self._load_cell_emb_from_pkl(cell_id)
            else:
                emb = self._load_cell_emb_from_dir(cell_id)
        except Exception as e:
            if self.strict_cell:
                raise
            print(f"⚠️ Warning: {e}. Falling back to zeros for cell_id='{cell_id}'.")
            emb = np.zeros(CELL_EMB_SHAPE, dtype=np.float32)

        if emb.shape != CELL_EMB_SHAPE:
            raise ValueError(
                f"cell embedding for '{cell_id}' has shape {emb.shape}, expected {CELL_EMB_SHAPE}. "
                f"Please ensure embeddings are already cropped to 978 landmark genes."
            )
        return emb

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = str(row['SMILES']).strip()
        cell_id = str(row['cell_id']).strip()

        atom_feat, adj = mol_to_graph_features(smiles)
        if atom_feat is None:
            return self.__getitem__((idx + 1) % len(self))

        cell_emb = self._get_cell_emb(cell_id)

        label = float(row['label']) if 'label' in row else 0.0

        return {
            'mol': torch.FloatTensor(atom_feat),
            'adj': torch.FloatTensor(adj),
            'cell': torch.FloatTensor(cell_emb),
            'label': torch.tensor(label, dtype=torch.float32)
        }

class GraphCollator:
    def __init__(self, device):
        self.device = device

    def __call__(self, batch):
        batch_size = len(batch)

        atom_lens = [b['mol'].shape[0] for b in batch]
        max_atoms = max(atom_lens)

        mol_batch = torch.zeros(batch_size, max_atoms, NUM_ATOM_FEAT, device=self.device)
        adj_batch = torch.zeros(batch_size, max_atoms, max_atoms, device=self.device)

        cell_batch = torch.stack([b['cell'] for b in batch]).to(self.device)
        label_batch = torch.stack([b['label'] for b in batch]).to(self.device)

        for i, b in enumerate(batch):
            length = atom_lens[i]
            mol_batch[i, :length, :] = b['mol'].to(self.device)
            adj_batch[i, :length, :length] = b['adj'].to(self.device)

        return mol_batch, adj_batch, cell_batch, label_batch, atom_lens