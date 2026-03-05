import argparse
import glob
import os
import pickle
from datetime import datetime
from typing import Set

import pandas as pd
from tqdm import tqdm

from src.data.dataset import mol_to_graph_features, NUM_ATOM_FEAT
from src.utils import canonicalize_smiles


def collect_smiles(pathways_root):
    paths = glob.glob(os.path.join(pathways_root, "*", "train.csv")) + glob.glob(os.path.join(pathways_root, "*", "test.csv"))
    smiles_set: Set[str] = set()
    for p in paths:
        df = pd.read_csv(p)
        if "SMILES" not in df.columns:
            for c in df.columns:
                if c.lower() == "smiles":
                    df = df.rename(columns={c: "SMILES"})
                    break
        if "SMILES" not in df.columns:
            raise ValueError(f"Missing SMILES column in {p}. Found: {list(df.columns)}")
        smiles_set.update([str(s).strip() for s in df["SMILES"].tolist()])
    return smiles_set


def collect_smiles_from_file(csv_path):
    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        # Try case insensitive
        for c in df.columns:
            if c.lower() == "smiles":
                df = df.rename(columns={c: "SMILES"})
                break
    if "SMILES" not in df.columns:
        raise ValueError(f"Missing SMILES column in {csv_path}")
    return set([str(s).strip() for s in df["SMILES"].tolist()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathways_root", type=str, default=None, help="Directory with P0/train.csv structure")
    ap.add_argument("--input_csv", type=str, default=None, help="Single CSV file (e.g. candidates.csv)")
    ap.add_argument("--output_path", type=str, default="data/cache/smiles_graph.pkl")
    ap.add_argument("--isomeric", type=int, default=1)
    args = ap.parse_args()

    if not args.pathways_root and not args.input_csv:
        raise ValueError("Must provide either --pathways_root or --input_csv")

    isomeric = bool(args.isomeric)

    if args.input_csv:
        print(f"Collecting SMILES from file: {args.input_csv}")
        smiles_raw = collect_smiles_from_file(args.input_csv)
    else:
        print(f"Collecting SMILES from root: {args.pathways_root}")
        smiles_raw = collect_smiles(args.pathways_root)

    smiles = []
    for s in smiles_raw:
        try:
            smiles.append(canonicalize_smiles(s, isomeric=isomeric))
        except Exception:
            smiles.append(s)

    smiles_unique = sorted(set(smiles))
    
    # Load existing if available
    data = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded existing cache with {len(data)} entries.")

    created = 0
    skipped = 0
    for s in tqdm(smiles_unique, desc="Building SMILES graph cache"):
        if s in data:
            skipped += 1
            continue
        atom_feat, adj = mol_to_graph_features(s)
        if atom_feat is None:
            continue
        data[s] = (atom_feat, adj)
        created += 1

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Saved cache: {args.output_path}")
    print(f"Unique SMILES scanned: {len(smiles_unique)} | newly cached: {created} | already existed: {skipped}")


if __name__ == "__main__":
    main()
