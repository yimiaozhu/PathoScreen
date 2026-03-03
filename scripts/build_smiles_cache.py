import argparse
import glob
import os
from datetime import datetime
from typing import Set

import pandas as pd
from tqdm import tqdm

from ..src.data.dataset import mol_to_graph_features, NUM_ATOM_FEAT
from ..src.utils import canonicalize_smiles
from ..src.data.cache import SmilesGraphCache


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathways_root", type=str, default="data/pathways")
    ap.add_argument("--out_dir", type=str, default="data/cache")
    ap.add_argument("--cache_name", type=str, default="smiles_graph_cache_v1.pkl")
    ap.add_argument("--isomeric", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, args.cache_name)
    cache = SmilesGraphCache(cache_path)

    isomeric = bool(args.isomeric)

    smiles_raw = collect_smiles(args.pathways_root)
    smiles = []
    for s in smiles_raw:
        try:
            smiles.append(canonicalize_smiles(s, isomeric=isomeric))
        except Exception:
            smiles.append(s)

    smiles_unique = sorted(set(smiles))
    data = cache.load()

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

    cache.save()
    cache.save_meta({
        "version": "v1",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "pathways_root": args.pathways_root,
        "isomeric": isomeric,
        "num_atom_feat": NUM_ATOM_FEAT,
        "num_unique_smiles_total": len(smiles_unique),
        "num_cached_total": len(data),
        "num_new_created": created,
        "num_skipped_existing": skipped,
    })

    print(f"Saved cache: {cache_path}")
    print(f"Unique SMILES scanned: {len(smiles_unique)} | newly cached: {created} | already existed: {skipped}")


if __name__ == "__main__":
    main()
