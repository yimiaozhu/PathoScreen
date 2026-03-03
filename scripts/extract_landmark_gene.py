import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def load_indices(indices_path):
    df = pd.read_csv(indices_path, sep=None, engine='python')
    idx = df['index'].astype(int).to_numpy()
    return idx.astype(np.int64)


def iter_files(input_dir, pattern):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    files = sorted(p.glob(pattern))
    return files


def crop_embedding(path, indices):
    emb = np.load(path)
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]
    if emb.ndim != 2:
        raise ValueError(f"Unexpected embedding shape in {path.name}: {emb.shape}")
    if emb.shape[0] <= indices.max():
        raise ValueError(
            f"{path.name}: embedding has {emb.shape[0]} rows but max index is {indices.max()}."
        )
    return emb[indices, :]


def main():
    ap = argparse.ArgumentParser(description="Crop scFoundation gene embeddings to 978 landmark genes using a verified index file.")
    ap.add_argument("--indices_tsv", type=str, default='./data/cell_matrix/landmark_gene_index.csv', help="TSV/CSV with columns: gene_name, index (0-based).")
    ap.add_argument("--input_dir", type=str, required=True, help="Directory containing embeddings.")
    ap.add_argument("--pattern", type=str, default='*_gene_emb.npy', help="Glob pattern for embedding files.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for cropped embeddings.")
    ap.add_argument("--suffix", type=str, default="_landmark978", help="Suffix appended to output filename.")
    args = ap.parse_args()

    indices = load_indices(args.indices_tsv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = iter_files(args.input_dir, args.pattern)
    if not files:
        raise ValueError(f"No files matched pattern '{args.pattern}' in {args.input_dir}")

    print(f"Loaded {len(indices)} landmark indices from: {args.indices_tsv}")
    print(f"Found {len(files)} embedding files in: {args.input_dir}")

    for fp in files:
        cropped = crop_embedding(fp, indices)
        out_name = fp.stem + args.suffix + fp.suffix
        out_path = out_dir / out_name
        np.save(out_path, cropped.astype(np.float32))
        print(f"✅ {fp.name} -> {out_name}  shape={cropped.shape}")

    print(f"\nDone. Output dir: {out_dir}")


if __name__ == "__main__":
    main()