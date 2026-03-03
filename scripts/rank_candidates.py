import os
import json
import argparse
import pandas as pd
import numpy as np
from src.utils import canonicalize_smiles

def main():
    parser = argparse.ArgumentParser(description="Rank Candidates via PCS (Step 3)")
    parser.add_argument("--pred_dir", type=str, required=True, help="Dir containing PX_pred.csv files")
    parser.add_argument("--brier_json", type=str, required=True, help="Path to brier_scores.json")
    parser.add_argument("--output_file", type=str, default="final_rank.csv")
    args = parser.parse_args()

    with open(args.brier_json, "r") as f:
        brier_scores = json.load(f)
    
    weights = {int(k): 1.0 / (v + 1e-6) for k, v in brier_scores.items()}
    print("🔹 Pathway Weights:", weights)

    master_df = None
    pathway_ids = sorted([int(k) for k in brier_scores.keys()])
    
    for pid in pathway_ids:
        csv_path = os.path.join(args.pred_dir, f"P{pid}_pred.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        
        df['canon_smiles'] = df['SMILES'].apply(lambda x: canonicalize_smiles(x, isomeric=True))
        df = df.dropna(subset=['canon_smiles'])
        
        df = df.groupby('canon_smiles', as_index=False).agg({
            f"P{pid}_score": 'max',
            f"P{pid}_label": 'max'
        })
        
        if master_df is None:
            master_df = df
        else:
            master_df = pd.merge(master_df, df, on='canon_smiles', how='outer')

    print("Calculating PCS...")
    weighted_sum = 0
    total_weight = 0
    total_votes = 0
    
    for pid in pathway_ids:
        score_col = f"P{pid}_score"
        label_col = f"P{pid}_label"
        
        if score_col in master_df.columns:
            w = weights.get(pid, 0)
            scores = master_df[score_col].fillna(0)
            weighted_sum += scores * w
            
            total_weight += w * master_df[score_col].notna().astype(float)
            
            total_votes += master_df[label_col].fillna(0)

    master_df['PCS'] = weighted_sum / (total_weight + 1e-8)
    master_df['Vote_Count'] = total_votes

    master_df = master_df.sort_values(by=['Vote_Count', 'PCS'], ascending=[False, False])
    master_df.to_csv(args.output_file, index=False)
    print(f"Final ranking saved to {args.output_file}")

if __name__ == "__main__":
    main()