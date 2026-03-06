#!/bin/bash
# scripts/run_scfoundation_emb.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$1" ]; then
    echo "Wrong! please provide correct file path!"
    exit 1
fi

CSV_FILE="$(realpath "$1")"
OUTPUT_DIR="$(mkdir -p "${2:-${PROJECT_ROOT}/data/cell_emb}" && realpath "${2:-${PROJECT_ROOT}/data/cell_emb}")"

SC_FOUNDATION_DIR="${PROJECT_ROOT}/external/scFoundation/model"
MODEL_CKPT="${SC_FOUNDATION_DIR}/models/models.ckpt"
LANDMARK_CSV="${PROJECT_ROOT}/data/cell_emb/landmark_gene_index.csv"

TASK_NAME=$(python -c "
import pandas as pd
try:
    df = pd.read_csv('$CSV_FILE', index_col=0)
    num_cells = len(df)
    if num_cells == 1:
        print(str(df.index[0]).strip())
    else:
        print(f'{num_cells}_cells')
except Exception as e:
    print('Error_parsing_csv')
")

if [ "$TASK_NAME" == "Error_parsing_csv" ]; then
    echo "Error: Unable to parse CSV file. Please ensure it conforms to the corresponding format"
    exit 1
fi

echo "Detected input cells. Setting task_name to: $TASK_NAME"
echo "Starting scFoundation embedding generation..."

cd "${SC_FOUNDATION_DIR}" || exit

python "get_embedding.py" \
    --task_name "${TASK_NAME}" \
    --input_type singlecell \
    --output_type gene \
    --pre_normalized F \
    --pool_type all \
    --tgthighres t4 \
    --data_path "$CSV_FILE" \
    --save_path "$OUTPUT_DIR" \
    --ckpt_name 01B-resolution \
    --model_path "$MODEL_CKPT"

RAW_NPY="${OUTPUT_DIR}/${TASK_NAME}_01B-resolution_singlecell_gene_embedding_t4_resolution.npy"

echo "------------------------------------------------------"
echo "Extracting 978 landmark genes to create PathoScreen independent cell matrices..."
python -c "
import numpy as np
import pandas as pd
import pickle
import os

raw_npy_path = '${RAW_NPY}'
output_dir = '${OUTPUT_DIR}'
landmark_csv = '${LANDMARK_CSV}'
csv_file = '${CSV_FILE}'

raw_emb = np.load(raw_npy_path)
if raw_emb.ndim == 2:
    raw_emb = np.expand_dims(raw_emb, 0)

input_df = pd.read_csv(csv_file, index_col=0)
cell_ids = [str(x).strip() for x in input_df.index]

if len(cell_ids) != raw_emb.shape[0]:
    print(f'Error: Sample mismatch! CSV has {len(cell_ids)}, NPY has {raw_emb.shape[0]}')
    exit(1)

landmark_df = pd.read_csv(landmark_csv)
landmark_indices = landmark_df['index'].astype(int).values

landmark_emb = raw_emb[:, landmark_indices, :]

num_cells = len(cell_ids)
if num_cells == 1:
    cid = cell_ids[0]
    save_name = cid if cid and cid != 'nan' else 'unnamed_cell'
    out_path = os.path.join(output_dir, f'{save_name}_scFoundation_input_gene_emb.npy')
    np.save(out_path, landmark_emb[0])
    print(f'Saved single cell .npy: {out_path}')
else:
    out_path = os.path.join(output_dir, f'{num_cells}_cells_scFoundation_input_gene_emb.pkl')
    cell_emb_dict = {cid: landmark_emb[i] for i, cid in enumerate(cell_ids)}
    with open(out_path, 'wb') as f:
        pickle.dump(cell_emb_dict, f)
    print(f'Saved {num_cells} cells into .pkl: {out_path}')
"

rm "$RAW_NPY"
echo "Cleaned up raw full-gene embeddings to save disk space."
