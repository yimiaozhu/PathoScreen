#!/bin/bash
# scripts/run_scfoundation_emb.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "$1" ]; then
    echo "错误: 请提供单细胞表达矩阵文件路径。"
    exit 1
fi

CSV_FILE="$(realpath "$1")"
OUTPUT_DIR="$(mkdir -p "${2:-${PROJECT_ROOT}/data/cell_emb}" && realpath "${2:-${PROJECT_ROOT}/data/cell_emb}")"

SC_FOUNDATION_DIR="${PROJECT_ROOT}/external/scFoundation/model"
MODEL_CKPT="${SC_FOUNDATION_DIR}/models/models.ckpt"

echo "Starting scFoundation embedding generation..."

cd "${SC_FOUNDATION_DIR}" || exit

python "get_embedding.py" \
    --task_name "Custom_gene" \
    --input_type singlecell \
    --output_type gene \
    --pre_normalized F \
    --pool_type all \
    --tgthighres t4 \
    --data_path "$CSV_FILE" \
    --save_path "$OUTPUT_DIR" \
    --ckpt_name 01B-resolution \
    --model_path "$MODEL_CKPT" 

echo "Embeddings saved to $OUTPUT_DIR"