#!/bin/bash
# scripts/run_scfoundation_emb.sh

# 获取当前脚本所在目录的上一级目录（即项目根目录）
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 检查用户是否提供了输入文件（singlecell expression profile）
if [ -z "$1" ]; then
    echo "错误: 请提供单细胞表达矩阵文件 (CSV) 的路径。"
    echo "用法: bash scripts/run_scfoundation_emb.sh <path_to_your_singlecell_expression_profile.csv> [output_dir]"
    exit 1
fi

# INPUT_DIR/CSV_FILE: 用户自己提供的 singlecell expression profile 路径
CSV_FILE="$1"

# 允许用户自定义输出目录（参数2），默认为 data/Rebuttal_UMAP_Result
OUTPUT_DIR="${2:-${PROJECT_ROOT}/data/Rebuttal_UMAP_Result}"

# 定义相对路径
SC_FOUNDATION_DIR="${PROJECT_ROOT}/external/scFoundation/model"

# 根据要求，将 MODEL_CKPT 设置为子模块中模型的默认存放路径
MODEL_CKPT="${PROJECT_ROOT}/external/scFoundation/model/models/models.ckpt"

# 基因索引文件
GENE_INDEX_FILE="${SC_FOUNDATION_DIR}/OS_scRNA_gene_index.19264.tsv"

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 可以根据需要修改细胞类型标识
CELL_TYPE="Custom"

echo "Starting scFoundation embedding generation..."
python "${SC_FOUNDATION_DIR}/get_embedding.py" \
    --task_name "Pseudo_${CELL_TYPE}_gene" \
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