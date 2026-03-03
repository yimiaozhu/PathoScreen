import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PathwayConfig:
    # Model architecture
    atom_dim: int = 34
    gene_input_dim: int = 512
    num_latents: int = 32

    hid_dim: int = 256
    n_layers: int = 3
    n_heads: int = 8
    pf_dim: int = 1024
    mlp_hid_dims: List[int] = field(default_factory=lambda: [512, 512, 128])

    dropout: float = 0.2
    gene_dropout: float = 0.5
    mlp_dropout: float = 0.2


PATHWAY_NAMES = {
    0: "fatty_acid_metabolism",
    1: "liver_insulin_signaling",
    2: "antioxidant_defense",
    3: "anti-apoptosis",         
    4: "mitochondrial_function",
    5: "anti-inflammation",      
    6: "TGF_beta_signaling"
}


PATHWAY_CONFIGS: Dict[int, PathwayConfig] = {
    0: PathwayConfig(hid_dim=256, n_layers=3, n_heads=8, pf_dim=1024, mlp_hid_dims=[512, 512, 128], dropout=0.2, gene_dropout=0.5, mlp_dropout=0.2),
    1: PathwayConfig(hid_dim=256, n_layers=4, n_heads=4, pf_dim=1024, mlp_hid_dims=[256, 128, 128], dropout=0.1, gene_dropout=0.4, mlp_dropout=0.4),
    2: PathwayConfig(hid_dim=256, n_layers=3, n_heads=8, pf_dim=1024, mlp_hid_dims=[512, 512, 256], dropout=0.1, gene_dropout=0.5, mlp_dropout=0.4),
    3: PathwayConfig(hid_dim=128, n_layers=3, n_heads=8, pf_dim=512,  mlp_hid_dims=[256, 256, 128], dropout=0.1, gene_dropout=0.5, mlp_dropout=0.2),
    4: PathwayConfig(hid_dim=128, n_layers=3, n_heads=4, pf_dim=512,  mlp_hid_dims=[256, 256, 128], dropout=0.1, gene_dropout=0.5, mlp_dropout=0.2),
    5: PathwayConfig(hid_dim=256, n_layers=3, n_heads=8, pf_dim=1024, mlp_hid_dims=[512, 512, 256], dropout=0.1, gene_dropout=0.5, mlp_dropout=0.2),
    6: PathwayConfig(hid_dim=256, n_layers=3, n_heads=8, pf_dim=1024, mlp_hid_dims=[512, 512, 256], dropout=0.1, gene_dropout=0.5, mlp_dropout=0.2),
}


def get_config(pathway_id: int) -> PathwayConfig:
    if pathway_id not in PATHWAY_CONFIGS:
        raise ValueError(f"Unknown pathway_id: {pathway_id}")
    return PATHWAY_CONFIGS[pathway_id]


def get_checkpoint_path(pathway_id: int) -> str:
    return f"checkpoints/P{pathway_id}.pt"

