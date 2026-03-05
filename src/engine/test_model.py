import torch
import pytest
from src.config import get_config
from src.models import PathoScreen

@pytest.mark.parametrize("pid", [0, 3, 6])
def test_model_forward_shape(pid):
    """Test if model forward pass produces correct output shape (B, 2)"""
    config = get_config(pid)
    device = torch.device("cpu")
    model = PathoScreen(config, device)
    
    # Mock Inputs
    batch_size = 4
    max_atoms = 20
    
    # (B, max_atoms, atom_dim)
    mol = torch.randn(batch_size, max_atoms, config.atom_dim)
    # (B, max_atoms, max_atoms)
    adj = torch.randn(batch_size, max_atoms, max_atoms)
    # (B, 978, 512) - assuming scFoundation output shape
    cell = torch.randn(batch_size, 978, config.gene_input_dim)
    # Atom counts
    atom_num = [10, 15, 20, 5]
    
    logits, attn = model(mol, adj, cell, atom_num)
    
    assert logits.shape == (batch_size, 2)
    assert not torch.isnan(logits).any()

def test_config_integrity():
    """Ensure all 7 pathways have configs"""
    for i in range(7):
        cfg = get_config(i)
        assert cfg.hid_dim > 0
        
    with pytest.raises(ValueError):
        get_config(999)