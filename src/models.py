import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() > 1:
            inputs = inputs[:, 1]
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        if self.reduction == 'mean': return loss.mean()
        elif self.reduction == 'sum': return loss.sum()
        else: return loss


class SelfAttention(nn.Module):
    """Used for Self-Attention and Cross-Attention."""
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0, f"Hidden dim {hid_dim} must be divisible by heads {n_heads}"
        self.head_dim = hid_dim // n_heads
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention_weights = self.do(F.softmax(energy, dim=-1))
        
        x = torch.matmul(attention_weights, V)
        
        x = x.permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.hid_dim)
        x = self.fc(x)
        
        return x, attention_weights


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x.permute(0, 2, 1)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hid_dims, output_dim=2, dropout=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hid_dim in hid_dims:
            layers.append(nn.Linear(prev_dim, hid_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hid_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GeneEncoder(nn.Module):
    def __init__(self, gene_dim, hid_dim, num_latents, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim

        self.feature_transform = nn.Sequential(
            nn.Linear(gene_dim, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.latents = nn.Parameter(torch.randn(1, num_latents, hid_dim))

        self.aggregator = SelfAttention(hid_dim, n_heads=8, dropout=dropout, device=device)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, gene_expression):
        batch_size = gene_expression.size(0)

        gene_feats = self.feature_transform(gene_expression)

        latents = self.latents.repeat(batch_size, 1, 1)
        
        aggregated_latents, attn_weights = self.aggregator(latents, gene_feats, gene_feats)

        latents_out = self.ln(latents + aggregated_latents)
        return latents_out, attn_weights
    

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout, device)
        self.ea = SelfAttention(hid_dim, n_heads, dropout, device)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, mol, cell, mol_mask=None, cell_mask=None):
        _mol, _ = self.sa(mol, mol, mol, mol_mask)
        mol = self.ln(mol + self.do(_mol))

        _mol, attention = self.ea(mol, cell, cell, cell_mask)
        mol = self.ln(mol + self.do(_mol))
        
        _mol = self.pf(mol)
        mol = self.ln(mol + self.do(_mol))
        return mol, attention


class Decoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.hid_dim = config.hid_dim
        self.ln = nn.LayerNorm(config.hid_dim)

        self.ft = nn.Linear(config.atom_dim, config.hid_dim)
        self.do = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(
                config.hid_dim, config.n_heads, config.pf_dim, 
                config.dropout, device
            )
            for _ in range(config.n_layers)
        ])

        self.classifier = MLPClassifier(
            input_dim=config.hid_dim, 
            hid_dims=config.mlp_hid_dims, 
            output_dim=2, 
            dropout=config.mlp_dropout
        )

    def forward(self, mol, cell, mol_mask=None, cell_mask=None):
        mol = self.ft(mol)
        
        final_attn_weights = None
        
        for layer in self.layers:
            mol, final_attn_weights = layer(mol, cell, mol_mask, cell_mask)

        norm = torch.norm(mol, dim=2)
        if mol_mask is not None:
            mask_sq = mol_mask.squeeze(1).squeeze(1)
            norm = norm.masked_fill(mask_sq == 0, -1e9)

        alpha = F.softmax(norm, dim=1).unsqueeze(-1)
        mol_repr = (mol * alpha).sum(dim=1)

        logits = self.classifier(mol_repr)
        return logits, final_attn_weights


class PathoScreen(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.weight = nn.Parameter(torch.FloatTensor(config.atom_dim, config.atom_dim))
        self.init_weight()

        self.gene_encoder = GeneEncoder(
            gene_dim=config.gene_input_dim,
            hid_dim=config.hid_dim,
            num_latents=config.num_latents,
            dropout=config.gene_dropout,
            device=device
        )

        self.decoder = Decoder(config, device)

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def gcn(self, x, adj):
        support = torch.matmul(x, self.weight)
        output = torch.bmm(adj, support)
        return output

    def make_masks(self, atom_num, compound_max_len):
        N = len(atom_num)
        
        mol_mask = torch.zeros((N, compound_max_len), device=self.device)
        for i, num in enumerate(atom_num):
            mol_mask[i, :num] = 1
        mol_mask = mol_mask.unsqueeze(1).unsqueeze(2)
        
        cell_mask = torch.ones((N, self.config.num_latents), device=self.device)
        cell_mask = cell_mask.unsqueeze(1).unsqueeze(2)

        return mol_mask, cell_mask

    def forward(self, compound, adj, gene_expr, atom_num):
        mol_feat = self.gcn(compound, adj)
        
        cell_latents, _ = self.gene_encoder(gene_expr) 

        mol_mask, cell_mask = self.make_masks(atom_num, compound.shape[1])
        logits, attn_weights = self.decoder(
            mol=mol_feat, 
            cell=cell_latents, 
            mol_mask=mol_mask, 
            cell_mask=cell_mask
        )
        
        return logits, attn_weights