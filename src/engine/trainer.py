import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_curve
from tqdm import tqdm

from src.data.dataset import GraphCollator
from src.models import FocalLoss

class Trainer:
    def __init__(self, model, hp, device):
        self.model = model
        self.device = device
        self.batch_size = hp.get("batch_size", 32)
        self.lr = float(hp.get("lr", 1e-4))
        self.weight_decay = float(hp.get("weight_decay", 1e-4))
        self.num_workers = hp.get("num_workers", 4)
        self.pin_memory = hp.get("pin_memory", True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        
        # Loss
        self.criterion = FocalLoss(
            alpha=hp.get("focal_alpha", 0.25), 
            gamma=hp.get("focal_gamma", 2.0)
        )

    def train_epoch(self, dataset):
        self.model.train()
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=GraphCollator(self.device),
            pin_memory=self.pin_memory
        )
        
        epoch_loss = 0
        for mol, adj, cell, label, atom_lens in loader:
            self.optimizer.zero_grad()
            
            logits, _ = self.model(mol, adj, cell, atom_lens)
            loss = self.criterion(logits, label)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
        return epoch_loss / len(loader)

class Evaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataset, fixed_threshold=None):
        self.model.eval()
        loader = DataLoader(
            dataset, 
            batch_size=32, 
            shuffle=False, 
            num_workers=2,
            collate_fn=GraphCollator(self.device)
        )
        
        y_true = []
        y_scores = []
        
        for mol, adj, cell, label, atom_lens in loader:
            logits, _ = self.model(mol, adj, cell, atom_lens)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            y_true.extend(label.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        # Metrics
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        
        # Determine threshold (maximize F1) if not provided
        if fixed_threshold is None:
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1s)
            threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        else:
            threshold = fixed_threshold
            
        y_pred = (y_scores >= threshold).astype(int)
        
        return {
            "AUROC": auroc,
            "AUPRC": auprc,
            "F1": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
            "BALANCED_ACC": balanced_accuracy_score(y_true, y_pred),
            "threshold": threshold
        }