import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    balanced_accuracy_score,
)

from src.models import FocalLoss
from src.data.dataset import GraphCollator


def _to_float(x, name: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Hyperparameter `{name}` must be a number, got {x!r} ({type(x).__name__}).") from e


def _to_int(x, name: str) -> int:
    try:
        return int(x)
    except Exception as e:
        raise TypeError(f"Hyperparameter `{name}` must be an int, got {x!r} ({type(x).__name__}).") from e


class Trainer:
    def __init__(self, model, hp: dict, device: torch.device):
        self.model = model
        self.device = device

        self.batch_size = _to_int(hp.get("batch_size", 32), "batch_size")
        lr = _to_float(hp.get("lr", 1e-4), "lr")
        weight_decay = _to_float(hp.get("weight_decay", 0.0), "weight_decay")

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        focal_alpha = _to_float(hp.get("focal_alpha", 0.25), "focal_alpha")
        focal_gamma = _to_float(hp.get("focal_gamma", 2.0), "focal_gamma")
        self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        self.collator = GraphCollator(device)

    @staticmethod
    def _resolve_labels(dataset) -> np.ndarray:
        if hasattr(dataset, "get_labels"):
            labels = np.asarray(dataset.get_labels(), dtype=np.int64)
            return labels

        if isinstance(dataset, Subset):
            base = dataset.dataset
            if not hasattr(base, "get_labels"):
                raise AttributeError(
                    "Subset base dataset does not implement get_labels(). "
                    "Implement get_labels() in PathoScreenDataset."
                )
            base_labels = np.asarray(base.get_labels(), dtype=np.int64)
            idx = np.asarray(dataset.indices, dtype=np.int64)
            return base_labels[idx]

        raise AttributeError(
            "Dataset must implement get_labels() or be a torch.utils.data.Subset wrapping a dataset that implements get_labels()."
        )

    def _get_balanced_sampler(self, dataset):
        labels = self._resolve_labels(dataset)

        counts = np.bincount(labels, minlength=2).astype(np.float64)
        counts[counts == 0] = 1.0  
        class_weights = 1.0 / counts
        sample_weights = class_weights[labels]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    def train_epoch(self, dataset) -> float:
        self.model.train()

        sampler = self._get_balanced_sampler(dataset)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            collate_fn=self.collator,
        )

        total_loss = 0.0
        count = 0

        for mol, adj, cell, label, atom_nums in loader:
            logits, _ = self.model(mol, adj, cell, atom_nums)
            loss = self.criterion(logits, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            bs = label.size(0)
            total_loss += float(loss.item()) * bs
            count += bs

        return total_loss / max(count, 1)


class Evaluator:
    def __init__(self, model, device: torch.device, batch_size: int = 32):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.collator = GraphCollator(device)

    def evaluate(self, dataset, fixed_threshold: float | None = None) -> dict:
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collator,
        )

        y_true = []
        y_scores = []

        with torch.no_grad():
            for mol, adj, cell, label, atom_nums in loader:
                logits, _ = self.model(mol, adj, cell, atom_nums)
                probs = torch.softmax(logits, dim=1)[:, 1]
                y_true.append(label.detach().cpu().numpy())
                y_scores.append(probs.detach().cpu().numpy())

        y_true = np.concatenate(y_true).astype(int)
        y_scores = np.concatenate(y_scores).astype(float)

        if len(np.unique(y_true)) < 2:
            auprc = 0.0
            prec = np.array([1.0])
            rec = np.array([0.0])
            thresholds = np.array([])
        else:
            prec, rec, thresholds = precision_recall_curve(y_true, y_scores)
            auprc = float(auc(rec, prec))

        try:
            auroc = float(roc_auc_score(y_true, y_scores)) if len(np.unique(y_true)) >= 2 else 0.0
        except Exception:
            auroc = 0.0

        if fixed_threshold is not None:
            thresh = float(fixed_threshold)
        else:
            f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
            best_idx = int(np.argmax(f1s))
            if thresholds.size == 0:
                thresh = 0.5
            else:
                thresh = float(thresholds[min(best_idx, thresholds.size - 1)])

        y_pred = (y_scores >= thresh).astype(int)

        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 or len(np.unique(y_true)) > 1 else 0.0
        bacc = float(balanced_accuracy_score(y_true, y_pred))

        return {
            "AUPRC": auprc,
            "AUROC": auroc,
            "F1": f1,
            "MCC": mcc,
            "BALANCED_ACC": bacc,
            "threshold": thresh,
        }
