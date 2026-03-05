import json
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from src.config import get_config, PATHWAY_NAMES
from src.paths import PathoScreenPaths
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset
from src.engine.trainer import Trainer, Evaluator

def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def run_training(args):
    # 1. Config & Init
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    with open(args.config, "r", encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    
    # Resolve paths
    data_cfg = full_cfg['data']
    train_path = data_cfg['train_path'].format(pathway_id=args.pathway_id)
    test_path = data_cfg['test_path'].format(pathway_id=args.pathway_id)
    
    hp = full_cfg['train']
    seed = int(hp.get("seed", 666))
    hp["seed"] = seed
    _seed_everything(seed)
    
    arch_config = get_config(args.pathway_id)
    pathway_name = PATHWAY_NAMES.get(args.pathway_id, f"P{args.pathway_id}")
    
    print(f"[PathoScreen] Training P{args.pathway_id}: {pathway_name}")
    print(f"  Train Data: {train_path}")
    print(f"  Device: {device}")

    # 2. Data Loading
    full_dataset = PathoScreenDataset(
        csv_path=train_path,
        cell_matrix_path=data_cfg["cell_matrix_path"],
        mode="train",
        smiles_cache_path=args.smiles_cache
    )
    labels = np.asarray(full_dataset.get_labels(), dtype=int)
    
    # 3. Phase 1: 5-Fold CV
    print("\n" + "="*60)
    print("[Phase 1] 5-Fold CV (Hyperparam Search)")
    print("="*60)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    epochs = hp["epochs"]
    cv_auprc = np.zeros((n_folds, epochs))
    cv_thresh = np.zeros((n_folds, epochs))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"Fold {fold_idx+1}/{n_folds}...")
        train_sub = torch.utils.data.Subset(full_dataset, train_idx)
        val_sub = torch.utils.data.Subset(full_dataset, val_idx)
        
        model = PathoScreen(arch_config, device).to(device)
        trainer = Trainer(model, hp, device)
        evaluator = Evaluator(model, device)
        
        for epoch in range(epochs):
            trainer.train_epoch(train_sub)
            metrics = evaluator.evaluate(val_sub)
            
            cv_auprc[fold_idx, epoch] = metrics["AUPRC"]
            cv_thresh[fold_idx, epoch] = metrics["threshold"]
            
    # Select Best Epoch
    mean_auprc = cv_auprc.mean(axis=0)
    best_epoch_idx = np.argmax(mean_auprc)
    best_epoch = best_epoch_idx + 1
    
    # Select Threshold (Average of optimal thresholds at best epoch)
    final_threshold = cv_thresh[:, best_epoch_idx].mean()
    
    print(f"\n✅ Optimal Epoch: {best_epoch} (Mean AUPRC: {mean_auprc[best_epoch_idx]:.4f})")
    print(f"✅ Optimal Threshold: {final_threshold:.4f}")
    
    # 4. Phase 2: Retraining
    print("\n" + "="*60)
    print(f"[Phase 2] Retraining on Full Dataset ({best_epoch} epochs)")
    print("="*60)
    
    final_model = PathoScreen(arch_config, device).to(device)
    final_trainer = Trainer(final_model, hp, device)
    
    for epoch in range(best_epoch):
        loss = final_trainer.train_epoch(full_dataset)
        if (epoch+1) % 10 == 0 or (epoch+1) == best_epoch:
            print(f"  Ep {epoch+1}/{best_epoch}: loss={loss:.4f}")
            
    # 5. Save Artifacts
    paths = PathoScreenPaths(Path(args.output_root), args.pathway_id)
    paths.ensure()
    
    torch.save(final_model.state_dict(), paths.checkpoint_best())
    print(f"Saved model: {paths.checkpoint_best()}")
    
    # 6. Independent Test
    print("\n" + "="*60)
    print("[Phase 3] Independent Test Evaluation")
    print("="*60)
    
    try:
        test_dataset = PathoScreenDataset(
            csv_path=test_path,
            cell_matrix_path=data_cfg["cell_matrix_path"],
            mode="eval",
            smiles_cache_path=args.smiles_cache
        )
        final_evaluator = Evaluator(final_model, device)
        test_metrics = final_evaluator.evaluate(test_dataset, fixed_threshold=final_threshold)
        
        print(f"Test Metrics (Thr={final_threshold:.4f}):")
        print(f"  AUPRC: {test_metrics['AUPRC']:.4f}")
        print(f"  AUROC: {test_metrics['AUROC']:.4f}")
        print(f"  F1:    {test_metrics['F1']:.4f}")
        
        # Save Metrics
        summary = {
            "pathway_id": args.pathway_id,
            "best_epoch": int(best_epoch),
            "threshold": float(final_threshold),
            "cv_mean_auprc": float(mean_auprc[best_epoch_idx]),
            "test_metrics": test_metrics
        }
        with open(paths.metrics_dir / "train_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            
    except FileNotFoundError:
        print(f"⚠️ Test file not found: {test_path}. Skipping evaluation.")