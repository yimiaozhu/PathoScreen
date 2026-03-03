import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold

from src.config import get_config, PATHWAY_NAMES
from src.paths import PathoScreenPaths
from src.models import PathoScreen
from src.data.dataset import PathoScreenDataset
from src.engine.trainer import Trainer, Evaluator


def load_train_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="PathoScreen Training Pipeline (Scientific Rigor)")
    parser.add_argument("--pathway_id", type=int, required=True, help="Pathway ID (0-6) to define Model Architecture")
    parser.add_argument("--config", type=str, default="configs/train_template.yaml", help="Path to training hyperparameters")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_root", type=str, default="output", help="Root dir to save all artifacts (output/P{pathway_id}/...)")
    args = parser.parse_args()

    # 1. Initialization
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    arch_config = get_config(args.pathway_id)
    
    train_cfg = load_train_config(args.config)
    data_cfg = train_cfg['data']
    hp = train_cfg['train']
    
    seed = int(hp.get("seed", 666))
    hp["seed"] = seed
    hp["epochs"] = int(hp.get("epochs", 300))

    _seed_everything(seed)

    pathway_name = PATHWAY_NAMES.get(args.pathway_id, f"unknown_{args.pathway_id}")
    print(f"[PathoScreen] pathway_id={args.pathway_id} ({pathway_name})")
    print(f"Device: {device}")
    print(f"Epochs (max): {hp['epochs']}")

    # 2. Data preparation
    full_dataset = PathoScreenDataset(
        csv_path=data_cfg["train_path"],
        cell_matrix_path=data_cfg["cell_matrix_path"],
        mode="train",
    )
    labels = np.asarray(full_dataset.get_labels(), dtype=int)
    
    test_dataset = PathoScreenDataset(
        csv_path=data_cfg["test_path"],
        cell_matrix_path=data_cfg["cell_matrix_path"],
        mode="eval",
    )

    # 3. 5-fold cv
    print("\n" + "="*100)
    print("[Phase 1] 5-Fold CV to determine Optimal Epoch (Peak of Mean PRC)")
    print("="*100)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    cv_auprc_history = np.zeros((n_folds, hp["epochs"]), dtype=float)
    cv_thresh_history = np.zeros((n_folds, hp["epochs"]), dtype=float)

    cv_metric_history = {
        "AUROC": np.zeros((n_folds, hp["epochs"]), dtype=float),
        "F1": np.zeros((n_folds, hp["epochs"]), dtype=float),
        "MCC": np.zeros((n_folds, hp["epochs"]), dtype=float),
        "BALANCED_ACC": np.zeros((n_folds, hp["epochs"]), dtype=float),
    }

    fold_splits = list(skf.split(np.zeros(len(labels)), labels))

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"\nFold {fold_idx + 1}/{n_folds}")

        train_sub = torch.utils.data.Subset(full_dataset, train_idx)
        val_sub = torch.utils.data.Subset(full_dataset, val_idx)

        model = PathoScreen(arch_config, device).to(device)
        trainer = Trainer(model, hp, device)
        evaluator = Evaluator(model, device)
        
        for epoch in range(hp["epochs"]):
            loss = trainer.train_epoch(train_sub)
            val_metrics = evaluator.evaluate(val_sub)  # threshold selected by F1-max on PR curve

            cv_auprc_history[fold_idx, epoch] = val_metrics["AUPRC"]
            cv_thresh_history[fold_idx, epoch] = val_metrics["threshold"]
            for k in cv_metric_history.keys():
                cv_metric_history[k][fold_idx, epoch] = float(val_metrics[k])

            if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
                print(f"  Ep {epoch + 1:>3d}: loss={loss:.4f} | val_AUPRC={val_metrics['AUPRC']:.4f}")

    # 4. Determination of epoch and threshold
    mean_prc_curve = cv_auprc_history.mean(axis=0)
    best_epoch_idx = int(np.argmax(mean_prc_curve))
    best_epoch = best_epoch_idx + 1

    fold_thresholds_at_best = cv_thresh_history[:, best_epoch_idx]
    final_threshold = float(fold_thresholds_at_best.mean())

    print("\n" + "=" * 100)
    print("[Phase 1b] 5-fold CV reported metrics (each fold at its own AUPRC-maximizing epoch)")
    print("=" * 100)

    fold_best_epoch_idx = cv_auprc_history.argmax(axis=1)  
    fold_best_epoch = (fold_best_epoch_idx + 1).tolist()

    fold_metrics = []
    for fold_idx in range(n_folds):
        eidx = int(fold_best_epoch_idx[fold_idx])
        m = {
            "best_epoch": int(eidx + 1),
            "AUPRC": float(cv_auprc_history[fold_idx, eidx]),
            "AUROC": float(cv_metric_history["AUROC"][fold_idx, eidx]),
            "F1": float(cv_metric_history["F1"][fold_idx, eidx]),
            "MCC": float(cv_metric_history["MCC"][fold_idx, eidx]),
            "BALANCED_ACC": float(cv_metric_history["BALANCED_ACC"][fold_idx, eidx]),
            "threshold": float(cv_thresh_history[fold_idx, eidx]),
        }
        fold_metrics.append(m)
        print(
            f"Fold {fold_idx + 1}: "
            f"best_epoch={m['best_epoch']} | "
            f"AUPRC={m['AUPRC']:.4f} | AUROC={m['AUROC']:.4f} | F1={m['F1']:.4f} | "
            f"MCC={m['MCC']:.4f} | BACC={m['BALANCED_ACC']:.4f} | thr={m['threshold']:.4f}"
        )

    cv_report_summary = {}
    for key in ["AUPRC", "AUROC", "F1", "MCC", "BALANCED_ACC"]:
        vals = np.array([fm[key] for fm in fold_metrics], dtype=float)
        cv_report_summary[key] = {
            "per_fold": vals.tolist(),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        }

    print("\n[5-fold CV Summary: AVG ± STD]")
    for key in ["AUPRC", "AUROC", "F1", "MCC", "BALANCED_ACC"]:
        mu = cv_report_summary[key]["mean"]
        sd = cv_report_summary[key]["std"]
        print(f"  {key}: {mu:.4f} ± {sd:.4f}")

    # 5. Final retraining (Phase 2)
    print("\n" + "=" * 80)
    print(f"[Phase 2] Retrain on full training set for {best_epoch} epochs")
    print("=" * 80)

    final_model = PathoScreen(arch_config, device).to(device)
    final_trainer = Trainer(final_model, hp, device)

    for epoch in range(best_epoch):
        loss = final_trainer.train_epoch(full_dataset)
        if (epoch + 1) % 10 == 0 or (epoch + 1) == best_epoch:
            print(f"  Retrain Ep {epoch + 1}/{best_epoch}: loss={loss:.4f}")
            
    # 6. Save model
    paths = PathoScreenPaths(Path(args.output_root), args.pathway_id)
    paths.ensure()

    torch.save(final_model.state_dict(), paths.checkpoint_last())
    torch.save(final_model.state_dict(), paths.checkpoint_best())
    print(f"Saved checkpoint (last): {paths.checkpoint_last()}")
    print(f"Saved checkpoint (best): {paths.checkpoint_best()}")
    
    # 7. Independent Testing
    print("\n" + "=" * 80)
    print("[Phase 3] Independent test evaluation (fixed threshold from CV)")
    print("=" * 80)

    final_evaluator = Evaluator(final_model, device)
    test_metrics = final_evaluator.evaluate(test_dataset, fixed_threshold=final_threshold)

    print(
        f"Test: AUPRC={test_metrics['AUPRC']:.4f} | AUROC={test_metrics['AUROC']:.4f} | "
        f"F1={test_metrics['F1']:.4f} | MCC={test_metrics['MCC']:.4f} | BACC={test_metrics['BALANCED_ACC']:.4f} | "
        f"thr={test_metrics['threshold']:.4f}"
    )

    out_dir = paths.p_dir
    _ensure_dir(paths.metrics_dir)
    train_summary = {
        "pathway_id": args.pathway_id,
        "pathway_name": pathway_name,
        "best_epoch": best_epoch,
        "final_threshold": final_threshold,
        "cv_selection": {
            "epoch_metric": "mean_AUPRC",
            "threshold_metric": "F1_max_on_PR",
        },
        "seed": seed,
    }

    metrics_json = {
        "cv_5fold": {
            "n_folds": n_folds,
            "reported": {
                "per_fold_best_epoch": fold_best_epoch,
                "per_fold": fold_metrics,
                "summary": cv_report_summary,
            },
            "selection": {
                "best_epoch": best_epoch,
                "best_epoch_mean_AUPRC": float(mean_prc_curve[best_epoch_idx]),
                "final_threshold": final_threshold,
                "fold_thresholds_at_best_epoch": fold_thresholds_at_best.tolist(),
            },
        },
        "independent_test": test_metrics,
    }

    (paths.metrics_dir / "train_summary.json").write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    (paths.metrics_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    print(f"\nSaved: {paths.metrics_dir / 'train_summary.json'}")
    print(f"Saved: {paths.metrics_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()