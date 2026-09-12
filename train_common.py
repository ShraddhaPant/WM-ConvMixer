"""
train_common.py

Shared training/eval logic for H2 and H3, mirroring train_h1.py's exact
recipe (optimizer, scheduler, early stopping, class-weight formula) so
H1/H2/H3 results are directly comparable. Only `use_sca` differs between
the two calling scripts.
"""

import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, recall_score, confusion_matrix)

from models.backbone.modules.dataset import ELPVDataset
from models.wm_convmixer import WMConvMixer

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 40
PATIENCE = 8          # early stopping on val macro-F1, same as H1
BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_class_weights(labels, num_classes=4):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "per_class_recall": recall_score(all_labels, all_preds, average=None).tolist(),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def train_model(model, model_name, train_loader, val_loader, class_weights):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"[{model_name}] epoch {epoch+1:02d}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{model_name}] early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def run_experiment(use_sca: bool, model_name: str, output_path: str, result_key: str):
    set_seed()

    train_ds = ELPVDataset(split="train")
    mean, std = train_ds.computed_stats
    val_ds = ELPVDataset(split="val", normalize_stats=(mean, std))
    test_ds = ELPVDataset(split="test", normalize_stats=(mean, std))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    class_weights = get_class_weights(train_ds.labels)

    model = WMConvMixer(
        in_channels=1,
        dim=128,
        depth=8,
        kernel_size=9,
        wavelet_patch_size=5,
        morphology_length=7,
        num_classes=4,
        use_sca=use_sca,
    )

    print(f"\n===== TRAINING {model_name} (use_sca={use_sca}) =====")
    set_seed()  # same init/shuffle conditions as H1's per-model reseed
    trained = train_model(model, model_name, train_loader, val_loader, class_weights)
    test_metrics = evaluate(trained, test_loader)

    print(f"\n[{model_name}] TEST macro_f1={test_metrics['macro_f1']:.4f}  "
          f"balanced_acc={test_metrics['balanced_accuracy']:.4f}  "
          f"accuracy={test_metrics['accuracy']:.4f}")

    results = {result_key: test_metrics}
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{result_key} RESULTS SAVED")
    print(f"File: {output_path}")
    return results