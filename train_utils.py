import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, recall_score, confusion_matrix)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)          # <-- add: removes thread-order nondeterminism on CPU
    torch.use_deterministic_algorithms(True) 


def get_class_weights(labels, num_classes=4):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
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


def train_model(model, model_name, train_loader, val_loader, class_weights, device,
                 epochs=40, patience=8, lr=3e-4, weight_decay=1e-4):
    """
    Shared training procedure — used identically by H1 through H5 so only
    the MODEL changes between experiments, never the training conditions
    (review Section IX-D).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1, best_state, patience_counter = -1, None, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        avg_train_loss = total_loss / len(train_loader.dataset)
        print(f"[{model_name}] epoch {epoch+1:02d}/{epochs}  "
              f"train_loss={avg_train_loss:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_name}] early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model