import numpy as np

def build_lstm_windows(input_data, output_data, window=100, stride=1, normalize=True):
    """
    Build sliding windows for LSTM input and aligned labels (no cross-boundary windows).

    Args:
        input_data (ndarray): shape (num_patches, patch_len), float32 input signals.
        output_data (ndarray): shape (num_patches, patch_len), int labels per timestep.
        window (int): number of timesteps per window.
        stride (int): stride between consecutive windows.
        normalize (bool): if True, z-score normalize each patch independently.

    Returns:
        X (ndarray): shape (num_windows, window, 1), float32 LSTM inputs.
        y (ndarray): shape (num_windows,), int64 labels aligned to last timestep in window.
    """
    Xs, ys = [], []
    for x_row, y_row in zip(input_data, output_data.astype(int)):
        # optional per-patch normalization
        x = x_row.astype(np.float32)
        if normalize:
            x = (x - x.mean()) / (x.std() + 1e-8)

        # build windows inside this patch
        idx = np.arange(0, x.shape[0] - window + 1, stride, dtype=np.int32)
        if idx.size == 0:
            continue

        # stack into (num_windows_in_patch, window, 1)
        X_row = np.stack([x[i:i+window] for i in idx], axis=0)[..., None]

        # label = class at the last timestep of each window
        y_row = y_row[idx + window - 1]

        Xs.append(X_row)
        ys.append(y_row)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)

    print(f"✅ Built LSTM windows: X {X.shape}, y {y.shape}")
    return X, y


def train_cpr_waveform_model(X, y, N_CLASSES, input_size=1, hidden_size=256, num_layers=2, num_epochs=15, batch_size=256, learning_rate=1e-3):
    """
    Train an LSTM classifier for CPR waveform segmentation.
    
    Parameters:
    X: numpy array of input features
    y: numpy array of labels
    N_CLASSES: number of classes
    input_size: input dimension (default 1)
    hidden_size: LSTM hidden size (default 256)
    num_layers: number of LSTM layers (default 2)
    num_epochs: number of training epochs (default 15)
    batch_size: training batch size (default 256)
    learning_rate: learning rate (default 1e-3)
    
    Returns:
    model: trained LSTM model
    results: dictionary with train/val/test metrics
    """
    
    from sklearn.model_selection import StratifiedShuffleSplit
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Stratified 80/10/10 split ---
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_idx, test_idx = next(sss.split(X, y))

    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]

    # Split remaining 20% (val) from train_val (80%)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.1111, random_state=42)
    # 0.1111 of 0.9 ≈ 0.1 → gives 80/10/10 overall
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))

    X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
    X_val,   y_val   = X_train_val[val_idx],   y_train_val[val_idx]

    # --- Convert to torch tensors ---
    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t   = torch.from_numpy(X_val).float()
    y_val_t   = torch.from_numpy(y_val).long()
    X_test_t  = torch.from_numpy(X_test).float()
    y_test_t  = torch.from_numpy(y_test).long()

    # --- Dataset & DataLoader ---
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds   = TensorDataset(X_val_t,   y_val_t)
    test_ds  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout1=0.3, dropout2=0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout1 if num_layers > 1 else 0.0
            )
            self.dropout1 = nn.Dropout(dropout1)
            self.fc1 = nn.Linear(hidden_size, 64)
            self.relu = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout2)
            self.fc_out = nn.Linear(64, num_classes)

        def forward(self, x):
            out_seq, (h_n, c_n) = self.lstm(x)
            h_last = h_n[-1]
            x = self.dropout1(h_last)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout2(x)
            return self.fc_out(x)

    model = LSTMClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=N_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Train / Eval helpers ---
    def train_epoch(model, loader, optimizer, criterion):
        model.train()
        tot_loss = tot_correct = tot_samples = 0
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * xb.size(0)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            tot_samples += xb.size(0)
        return tot_loss / tot_samples, tot_correct / tot_samples

    def eval_epoch(model, loader, criterion):
        model.eval()
        tot_loss = tot_correct = tot_samples = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                tot_loss += loss.item() * xb.size(0)
                tot_correct += (logits.argmax(1) == yb).sum().item()
                tot_samples += xb.size(0)
        return tot_loss / tot_samples, tot_correct / tot_samples

    # --- Train (train/val) ---
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc     = eval_epoch(model, val_loader, criterion)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch:2d} — train_loss {train_loss:.4f}, acc {train_acc:.4f} | val_loss {val_loss:.4f}, acc {val_acc:.4f}")

    # --- Final TEST evaluation (10% held-out) ---
    test_loss, test_acc = eval_epoch(model, test_loader, criterion)
    print(f"TEST — loss {test_loss:.4f}, acc {test_acc:.4f}")
    
    # Return results
    results = {
        'model': model,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }
    
    return results

# Example usage:
# results = train_cpr_waveform_model(X, y, N_CLASSES=3)
# model = results['model']
# test_acc = results['test_accuracy']



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pandas as pd

def split_data_stratified(X, y, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets with stratified sampling
    to ensure even label distribution across all splits.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix of shape (n_samples, n_features)
    y : array-like
        Target labels of shape (n_samples,)
    train_size : float, default=0.7
        Proportion of dataset to include in train split
    val_size : float, default=0.15
        Proportion of dataset to include in validation split
    test_size : float, default=0.15
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test) - split datasets
    """
    
    # First, split into train and temp (combining val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=train_size,
        stratify=y,
        random_state=random_state
    )
    
    # Then split temp into val and test
    val_test_size = val_size / (val_size + test_size)  # Proportion for validation within temp
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_test_size,
        stratify=y_temp,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_label_distribution(y_train, y_val, y_test, y_full=None, print_full=False):
    """
    Print label distribution across splits to verify even distribution.
    
    Parameters:
    -----------
    y_train, y_val, y_test : array-like
        Target labels for each split
    y_full : array-like, optional
        Full target labels for comparison
    print_full : bool, default=False
        Whether to print full label distribution
    """
    
    def get_label_counts(y):
        unique_labels, counts = np.unique(y, return_counts=True)
        return dict(zip(unique_labels, counts))
    
    print("Label Distribution:")
    print("-" * 30)
    
    if print_full and y_full is not None:
        full_counts = get_label_counts(y_full)
        print("Full dataset:")
        for label, count in full_counts.items():
            print(f"  Label {label}: {count}")
        print()
    
    train_counts = get_label_counts(y_train)
    val_counts = get_label_counts(y_val)
    test_counts = get_label_counts(y_test)
    
    print("Train set:")
    for label, count in train_counts.items():
        print(f"  Label {label}: {count}")
    print()
    
    print("Validation set:")
    for label, count in val_counts.items():
        print(f"  Label {label}: {count}")
    print()
    
    print("Test set:")
    for label, count in test_counts.items():
        print(f"  Label {label}: {count}")
    print()

# Example usage:
# X_train, X_val, X_test, y_train, y_val, y_test = split_data_stratified(X, y, train_size=0.7, val_size=0.15, test_size=0.15)
# print_label_distribution(y_train, y_val, y_test, y)

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        last_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(last_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)            # (B, T, H)
        last = out[:, -1, :]             # take last timestep
        logits = self.head(last)         # (B, C)
        return logits

def train_lstm_classifier(
    X_train: Union[np.ndarray, torch.Tensor],  # shape (N, T, F)
    y_train: Union[np.ndarray, torch.Tensor],  # shape (N,) ints
    X_val:   Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_val:   Optional[Union[np.ndarray, torch.Tensor]] = None,
    *,
    hidden_size: int = 128,
    num_layers: int = 1,
    dropout: float = 0.0,
    bidirectional: bool = False,
    batch_size: int = 256,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 5,
    clip_norm: Optional[float] = None,
    num_workers: int = 0,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Builds & trains an LSTM classifier with tqdm batch-by-batch bars and early stopping.

    Early stopping criterion: best validation loss (if val provided), else best train loss.
    Patience counts epochs without improvement.

    Returns:
        model: trained LSTMClassifier
        history: dict with keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc'
    """
    # ---------- device ----------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- tensors ----------
    def as_tensor(x, dtype=None):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(dtype) if dtype is not None else torch.from_numpy(x)
        return x

    X_train = as_tensor(X_train, torch.float32)
    y_train = as_tensor(y_train, torch.long)
    if X_val is not None and y_val is not None:
        X_val = as_tensor(X_val, torch.float32)
        y_val = as_tensor(y_val, torch.long)

    # ---------- dataset & loaders ----------
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device=="cuda"))
    val_loader = None
    if X_val is not None and y_val is not None:
        val_ds = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=(device=="cuda"))

    # ---------- model ----------
    input_dim = X_train.shape[-1]
    num_classes = int(torch.unique(y_train).numel())  # adapt to your labels
    model = LSTMClassifier(
        input_dim=input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    # ---------- loss/opt ----------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---------- history / early stopping ----------
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_metric = float("inf")
    best_state = None
    wait = 0

    # ---------- training loop ----------
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()

            # batch metrics
            running_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        # ----- validation -----
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]", leave=False)
                for xb, yb in pbar_val:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    v_loss += loss.item() * xb.size(0)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == yb).sum().item()
                    v_total += xb.size(0)
                    pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})
            val_loss = v_loss / max(1, v_total)
            val_acc = v_correct / max(1, v_total)

        # ----- logging (TensorFlow-ish one-liner) -----
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        if val_loader is not None:
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if verbose:
                print(f"Epoch {epoch:03d}/{epochs} - "
                      f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            metric_now = val_loss
        else:
            if verbose:
                print(f"Epoch {epoch:03d}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f}")
            metric_now = train_loss

        # ----- early stopping on (val_)loss -----
        if metric_now < best_metric - 1e-6:
            best_metric = metric_now
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best {( 'val_' if val_loader else '' )}loss={best_metric:.4f}).")
                break

    # load best weights if we have them
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
