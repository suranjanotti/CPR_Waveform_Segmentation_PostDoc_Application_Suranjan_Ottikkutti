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
