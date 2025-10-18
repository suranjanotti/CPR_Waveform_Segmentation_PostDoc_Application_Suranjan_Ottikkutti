# my_plotting_fun.py
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_arterial_waveform(input_data, output_data, signal_freq=200, 
                           class_labels=None, class_colors=None):
    """
    Plot arterial waveform with shaded segments based on output class.
    
    Parameters:
    -----------
    input_data : array-like
        Arterial waveform data (1D array)
    output_data : array-like
        Output class labels (1D array)
    signal_freq : int, default=200
        Sampling rate in Hz
    class_labels : dict, optional
        Dictionary mapping class IDs to labels (e.g., {3: "Diastolic", 4: "Systolic"})
    class_colors : dict, optional
        Dictionary mapping class IDs to colors (e.g., {3: "#8EC9F0", 4: "#F29A9A"})
    
    Returns:
    --------
    tuple
        (figure, axis) of the created plot
    """
    
    # Set default class labels and colors if not provided
    if class_labels is None:
        class_labels = {3: "Diastolic", 4: "Systolic"}
    
    if class_colors is None:
        class_colors = {3: "#8EC9F0", 4: "#F29A9A"}
    
    # Create time array
    time = np.arange(len(input_data)) / signal_freq
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot arterial waveform
    sns.lineplot(
        x=time,
        y=input_data,
        ax=ax1,
        label='Arterial Waveform',
        color='tab:blue'
    )
    
    # Secondary y-axis for output class
    ax2 = ax1.twinx()
    sns.lineplot(
        x=time,
        y=output_data,
        ax=ax2,
        linestyle='--',
        color='tab:orange',
    )

    # Shading function
    def shade_segments(ax, t, y_classes):
        # Get unique classes present in the data
        unique_classes = np.unique(y_classes)
        
        # Only shade if we have classes to shade
        for cid in unique_classes:
            # Skip if class is not in our color mapping
            if cid not in class_colors:
                continue
                
            m = (y_classes == cid)
            if not m.any():
                continue
                
            edges = np.diff(np.r_[False, m, False]).nonzero()[0]
            starts, ends = edges[0::2], edges[1::2]
            for s, e in zip(starts, ends):
                ax.axvspan(t[s], t[e-1], color=class_colors[cid], alpha=0.25, zorder=0)

    # Call shading after plotting
    shade_segments(ax1, time, output_data)
    
    # Custom legend (no duplicates, placed outside)
    # Determine which classes are present in the output data
    unique_classes = np.unique(output_data)
    legend_handles = [
        Line2D([], [], color='tab:blue', label='Arterial Waveform'),
        Line2D([], [], color='tab:orange', linestyle='--', label='Output Class'),
    ]
    for cid in unique_classes:
        if cid in class_labels and cid in class_colors:
            legend_handles.append(Patch(facecolor=class_colors[cid], alpha=0.25, label=class_labels[cid]))
    
    ax1.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.15, 1.0),  # move further right
        borderaxespad=0.,
        frameon=True
    )
    
    # Titles and labels
    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Arterial Waveform")
    ax2.set_ylabel("Output Class")
    ax1.set_title("Example Slice of Arterial Waveform Over Time During Normal Heartbeat")
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()
    
    return fig, ax1

# Example usage:
# plot_arterial_waveform(input_data, output_data, signal_freq=200)
# plot_arterial_waveform(input_data, output_data, signal_freq=200, 
#                       class_labels={3: "Diastolic", 4: "Systolic"},
#                       class_colors={3: "#8EC9F0", 4: "#F29A9A"})


import numpy as np
import pandas as pd
from typing import Tuple, List

def stroke_distributions(
    label_matrix: np.ndarray,
    label_a: int = 1,
    label_b: int = 2,
    zero_label: int = 0,
    allow_other_labels: bool = False,
    signal_freq: int = 200
) -> Tuple[List[float], List[float], pd.DataFrame]:
    """
    For a matrix of labels (shape: num_segments × time_length),
    calculate the duration (in seconds) between the middle points of when 
    label_a starts and ends, and when label_b starts and ends,
    across *all* segments.

    Args:
      label_matrix: 2D np.ndarray of ints, shape (N, T)
      label_a: starting label in first direction (default 1)
      label_b: starting label in reverse (default 2)
      zero_label: label representing "gap" (default 0)
      allow_other_labels: if False, then if any intervening label
        is neither zero_label nor the target, the gap is skipped.
        If True, zero_label counts still accumulate even if other labels appear.
      signal_freq: sampling rate in Hz (default 200)

    Returns:
      gaps_a_to_b: list of durations (in seconds) for each a→b transition
      gaps_b_to_a: list of durations (in seconds) for each b→a transition
      df: pandas DataFrame with columns ["direction", "zero_count"]
          with direction values in {"a→b", "b→a"} for plotting convenience.
    """
    num_segments, T = label_matrix.shape
    gaps_a_to_b: List[float] = []
    gaps_b_to_a: List[float] = []

    for seg_idx in range(num_segments):
        labels = label_matrix[seg_idx]
        i = 0
        while i < T:
            if labels[i] == label_a:
                # look ahead for first label_b
                j = i + 1
                valid = True
                while j < T and labels[j] != label_b:
                    if labels[j] != zero_label and labels[j] != label_a:
                        if not allow_other_labels:
                            valid = False
                            break
                    j += 1
                if valid and j < T and labels[j] == label_b:
                    # Calculate duration between middle points
                    # Find start and end of label_a
                    a_start = i
                    a_end = i
                    while a_end < T and labels[a_end] == label_a:
                        a_end += 1
                    a_middle = (a_start + a_end) / 2.0
                    
                    # Find start and end of label_b
                    b_start = j
                    b_end = j
                    while b_end < T and labels[b_end] == label_b:
                        b_end += 1
                    b_middle = (b_start + b_end) / 2.0
                    
                    # Calculate duration in seconds
                    duration = (b_middle - a_middle) / signal_freq
                    gaps_a_to_b.append(duration)
                    i = j
                else:
                    i += 1
            elif labels[i] == label_b:
                # reverse direction
                j = i + 1
                valid = True
                while j < T and labels[j] != label_a:
                    if labels[j] != zero_label and labels[j] != label_b:
                        if not allow_other_labels:
                            valid = False
                            break
                    j += 1
                if valid and j < T and labels[j] == label_a:
                    # Calculate duration between middle points
                    # Find start and end of label_b
                    b_start = i
                    b_end = i
                    while b_end < T and labels[b_end] == label_b:
                        b_end += 1
                    b_middle = (b_start + b_end) / 2.0
                    
                    # Find start and end of label_a
                    a_start = j
                    a_end = j
                    while a_end < T and labels[a_end] == label_a:
                        a_end += 1
                    a_middle = (a_start + a_end) / 2.0
                    
                    # Calculate duration in seconds
                    duration = (a_middle - b_middle) / signal_freq
                    gaps_b_to_a.append(duration)
                    i = j
                else:
                    i += 1
            else:
                i += 1

    # Build dataframe for convenience
    records = ([
        {"direction": f"{label_a}→{label_b}", "zero_count": z}
        for z in gaps_a_to_b
    ] + [
        {"direction": f"{label_b}→{label_a}", "zero_count": z}
        for z in gaps_b_to_a
    ])
    df = pd.DataFrame.from_records(records)

    return gaps_a_to_b, gaps_b_to_a, df

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_upstroke_downstroke_histograms(upstroke_cc, downstroke_cc, upstroke_nh, downstroke_nh):
    """
    Create a 2x2 plot with histograms of upstroke and downstroke values for CC and NH conditions.
    
    Parameters:
    upstroke_cc, downstroke_cc, upstroke_nh, downstroke_nh: arrays of values to plot
    """
    # Create a 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Histograms of Upstroke and Downstroke Durations', fontsize=16)
    
    # Plot 1: Upstroke CC
    sns.histplot(upstroke_cc, kde=True, ax=axes[0,0])
    axes[0,0].set_title('Upstroke CC')
    axes[0,0].set_xlabel('Duration (sec)')
    
    # Plot 2: Downstroke CC
    sns.histplot(downstroke_cc, kde=True, ax=axes[0,1])
    axes[0,1].set_title('Downstroke CC')
    axes[0,1].set_xlabel('Duration (sec)')
    
    # Plot 3: Upstroke NH
    sns.histplot(upstroke_nh, kde=True, ax=axes[1,0])
    axes[1,0].set_title('Upstroke NH')
    axes[1,0].set_xlabel('Duration (sec)')
    
    # Plot 4: Downstroke NH
    sns.histplot(downstroke_nh, kde=True, ax=axes[1,1])
    axes[1,1].set_title('Downstroke NH')
    axes[1,1].set_xlabel('Duration (sec)')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_upstroke_downstroke_histograms(upstroke_cc, downstroke_cc, upstroke_nh, downstroke_nh)
