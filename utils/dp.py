import pandas as pd
import ast
from collections import defaultdict
from typing import List, Tuple

import json
import numpy as np

def compute_mapping_ranges(layer_weights: dict, layer_biases: dict,
                           X_train: np.ndarray, file_out: str):
    """
    layer_weights, layer_biases: Dict like {'Layer1': W1, 'Layer2': W2, …}
    X_train: (N_samples, N_inputs) raw training data
    """
    ranges = {}
    # flatten image‐shaped input into (N_samples, nin)
    activations = X_train.copy()
    if activations.ndim > 2:
        activations = activations.reshape(activations.shape[0], -1)

    # Sort by numerical layer index
    for name in sorted(layer_weights.keys(),
                       key=lambda n: int(n.replace('Layer',''))):
        W = layer_weights[name]      # Keras Dense kernel: shape = (nin, nout)
        b = layer_biases[name]       # bias: shape = (nout,)

        # compute raw currents: (N_samples, nin) · (nin, nout) → (N_samples, nout)
        currents = activations.dot(W) + b[np.newaxis, :]

        ranges[name] = {
            "min": float(currents.min()),
            "max": float(currents.max())
        }

        # propagate through ReLU for next layer
        activations = np.maximum(currents, 0)

    with open(file_out, 'w') as f:
        json.dump(ranges, f, indent=2)

    return ranges



# SpiceCim.py
import json, numpy as np

# load once at top


def current_to_voltage_fixed(currents, layer_idx, vmax):
    """
    currents: list of lists, shape (n_samples, n_neurons)
    returns same shape lists with voltages in [0, vmax]
    """
    with open("./tmp/mapping_ranges.json") as f:
        _RANGES = json.load(f)
    arr = np.array(currents, dtype=float)
    r = _RANGES[f"{layer_idx}"]
    cmin, cmax = r["min"], r["max"]
    # avoid division by zero
    scale = vmax / (cmax - cmin) if cmax != cmin else 1.0
    volt = (arr - cmin) * scale
    # clamp
    volt = np.clip(volt, 0, vmax)
    return volt.tolist()



def current_to_voltage(
    summed_currents: List[List[float]],
    vmax: float
) -> List[List[float]]:
    """
    Convert each summed current (per flattened column) into a voltage in [0, vmax].
    - Negative currents → 0 V
    - Positive currents are linearly mapped so that imin → 0, imax → vmax,
      where imin/imax are the min/max over all non‑negative currents.
    """
    # flatten, pick non‑negative
    all_vals = [c for col in summed_currents for c in col]
    non_neg = [c for c in all_vals if c >= 0]
    if not non_neg:
        raise ValueError("No non‑negative currents found for scaling.")
    imin, imax = min(non_neg), max(non_neg)

    voltages: List[List[float]] = []
    for col in summed_currents:
        col_v = []
        for c in col:
            if c < 0:
                v = 0.0
            else:
                if imax == imin:
                    # all currents identical → max out
                    v = vmax
                else:
                    v = ( (c - imin) / (imax - imin) ) * vmax
            col_v.append(v)
        voltages.append(col_v)
    return voltages


# SpiceCim.py (below current_to_voltage_fixed)
#import numpy as np

def bias_to_voltage_signed(bias_list, layer_idx, vmax):
    """
    bias_list: list of floats *or* singleton sequences ([[b0], [b1], …])
    layer_idx: integer 1, 2, … matching your JSON keys "Layer1", "Layer2", …
    vmax: maximum absolute voltage

    Returns: [[v0], [v1], …] with v ∈ [-vmax, +vmax]
    """
    # 1) load the precomputed min/max for this layer
    with open("./tmp/mapping_ranges.json") as f:
        mapping = json.load(f)
    key = f"{layer_idx}"
    if key not in mapping:
        raise KeyError(f"Could not find mapping ranges for '{key}' in JSON")
    r = mapping[key]
    
    # 2) compute symmetric scale from ±max_abs_current -> ±vmax
    max_abs = max(abs(r["min"]), abs(r["max"]), 1e-12)
    scale   = vmax / max_abs
    
    # 3) unpack, scale, clamp, and re‑wrap
    out = []
    for b in bias_list:
        # unpack if b is a sequence
        val = float(b[0]) if isinstance(b, (list, tuple, np.ndarray)) else float(b)
        v   = val * scale
        v   = float(np.clip(v, -vmax, vmax))
        out.append([v])
    return out


def process_crossbar_csv_to_image_csv(
    input_path: str,
    output_path: str,
    layer_id,
    vmax: float = 1.0,
) -> List[List[float]]:
    """
    1) Reads the input CSV (Folder + time‑series lists),
    2) computes element‑wise (pos – neg),
    3) sums across all NN‑rows → `summed` currents,
    4) converts to voltages via `current_to_voltage(..., vmax)`,
    5) writes final CSV with rows=Image_0…Image_{L-1}, cols=0…M-1.
    Returns the voltage data: List of M lists of length L.
    """
    df = pd.read_csv(input_path)
    cur_cols = [c for c in df.columns if c != 'Folder']

    # parse into data[row][col]['pos'/'neg']
    data = defaultdict(lambda: defaultdict(dict))
    for _, row in df.iterrows():
        _, r_str, c_str, sign = row['Folder'].split('_')
        r, c = int(r_str), int(c_str)
        arr = []
        for col in cur_cols:
            lst = ast.literal_eval(row[col])
            cleaned = [x for x in lst if x is not None]
            arr.append(cleaned)
        data[r][c][sign] = arr

    # dimensions
    n_rows = max(data.keys()) + 1
    n_cols = max(next(iter(data.values())).keys()) + 1
    M = n_cols * len(cur_cols)   # total flattened columns

    # compute flattened per‑row
    flattened: List[List[List[float]]] = []
    for r in range(n_rows):
        row_flat: List[List[float]] = []
        for c in range(n_cols):
            pos = data[r][c].get('pos')
            neg = data[r][c].get('neg')
            if pos is None or neg is None:
                raise ValueError(f"Missing pos/neg for row {r}, col {c}")
            for p, n in zip(pos, neg):
                if len(p) != len(n):
                    print("Warning: Mismatched lengths in a time‑series")
                    # Truncate both to the same (shorter) length
                    min_len = min(len(p), len(n))
                    p = p[:min_len]
                    n = n[:min_len]
                row_flat.append([pi - ni for pi, ni in zip(p, n)])
        flattened.append(row_flat)

    # sum across rows, element‑wise → summed[j][t]
    L = len(flattened[0][0])
    summed: List[List[float]] = []
    for j in range(M):
        summed.append([
            sum(flattened[r][j][t] for r in range(n_rows))
            for t in range(L)
        ])

    # convert currents → voltages
    voltages = current_to_voltage_fixed(summed, layer_id, vmax)

    # build and write final CSV
    rows = []
    for t in range(L):
        row = {'Image_name': f'Image_{t}'}
        for j in range(M):
            row[str(j)] = voltages[j][t]
        rows.append(row)
    out_df = pd.DataFrame(rows, columns=['Image_name'] + [str(j) for j in range(M)])
    out_df.to_csv(output_path, index=False)

    return voltages

def compare_predictions(voltage_csv_path, tensor_csv_path, output_csv_path):
    """
    Compare hardware vs. software predictions.

    Parameters
    ----------
    voltage_csv_path : str
        Path to the hardware CSV. First column 'Image_name', remaining columns are class labels.
    tensor_csv_path : str
        Path to the software CSV. Each row looks like:
            Prediction for image_0.jpg,[2]
    output_csv_path : str
        Path where to write the comparison CSV, with columns:
            Image_name, Software_pred, Hardware_pred, Status
    """
    # 1) Read hardware voltages
    hw_df = pd.read_csv(voltage_csv_path)
    # Determine which class-column has the max for each row
    # idxmax returns the column label (e.g. '2', '9', etc.)
    hw_df['Hardware_pred'] = hw_df.drop(columns='Image_name').idxmax(axis=1).astype(int)
    
    # Keep only image names + hw predictions
    results_df = hw_df[['Image_name', 'Hardware_pred']].copy()
    
    # 2) Read software predictions
    # tensorOut.csv has a header row "image_name,software_pred"
    sw_df = pd.read_csv(tensor_csv_path)
    # Rename to match our conventions
    sw_df.rename(columns={'image_name':'Image_name',
                          'software_pred':'Software_pred'},
                 inplace=True)
    # Ensure Software_pred is integer
    sw_df['Software_pred'] = sw_df['Software_pred'].astype(int)
    # Create lower-case, extension-stripped key for merging
    sw_df['img_lc'] = (
        sw_df['Image_name']
          .str.lower()
          .str.replace(r'\.[^.]+$','', regex=True)
    )

    # 3) Merge on image name (case‐insensitive, just in case)
    #results_df['img_lc'] = results_df['Image_name'].str.lower()
    #sw_df['img_lc']      = sw_df['Image_name'].str.lower()
    #merged = pd.merge(
        #results_df,
        #sw_df[['img_lc','Software_pred']],
        #left_on='img_lc',
        #right_on='img_lc',
        #how='inner'
    #)

    # 3) Merge on image name (case‐insensitive, extension stripped)
    results_df['img_lc'] = (
        results_df['Image_name']
          .str.lower()
          .str.replace(r'\.[^.]+$','', regex=True)
    )
    merged = pd.merge(
        results_df,
        sw_df[['img_lc','Software_pred']],
        on='img_lc',
        how='inner'
    )

    # 4) Compare and flag Pass/Fail
    merged['Status'] = merged.apply(
        lambda r: 'Pass' if r['Hardware_pred'] == r['Software_pred'] else 'Fail',
        axis=1
    )

    # 5) Summary stats
    pass_count = (merged['Status'] == 'Pass').sum()
    fail_count = (merged['Status'] == 'Fail').sum()
    accuracy   = pass_count / (pass_count + fail_count) * 100

    # 6) Write out the comparison CSV
    out_df = merged[['Image_name', 'Software_pred', 'Hardware_pred', 'Status']]
    out_df.to_csv(output_csv_path, index=False)

    # 7) Print results
    print(f'Pass count: {pass_count}')
    print(f'Fail count: {fail_count}')
    print(f'Hardware accuracy: {accuracy:.2f}%')
    

