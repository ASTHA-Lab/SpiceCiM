# data_collect.py
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def read_and_process_file(file_path):
    """
    Reads a raw .tran file and returns two 1D numpy arrays:
      - params:   the name of each sample (e.g. '"time"', '"IPRB0:in"', …)
      - values:   the corresponding floating‑point value
    """
    params = []
    values = []
    with open(file_path, 'r') as f:
        header_passed = False
        for line in f:
            if not header_passed:
                if "VALUE" in line:
                    header_passed = True
                continue
            line = line.strip().lstrip('(').rstrip(')')
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            params.append(parts[0])
            try:
                values.append(float(parts[1]))
            except ValueError:
                values.append(np.nan)

    return np.array(params), np.array(values)


def time_index_finder(params, values, start_time, increment):
    """
    Finds the start/end indices (in the flat params/values arrays) for each
    [start_time + k*increment, start_time + (k+1)*increment) interval until EOF.
    Uses a single searchsorted on the sorted time vector.
    """
    # extract all the time samples
    time_mask = (params == '"time"')
    time_idxs = np.nonzero(time_mask)[0]
    time_vals = values[time_idxs]

    start_indices = []
    end_indices = []
    t0 = start_time

    # We know time_vals is sorted ascending; we'll march t0 forward
    while True:
        # find first time ≥ t0
        j0 = np.searchsorted(time_vals, t0, side='left')
        if j0 >= len(time_vals):
            break
        idx0 = time_idxs[j0]
        start_indices.append(idx0)

        # now find first time ≥ t0 + increment
        t1 = t0 + increment
        j1 = np.searchsorted(time_vals, t1, side='left')
        if j1 >= len(time_vals):
            # last bin runs to the end of the dataset
            idx1 = time_idxs[-1]
            end_indices.append(idx1)
            break
        idx1 = time_idxs[j1]
        end_indices.append(idx1)

        t0 = t1

    return start_indices, end_indices


def max_abs_value_finder(params, values, start_index, end_index, target_net):
    """
    Returns the single largest absolute value of `target_net` between start_index
    and end_index (inclusive).  Returns None if no samples of target_net appear.
    """
    seg_params = params[start_index:end_index+1]
    seg_vals   = values[start_index:end_index+1]
    mask = (seg_params == f'"{target_net}"')
    if not np.any(mask):
        return None
    sub = seg_vals[mask]
    return float(np.abs(sub).max())


def process_folder(folder_path, start_time_tick, increment, ncolumn):
    """
    Walk into folder_path/psf/tran.tran.tran, build the time windows once,
    then for each IPRB<k>:in net grab the max‐abs in each window.
    """
    target_path = os.path.join(folder_path, 'psf', 'tran.tran.tran')
    if not os.path.exists(target_path):
        return None

    # load into two NumPy arrays
    params, values = read_and_process_file(target_path)

    # build all our windows
    starts, ends = time_index_finder(params, values, start_time_tick, increment)
    if not starts:
        return None

    # prepare our output row
    result_row = {'Folder': os.path.basename(folder_path)}
    iprb_labels = [f'IPRB{col}:in' for col in range(ncolumn)]

    # for each net, run max‐abs over each window
    for label in iprb_labels:
        lst = []
        for s, e in zip(starts, ends):
            lst.append(max_abs_value_finder(params, values, s, e, label))
        result_row[label] = lst

    return result_row


def process_files_in_folder(base_path, start_time_tick, increment, output_csv_path, ncolumn):
    """
    Parallelizes process_folder() over all subdirectories of base_path,
    then dumps the combined DataFrame to CSV (same format as before).
    """
    # collect all sub‑folders
    folders = [os.path.join(r, d) for r, dirs, _ in os.walk(base_path) for d in dirs]
    data = []

    with ProcessPoolExecutor() as exe:
        futures = [exe.submit(process_folder, fld, start_time_tick, increment, ncolumn)
                   for fld in folders]
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                data.append(res)

    df_out = pd.DataFrame(data)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Output CSV file saved at: {output_csv_path}")

