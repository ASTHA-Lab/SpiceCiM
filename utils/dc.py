# data_collect.py
import os
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

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

def process_power_energy_in_folder(base_path: str,
                                   output_csv_path: str,
                                   power_label: str = ':pwr') -> None:
    """
    Recursively scan each immediate subfolder of `base_path` for Spectre psfascii results
    under **.../psf/tran.tran.tran**, compute average power and total energy from the
    `power_label` waveform (default ':pwr'), and write two CSVs:

      - <output_csv_path base>_pwr.csv : columns ['Folder', 'PWR'] (+ a final 'Total_pwr' row)
      - <output_csv_path base>_eng.csv : columns ['Folder', 'ENG'] (+ a final 'Total_eng' row)

    Rows with basenames matching {'psf'} or regexes r'^PE\\d+$' and r'^Tile\\d+$' are excluded.
    The total rows are elementwise sums across rows whose Folder starts with 'Array_'.

    NOTE: This function is standalone and does not modify other code. It reuses the existing
    `read_and_process_file()` defined in this module.
    """
    import re  # local import to avoid changing module-level imports

    # Derive output file names
    base, _ = os.path.splitext(output_csv_path)
    pwr_csv_path = f"{base}_pwr.csv"
    eng_csv_path = f"{base}_eng.csv"

    # Collect immediate sub-folders under base_path (same semantics as process_files_in_folder)
    folders_all = sorted({os.path.join(r, d) for r, dirs, _ in os.walk(base_path) for d in dirs})

    # Filter out unwanted basenames
    def _keep_folder(path: str) -> bool:
        bn = os.path.basename(path)
        if bn == 'psf':
            return False
        if re.fullmatch(r'PE\d+', bn):
            return False
        if re.fullmatch(r'Tile\d+', bn):
            return False
        return True

    folders = [f for f in folders_all if _keep_folder(f)]

    # If no folders, still emit empty CSVs with headers for reproducibility
    if not folders:
        pd.DataFrame(columns=['Folder', 'PWR']).to_csv(pwr_csv_path, index=False)
        pd.DataFrame(columns=['Folder', 'ENG']).to_csv(eng_csv_path, index=False)
        #print(f"Output CSV files saved at: {pwr_csv_path} and {eng_csv_path}")
        return

    pwr_rows: List[Dict[str, List[Optional[float]]]] = []
    eng_rows: List[Dict[str, List[Optional[float]]]] = []

    # Helper: compute (avg_power, energy) for a tran file using the psfascii reader
    def _power_energy_from_tran(file_path: str, power_label: str = power_label) -> Optional[Tuple[float, float]]:
        params, values = read_and_process_file(file_path)
        if params.size == 0 or values.size == 0:
            return None

        # time vector
        t_mask = (params == '"time"')
        if not np.any(t_mask):
            return None
        t_vals = values[t_mask]
        t_vals = t_vals[np.isfinite(t_vals)]
        if t_vals.size < 2:
            return None

        # power vector
        p_token = f'"{power_label}"'
        p_mask = (params == p_token)
        if not np.any(p_mask):
            return None
        p_vals = values[p_mask]
        p_vals = p_vals[np.isfinite(p_vals)]
        if p_vals.size == 0:
            return None

        n = min(t_vals.size, p_vals.size)
        if n < 2:
            return None
        t_vals = t_vals[:n]
        p_vals = p_vals[:n]

        # Ensure non-decreasing time
        if not np.all(np.diff(t_vals) >= 0):
            order = np.argsort(t_vals)
            t_vals = t_vals[order]
            p_vals = p_vals[order[:p_vals.size]] if p_vals.size != order.size else p_vals[order]

        duration = float(t_vals[-1] - t_vals[0])
        if duration <= 0:
            return None

        energy_J = float(np.trapz(p_vals, t_vals))
        avg_power_W = float(energy_J / duration)
        return avg_power_W, energy_J

    # For each immediate subfolder, recursively find all .../psf/tran.tran.tran files
    for folder in folders:
        tran_files: List[str] = []
        for r, _, files in os.walk(folder):
            if 'tran.tran.tran' in files and os.path.basename(r) == 'psf':
                tran_files.append(os.path.join(r, 'tran.tran.tran'))
        tran_files.sort()  # stable, deterministic order

        # Aggregate results for this folder
        avg_p_list: List[Optional[float]] = []
        eng_list: List[Optional[float]] = []
        for tf in tran_files:
            res = _power_energy_from_tran(tf, power_label=power_label)
            if res is None:
                avg_p_list.append(np.nan)
                eng_list.append(np.nan)
            else:
                avg_w, eng_j = res
                avg_p_list.append(avg_w)
                eng_list.append(eng_j)

        pwr_rows.append({'Folder': os.path.basename(folder), 'PWR': avg_p_list})
        eng_rows.append({'Folder': os.path.basename(folder), 'ENG': eng_list})

    # Sort rows by Folder for stable output
    pwr_rows.sort(key=lambda d: d.get('Folder', ''))
    eng_rows.sort(key=lambda d: d.get('Folder', ''))

    # Build DataFrames
    df_pwr = pd.DataFrame(pwr_rows, columns=['Folder', 'PWR'])
    df_eng = pd.DataFrame(eng_rows, columns=['Folder', 'ENG'])

    # Compute totals across rows whose Folder starts with 'Array_'
    def _elementwise_sum_lists(series_of_lists: pd.Series) -> List[float]:
        # Determine max length
        max_len = max((len(lst) if isinstance(lst, list) else 0) for lst in series_of_lists)
        if max_len == 0:
            return []
        # Stack with NaN padding then nansum
        mat = np.full((len(series_of_lists), max_len), np.nan, dtype=float)
        for i, lst in enumerate(series_of_lists):
            if isinstance(lst, list) and lst:
                n = min(len(lst), max_len)
                mat[i, :n] = lst[:n]
        # Elementwise sum ignoring NaNs
        return list(np.nansum(mat, axis=0))

    array_mask_pwr = df_pwr['Folder'].astype(str).str.startswith('Array_')
    array_mask_eng = df_eng['Folder'].astype(str).str.startswith('Array_')

    total_pwr_list = _elementwise_sum_lists(df_pwr.loc[array_mask_pwr, 'PWR'])
    total_eng_list = _elementwise_sum_lists(df_eng.loc[array_mask_eng, 'ENG'])

    # Append totals as final rows
    df_pwr = pd.concat([df_pwr, pd.DataFrame([{'Folder': 'Total_pwr', 'PWR': total_pwr_list}])],
                       ignore_index=True)
    df_eng = pd.concat([df_eng, pd.DataFrame([{'Folder': 'Total_eng', 'ENG': total_eng_list}])],
                       ignore_index=True)

    # Write CSVs
    df_pwr.to_csv(pwr_csv_path, index=False)
    df_eng.to_csv(eng_csv_path, index=False)
    #print(f"Output CSV files saved at: {pwr_csv_path} and {eng_csv_path}")
