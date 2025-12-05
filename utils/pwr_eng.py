import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd

# ---------- Parsing helpers ----------

_NUM_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

def _unwrap_wrappers(s: str) -> str:
    """Strip [], np.float64(...), float32(...), np.array([...]), array([...]) layers."""
    if s is None:
        return ""
    text = str(s).strip()

    # Strip outer [] repeatedly (e.g., "[1.23]" or "[[1.23]]")
    while text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()

    # Iteratively unwrap common wrappers a few times
    for _ in range(3):
        text = re.sub(r"(?:np\.)?(?:float|int)\d+\(\s*([^\)]*?)\s*\)", r"\1", text)  # np.float64(x) -> x
        text = re.sub(r"(?:np\.)?array\(\s*([^\)]*?)\s*\)", r"\1", text)             # np.array(x)   -> x
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
    return text

def _extract_numeric(text: str) -> Optional[float]:
    """Return the intended numeric value from a messy string."""
    cleaned = _unwrap_wrappers(text)
    m = _NUM_RE.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _read_csv_as_str_df(path: str) -> Optional[pd.DataFrame]:
    """Read CSV as strings, with a robust fallback."""
    try:
        return pd.read_csv(path, header=None, dtype=str, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, dtype=str)
        except Exception:
            return None

def _extract_total_from_csv(path: str, key: str) -> Optional[float]:
    """
    Find the numeric associated with `key` (e.g., 'Total_pwr' or 'Total_eng'):
      1) If a cell == key, take first numeric to the right; else below.
      2) If a cell contains key, try right cell; else parse after key in same cell.
      3) Fallback: raw-file text, first number after key.
    """
    df = _read_csv_as_str_df(path)

    if df is not None:
        # exact cell equals key
        where = (df == key)
        rows, cols = where.values.nonzero()
        if len(rows):
            r, c = rows[0], cols[0]
            # to the right
            for cc in range(c + 1, df.shape[1]):
                val = _extract_numeric(df.iat[r, cc])
                if val is not None:
                    return val
            # below
            if r + 1 < df.shape[0]:
                val = _extract_numeric(df.iat[r + 1, c])
                if val is not None:
                    return val

        # any cell containing the key
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                cell = df.iat[r, c]
                if cell is None:
                    continue
                s = str(cell)
                if key in s:
                    # prefer right neighbor
                    if c + 1 < df.shape[1]:
                        val = _extract_numeric(df.iat[r, c + 1])
                        if val is not None:
                            return val
                    # else parse after key within same cell
                    after = s.split(key, 1)[1]
                    val = _extract_numeric(after)
                    if val is not None:
                        return val

    # fallback: raw text
    try:
        txt = Path(path).read_text(errors="ignore")
        idx = txt.find(key)
        if idx != -1:
            after = txt[idx + len(key):]
            val = _extract_numeric(after)
            if val is not None:
                return val
    except Exception:
        pass

    return None

# ---------- Filename helpers ----------

# Match "..._<kind>( (n))?.csv" where kind is eng|pwr, case-insensitive
_SUFFIX_RE = re.compile(r"(?i)_(eng|pwr)\s*(?:\(\d+\))?\.csv$")

def _parse_layer_and_kind(filename: str) -> Optional[Tuple[str, str]]:
    """
    From a filename like '_FFN_W1_eng.csv' or '_FFN_W1_eng (1).csv',
    return ('_FFN_W1', 'eng'). Returns None if it doesn't match.
    """
    m = _SUFFIX_RE.search(filename)
    if not m:
        return None
    kind = m.group(1).lower()
    layer = filename[:m.start()]  # everything before the suffix pattern
    return layer, kind

# ---------- Main API ----------

def generate_power_energy_summary(folder_path: str,
                                  output_name: str = "power_energy_summary.csv") -> str:
    """
    Recursively scans `folder_path` for *_pwr*.csv and *_eng*.csv (handles duplicates
    like '(1)' suffix). Sums Total_pwr and Total_eng per layer across ALL subfolders.
    Writes a summary CSV with a final 'Total' row.
    """
    folder = Path(folder_path)

    # Accumulators track sum and count so we can leave cells blank if no values were found.
    # records[layer] = {"Power": {"sum": float, "count": int}, "Energy": {"sum": float, "count": int}}
    records: Dict[str, Dict[str, Dict[str, float]]] = {}

    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue

            parsed = _parse_layer_and_kind(fn)
            if not parsed:
                continue

            layer, kind = parsed
            full_path = str(Path(root) / fn)

            if kind == "pwr":
                val = _extract_total_from_csv(full_path, "Total_pwr")
                if val is not None:
                    rec = records.setdefault(layer, {"Power": {"sum": 0.0, "count": 0},
                                                     "Energy": {"sum": 0.0, "count": 0}})
                    rec["Power"]["sum"] += val
                    rec["Power"]["count"] += 1

            elif kind == "eng":
                val = _extract_total_from_csv(full_path, "Total_eng")
                if val is not None:
                    rec = records.setdefault(layer, {"Power": {"sum": 0.0, "count": 0},
                                                     "Energy": {"sum": 0.0, "count": 0}})
                    rec["Energy"]["sum"] += val
                    rec["Energy"]["count"] += 1

    # Build final table
    rows = []
    for layer in sorted(records.keys()):
        p_sum = records[layer]["Power"]["sum"]
        p_cnt = records[layer]["Power"]["count"]
        e_sum = records[layer]["Energy"]["sum"]
        e_cnt = records[layer]["Energy"]["count"]
        rows.append({
            "Layer": layer,
            "Power": (p_sum if p_cnt > 0 else None),
            "Energy": (e_sum if e_cnt > 0 else None),
        })

    df = pd.DataFrame(rows).set_index("Layer") if rows else pd.DataFrame(columns=["Layer", "Power", "Energy"]).set_index("Layer")

    # Totals (ignore None with to_numeric coercion)
    power_total = pd.to_numeric(df.get("Power"), errors="coerce").sum() if "Power" in df else 0.0
    energy_total = pd.to_numeric(df.get("Energy"), errors="coerce").sum() if "Energy" in df else 0.0

    df_out = pd.concat([df, pd.DataFrame({"Power": [power_total], "Energy": [energy_total]}, index=["Total"])])

    out_path = str(folder / output_name)
    df_out.to_csv(out_path)
    print(f'Power and Energy summery file: {out_path}')
    return out_path



#generate_power_energy_summary(path)
