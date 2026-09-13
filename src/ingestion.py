"""
Ingestion module for Saudi OGD AI Readiness Audit.
Handles encoding heterogeneity (UTF-8, UTF-8-BOM, Windows-1256, Latin-1)
and produces (DataFrame, metadata) tuples per file.
"""
import os
import hashlib
from datetime import datetime
from pathlib import Path

import chardet
import pandas as pd

ENCODING_CASCADE = ["utf-8", "utf-8-sig", "windows-1256", "latin-1"]
DELIMITERS = [",", ";", "\t", "|"]


def detect_encoding(path: str) -> str:
    """Sniff encoding from the first 65 KB, fall back to UTF-8."""
    with open(path, "rb") as f:
        raw = f.read(65536)
    res = chardet.detect(raw)
    return res.get("encoding") or "utf-8"


def try_read_csv(path: str) -> tuple[pd.DataFrame, str, str]:
    """
    Try the encoding cascade with multiple delimiters.
    Returns (df, encoding_used, delimiter_used).
    """
    detected = detect_encoding(path)
    cascade = [detected] + [e for e in ENCODING_CASCADE if e != detected]
    last_exc = None
    for enc in cascade:
        for sep in DELIMITERS:
            try:
                df = pd.read_csv(
                    path,
                    encoding=enc,
                    sep=sep,
                    engine="python",
                    on_bad_lines="skip",
                )
                # plausibility: prefer >=2 cols, accept 1-col only if comma-sep
                if df.shape[1] >= 2:
                    return df, enc, sep
                # If only one column and we haven't tried other seps, keep going
            except (UnicodeDecodeError, pd.errors.ParserError, Exception) as e:
                last_exc = e
                continue
    # fallback: re-try with comma and accept single-column result
    for enc in cascade:
        try:
            df = pd.read_csv(path, encoding=enc, sep=",", engine="python", on_bad_lines="skip")
            return df, enc, ","
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError(f"Could not parse {path}; last error: {last_exc}")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_columns(df: pd.DataFrame) -> dict:
    """Return counts of numeric / categorical / date / freetext columns."""
    counts = {"numeric": 0, "categorical": 0, "date": 0, "freetext": 0}
    for col in df.columns:
        s = df[col]
        # date detection: try parsing a sample
        is_date = False
        sample = s.dropna().astype(str).head(50)
        if len(sample) > 0:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if parsed.notna().mean() > 0.8:
                    is_date = True
            except Exception:
                pass
        if is_date:
            counts["date"] += 1
        elif pd.api.types.is_numeric_dtype(s):
            counts["numeric"] += 1
        else:
            # distinguish categorical vs freetext by cardinality and length
            nunique = s.nunique(dropna=True)
            if len(s) > 0 and nunique / max(len(s), 1) > 0.5 and \
               s.dropna().astype(str).str.len().mean() > 20:
                counts["freetext"] += 1
            else:
                counts["categorical"] += 1
    return counts


SECTOR_DIRS = {
    "Investment": "Investment",
    "Health": "health",
    "Education": "education",
    "Energy": "energy",
}

SECTOR_PREFIX = {
    "Investment": "INV",
    "Health": "HEA",
    "Education": "EDU",
    "Energy": "ENE",
}


def build_inventory(dataset_root: str) -> pd.DataFrame:
    """
    Walk the dataset/{Sector}/ folders, attempt to read every CSV,
    return an inventory DataFrame with one row per file.
    """
    rows = []
    for sector, subdir in SECTOR_DIRS.items():
        sector_path = Path(dataset_root) / subdir
        if not sector_path.exists():
            continue
        files = sorted([p for p in sector_path.iterdir() if p.suffix.lower() == ".csv"])
        prefix = SECTOR_PREFIX[sector]
        for i, p in enumerate(files, start=1):
            try:
                df, enc, sep = try_read_csv(str(p))
                col_types = classify_columns(df)
                row = {
                    "dataset_id": f"{prefix}-{i:02d}",
                    "sector": sector,
                    "file_name": p.name,
                    "file_path": str(p),
                    "encoding": enc,
                    "delimiter": sep,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "total_cells": len(df) * len(df.columns),
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "sha256": sha256_of(str(p)),
                    "numeric_cols": col_types["numeric"],
                    "categorical_cols": col_types["categorical"],
                    "date_cols": col_types["date"],
                    "freetext_cols": col_types["freetext"],
                    "load_ok": True,
                }
                rows.append(row)
            except Exception as e:
                rows.append({
                    "dataset_id": f"{prefix}-{i:02d}",
                    "sector": sector,
                    "file_name": p.name,
                    "file_path": str(p),
                    "load_ok": False,
                    "error": str(e),
                })
    return pd.DataFrame(rows)


def load_dataset(file_path: str) -> tuple[pd.DataFrame, dict]:
    """Convenience loader returning (df, profile_meta)."""
    df, enc, sep = try_read_csv(file_path)
    meta = {
        "encoding": enc,
        "delimiter": sep,
        "rows": len(df),
        "cols": len(df.columns),
        "loaded_at": datetime.utcnow().isoformat(),
    }
    return df, meta
