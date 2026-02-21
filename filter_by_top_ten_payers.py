"""
Filter hospital pricing CSV to only rows whose payer matches the US "top ten" list,
allowing for abbreviations / alternative names via normalization + keyword matching.

1 Define the top-ten payer list
2 Print the unique payer values found in the hospital system file
3 Build a  matcher:
   - normalizes text (lowercase, removes punctuation, expands common abbreviations)
   - uses keyword rules
4 Keeps only matched rows, drops the rest, and write a new filtered CSV
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# =========================
# 1) Top-ten payer list
# =========================
TOP_TEN_PAYERS = [
    "UnitedHealth Group",
    "Elevance Health (formerly Anthem)",
    "CVS Health (including Aetna)",
    "Centene Corporation",
    "Health Care Service Corporation (HCSC)",
    "Cigna Healthcare",
    "Kaiser Permanente",
    "Humana",
    "GuideWell (Florida Blue)",
    "Blue Cross Blue Shield",
]

# Canonical labels for filtered output
CANONICAL = {
    "unitedhealth": "UnitedHealth Group",
    "elevance": "Elevance Health (formerly Anthem)",
    "cvs_aetna": "CVS Health (including Aetna)",
    "centene": "Centene Corporation",
    "hcsc": "Health Care Service Corporation (HCSC)",
    "cigna": "Cigna Healthcare",
    "kaiser": "Kaiser Permanente",
    "humana": "Humana",
    "guidewell_floridablue": "GuideWell (Florida Blue)",
    "bcbs": "Blue Cross Blue Shield",
}

# Keywords for matching 
KEYWORDS = {
    "unitedhealth": [
        "unitedhealth", "united healthcare", "unitedhealthcare", "uhc", "optum"
    ],
    "elevance": [
        "elevance", "anthem", "amerigroup"  # Amerigroup is Elevance
    ],
    "cvs_aetna": [
        "aetna", "cvs health", "cvs"
    ],
    "centene": [
        "centene", "ambetter", "wellcare", "sunshine health", "peach state", "buckeye"
    ],
    "hcsc": [
        "health care service corporation", "hcsc",
        "bcbstx", "bcbs tx", "blue cross texas",
        "bcbsil", "bcbs il", "blue cross illinois",
        "bcbsok", "bcbs ok", "blue cross oklahoma",
        "bcbsnm", "bcbs nm", "blue cross new mexico",
        "bcbsmt", "bcbs mt", "blue cross montana",
    ],
    "cigna": [
        "cigna", "evernorth"
    ],
    "kaiser": [
        "kaiser", "kp", "kaiser permanente"
    ],
    "humana": [
        "humana"
    ],
    "guidewell_floridablue": [
        "guidewell", "florida blue", "blue cross florida", "bcbs fl", "bcbsfl"
    ],
    "bcbs": [
        # Catch all for Blue Cross Blue Shield variants that arent obviously HCSC
        "blue cross", "bluecross", "blue shield", "blueshield", "bcbs"
    ],
}


# ==========================
# Text normalization helpers
# ===========================

def normalize_text(s: str) -> str:
    """Lowercase, remove accents, normalize punctuation/whitespace, expand common abbrevs."""
    if s is None:
        return ""
    s = str(s)

    # Unicode normalize (strip accents) # Thanks, Brendon!
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower()

    # Expand common abbreviations before punctuation removal
    replacements = {
        "&": " and ",
        "/": " ",
        "u h c": " uhc ",
        "bc b s": " bcbs ",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Remove punctuation -> spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def simple_similarity(a: str, b: str) -> float: # thank you, again
    """
    Optional lightweight similarity (no external deps).
    Uses token overlap Jaccard as a cheap proxy.
    """
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# =========================
# 3) Matcher
# ==========================
def match_to_top_ten(payer_raw: str, min_jaccard: float = 0.45) -> Tuple[Optional[str], str]:
    """
    Returns (canonical_payer_or_None, match_reason).
    Match strategy:
      1 keyword contains rules
      2 fallback: token Jaccard similarity against canonical names
    """
    p = normalize_text(payer_raw)

    if not p:
        return None, "empty"

    # 1) keyword rules
    # Special handling: if it looks like HCSC-specific BCBS entities, prefer HCSC
    for key, kws in KEYWORDS.items():
        for kw in kws:
            if normalize_text(kw) in p:
                # If it matched generic bcbs but also looks like HCSC, let HCSC win later
                return CANONICAL[key], f"keyword:{key}"

    # 2) fallback similarity to canonical names (normalized)
    canon_norm = {v: normalize_text(v) for v in CANONICAL.values()}
    best_name = None
    best_score = 0.0
    for canon_name, canon_n in canon_norm.items():
        score = simple_similarity(p, canon_n)
        if score > best_score:
            best_score = score
            best_name = canon_name

    if best_name is not None and best_score >= min_jaccard:
        return best_name, f"jaccard:{best_score:.2f}"

    return None, "no_match"


# =========================
# Main
# ==========================

def main():
    # ======================
    # ===== EDIT THESE =====
    input_csv = Path(r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files\Jefferson_pricing_long.csv")
    payer_col = "payer"  # change if your column name differs (e.g., "Payer_Name")
    output_csv = input_csv.with_name(input_csv.stem + "_TOP10_PAYERS_ONLY.csv")
    # ======================

    df = pd.read_csv(input_csv, dtype=str, low_memory=False)

    if payer_col not in df.columns:
        raise ValueError(
            f"Column '{payer_col}' not found. Available columns:\n{list(df.columns)}"
        )

    # 2) unique list of payers in the hospital file
    unique_payers = (
        df[payer_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )
    print(f"\nUnique payers in file ({len(unique_payers)}):")
    for p in sorted(unique_payers)[:200]:
        print("  ", p)
    if len(unique_payers) > 200:
        print("  ... (showing first 200)")

    # 3) match and create canonical column
    matches = df[payer_col].apply(lambda x: match_to_top_ten(x))
    df["payer_top10_canonical"] = matches.apply(lambda t: t[0])
    df["payer_match_reason"] = matches.apply(lambda t: t[1])

    # 4) filter to matched payers only
    filtered = df[df["payer_top10_canonical"].notna()].copy()

    print("\nMatch counts by canonical payer:")
    print(filtered["payer_top10_canonical"].value_counts(dropna=False))

    print(f"\nRows before: {len(df):,}")
    print(f"Rows after : {len(filtered):,}")
    print(f"Dropped    : {len(df) - len(filtered):,}")

    filtered.to_csv(output_csv, index=False)
    print(f"\nSaved filtered CSV to:\n{output_csv}\n")


if __name__ == "__main__":
    main()

