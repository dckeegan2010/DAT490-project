import re
import pandas as pd

# ============================
# stuff to edit before running
# ============================
file_path = r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files\232825878_Temple_University_Main_standardcharges.csv"
output_path = r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files\Temple_pricing_long.csv"

# If some files have metadata rows at top, set skiprows (UPenn used skiprows=2)
SKIPROWS = 2 # 2 for UPenn and it looks like everyone else
# ============================
# ============================


def normalize_colname(c: str) -> str:
    """Case-insensitive + whitespace-normalized key for column lookup."""
    return re.sub(r"\s+", " ", str(c)).strip().lower()


def build_col_maps(columns):
    """
    Return (norm->original, original->norm).
    """
    norm_to_orig = {}
    orig_to_norm = {}
    for c in columns:
        n = normalize_colname(c)
        norm_to_orig[n] = c
        orig_to_norm[c] = n
    return norm_to_orig, orig_to_norm


def pick_first_existing(norm_to_orig, candidates_norm):
    """Return the original column name for the first candidate that exists, else None."""
    for n in candidates_norm:
        if n in norm_to_orig:
            return norm_to_orig[n]
    return None


def extract_cpt_code(df: pd.DataFrame, norm_to_orig: dict) -> pd.Series:
    """
    Find CPT code from ANY code|N column where code|N|type contains 'CPT'.
    Returns a Series 'cpt_code' with the first valid CPT found in ascending N order.
    """
    # Find all code columns like "code|1", "code|2", ... (case-insensitive)
    code_cols = []
    for ncol, orig in norm_to_orig.items():
        if re.fullmatch(r"code\|\d+", ncol):
            code_cols.append((int(ncol.split("|")[1]), orig))
    code_cols.sort(key=lambda x: x[0])

    if not code_cols:
        # No code|N columns found
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    cpt = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

    for n, code_orig in code_cols:
        type_norm = f"code|{n}|type"
        type_orig = norm_to_orig.get(type_norm)

        # If there is no type column, we can't confirm CPT; skip this code|N.
        if type_orig is None:
            continue

        code_vals = df[code_orig].astype(str).str.strip()
        type_vals = df[type_orig].astype(str).str.upper()

        valid = (
            type_vals.str.contains("CPT", na=False)
            & ~code_vals.isin(["", "NAN", "NONE", "NULL"])
        )

        # Fill only where CPT not already assigned
        fill_mask = cpt.isna() & valid
        cpt.loc[fill_mask] = code_vals.loc[fill_mask]

    return cpt


def main():
    df = pd.read_csv(file_path, skiprows=SKIPROWS, low_memory=False)

    norm_to_orig, _ = build_col_maps(df.columns)

    # Identify negotiated columns (wide payer columns)
    # Primary: negotiated_dollar (UPenn-style), but also catch negotiated_rate if present.
    negotiated_cols = [
        col for col in df.columns
        if any(k in normalize_colname(col) for k in ["negotiated_dollar", "negotiated rate", "negotiated_rate"])
    ]

    if not negotiated_cols:
        raise ValueError(
            "No negotiated columns found. Expected headers containing 'negotiated_dollar' or 'negotiated_rate'."
        )

    # Common id/descriptor columns (case-insensitive)
    id_candidates = {
        "description": ["description"],
        "setting": ["setting"],
        "billing_class": ["billing_class", "billing class"],
        "modifiers": ["modifiers", "modifier"],
        "gross": ["standard_charge|gross", "standard charge|gross", "gross", "standard_charge_gross"],
        "cash": ["standard_charge|discounted_cash", "standard charge|discounted_cash", "discounted_cash", "cash"],
    }

    id_cols = []
    for _, candidates in id_candidates.items():
        found = pick_first_existing(norm_to_orig, [normalize_colname(c) for c in candidates])
        if found is not None:
            id_cols.append(found)

    # Extract CPT from any code|N column (not just 2/3)
    df["cpt_code"] = extract_cpt_code(df, norm_to_orig)

    # Drop rows without CPT
    df = df.dropna(subset=["cpt_code"])

    # Keep original code columns too (optional but helpful)
    code_related = [c for c in df.columns if re.fullmatch(r"code\|\d+(\|type)?", normalize_colname(c))]
    id_cols_final = list(dict.fromkeys(id_cols + ["cpt_code"] + code_related))  # preserve order, unique

#this block is for temple code
# ===========================


    payer_col = pick_first_existing(norm_to_orig, ["payer_name", "payer name", "payer"])
    plan_col  = pick_first_existing(norm_to_orig, ["plan_name", "plan name", "plan"])

    if payer_col is not None:
        # Temple-style: payer + plan already in columns
        df_long = df[id_cols_final + [payer_col] + ([plan_col] if plan_col else []) + negotiated_cols].copy()

        df_long = df_long.rename(columns={payer_col: "payer"})
        if plan_col:
            df_long = df_long.rename(columns={plan_col: "plan"})
        else:
            df_long["plan"] = pd.NA

        negotiated_dollar_cols = [
            c for c in negotiated_cols if "negotiated_dollar" in normalize_colname(c)
        ]
        rate_col = negotiated_dollar_cols[0] if negotiated_dollar_cols else negotiated_cols[0]

        df_long = df_long.rename(columns={rate_col: "negotiated_rate"})

        df_long["negotiated_rate"] = pd.to_numeric(df_long["negotiated_rate"], errors="coerce")
        df_long = df_long.dropna(subset=["negotiated_rate"])

    else:
        # UPenn-style: payer encoded in headers
        df_long = df.melt(
            id_vars=id_cols_final,
            value_vars=negotiated_cols,
            var_name="payer_raw",
            value_name="negotiated_rate"
        )

        df_long = df_long.dropna(subset=["negotiated_rate"])
        df_long["negotiated_rate"] = pd.to_numeric(df_long["negotiated_rate"], errors="coerce")
        df_long = df_long.dropna(subset=["negotiated_rate"])

        parts = df_long["payer_raw"].astype(str).str.split("|", expand=True)
        if parts.shape[1] >= 3:
            df_long["payer"] = parts[1]
            df_long["plan"] = parts[2]
        else:
            df_long["payer"] = pd.NA
            df_long["plan"] = pd.NA

        df_long = df_long.drop(columns=["payer_raw"])



# ===========================
    # De-dup (common in these files)
    
    
    dedup_cols = [c for c in ["cpt_code", "payer", "plan", "negotiated_rate"] if c in df_long.columns]
    df_long = df_long.drop_duplicates(subset=dedup_cols)

    df_long.to_csv(output_path, index=False)
    print("Saved long-form file:", output_path)
    print("Rows:", len(df_long), "Unique CPTs:", df_long["cpt_code"].nunique())


if __name__ == "__main__":
    main()

''' 
 #This block doesnt work on Temple code
    # Melt wide -> long (payer columns become rows) :contentReference[oaicite:2]{index=2}
    df_long = df.melt(
        id_vars=id_cols_final,
        value_vars=negotiated_cols,
        var_name="payer_raw",
        value_name="negotiated_rate"
    )

    # Drop empty negotiated rates and coerce numeric :contentReference[oaicite:3]{index=3}
    df_long = df_long.dropna(subset=["negotiated_rate"])
    df_long["negotiated_rate"] = pd.to_numeric(df_long["negotiated_rate"], errors="coerce")
    df_long = df_long.dropna(subset=["negotiated_rate"])

    # Parse payer + plan from header if it follows: standard_charge|<payer>|<plan>|negotiated_dollar :contentReference[oaicite:4]{index=4}
    # If it doesn't match, keep payer_raw and set payer/plan to NA.
    parts = df_long["payer_raw"].astype(str).str.split("|", expand=True)
    if parts.shape[1] >= 3:
        df_long["payer"] = parts[1]
        df_long["plan"] = parts[2]
    else:
        df_long["payer"] = pd.NA
        df_long["plan"] = pd.NA

    # Optional, drop payer_raw once parsed
    # (keep it if you want to debug  headers)
    # df_long = df_long.drop(columns=["payer_raw"])
    '''
    
    






