"""
Combine hospital pricing files and enrich with:
- hospital_name (from filename)
- total_cases + market_share (from CMS Hospital Service Area file; HHI calculator logic)
- weighted ACS demographics (from ACS API; demographic script logic)

Expected pricing filenames:
  Hospital_pricing_long_TOP10_PAYERS_ONLY.csv
Where Hospital is one of: UPenn, Jefferson, Temple, Cooper
"""

from __future__ import annotations

import os
import glob
import pandas as pd
import requests

# =========================
# ====== EDIT THESE ======
# =========================

# Folder containing the pricing files
PRICING_DIR = r"C:\Users\dckee\OneDrive\Documents\DAT490\price_files"

# CMS Hospital Service Area file (same one used in both scripts)
CMS_HSA_PATH = r"C:\Users\dckee\OneDrive\Documents\DAT490\Hospital_Service_Area_2024.csv"

# Output combined CSV path
OUT_CSV = os.path.join(PRICING_DIR, "ALL_HOSPITALS_pricing_long_TOP10_with_marketshare_demographics.csv")

# ACS settings (from your demographic script)
ACS_YEAR = 2023
CENSUS_API_KEY = "97b91818f7a066db796faf5506722cde3271f8f9"   # API key

# If your pricing files still huge, keep chunking on
USE_CHUNKS = True
CHUNK_SIZE = 250_000

# =========================
# Hospital/provider
# =========================


PROVIDERS = {
    "390111": "UPenn",
    "390174": "Jefferson",
    "390027": "Temple",
    "310017": "Cooper",
}

WEIGHT_COL = "TOTAL_CASES"

# ACS DP profile variables
ACS_VARS = {
    "DP05_0001E": "total_population",
    "DP03_0062E": "median_household_income",
    "DP03_0119PE": "pct_below_poverty",
    "DP05_0071PE": "pct_white_alone",
    "DP05_0072PE": "pct_black_alone",
    "DP05_0074PE": "pct_asian_alone",
    "DP05_0079PE": "pct_hispanic_any_race",
}


# =========================
# Helpers
# =========================

def parse_hospital_from_filename(path: str) -> str:
    """
    Extract hospital_name from filename:
      UPenn_pricing_long_TOP10_PAYERS_ONLY.csv -> UPenn
    """
    base = os.path.basename(path)
    if "_pricing_long_TOP10_PAYERS_ONLY.csv" not in base:
        raise ValueError(f"Unexpected filename format: {base}")
    return base.split("_pricing_long_TOP10_PAYERS_ONLY.csv")[0].strip()


def fetch_acs_all_zctas(year: int, var_codes: list[str], api_key: str = "") -> pd.DataFrame:
    """
    Pull ACS profile variables for all ZCTAs in one request.
    Returns DataFrame: zcta + requested variables
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5/profile"
    params = {"get": ",".join(var_codes), "for": "zip code tabulation area:*"}
    if api_key:
        params["key"] = api_key

    r = requests.get(base, params=params, timeout=120)
    if "application/json" not in r.headers.get("Content-Type", ""):
        print("Census API did not return JSON. First 500 chars:\n")
        print(r.text[:500])
        r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0]).rename(columns={"zip code tabulation area": "zcta"})

    for c in var_codes:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["zcta"] = df["zcta"].astype(str).str.zfill(5)
    return df


def weighted_avg(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    w = group[weight_col]
    x = group[value_col]
    ok = x.notna() & (w > 0)
    if ok.sum() == 0:
        return float("nan")
    return (x[ok] * w[ok]).sum() / w[ok].sum()


def build_hospital_attributes(cms_path: str) -> pd.DataFrame:
    """
    Builds a hospital-level attribute table with:
      hospital_name, total_cases, market_share, and weighted ACS demographics
    """
    # ---- Read CMS HSA ----
    df = pd.read_csv(cms_path, dtype=str, low_memory=False)

    needed = {"MEDICARE_PROV_NUM", "ZIP_CD_OF_RESIDENCE", WEIGHT_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CMS file is missing columns: {sorted(missing)}")

    df["MEDICARE_PROV_NUM"] = df["MEDICARE_PROV_NUM"].astype(str).str.strip()
    df["ZIP_CD_OF_RESIDENCE"] = df["ZIP_CD_OF_RESIDENCE"].astype(str).str.strip().str.zfill(5)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0)

    # Filter to the providers of interest
    df = df[df["MEDICARE_PROV_NUM"].isin(PROVIDERS.keys())].copy()
    df["hospital_name"] = df["MEDICARE_PROV_NUM"].map(PROVIDERS)

    # ---- total_cases + market_share (HHI calculator logic) ----
    totals = df.groupby("hospital_name")[WEIGHT_COL].sum().rename("total_cases")
    shares = (totals / totals.sum()).rename("market_share")  # fraction, e.g. 0.31

    hosp_ms = pd.concat([totals, shares], axis=1).reset_index()

    # Weighted ACS demographics by hospital 
    # Cases by hospital x ZIP
    hz = (
        df.groupby(["hospital_name", "ZIP_CD_OF_RESIDENCE"], as_index=False)[WEIGHT_COL]
          .sum()
          .rename(columns={"ZIP_CD_OF_RESIDENCE": "zcta", WEIGHT_COL: "cases"})
    )

    var_codes = list(ACS_VARS.keys())
    acs = fetch_acs_all_zctas(ACS_YEAR, var_codes, api_key=CENSUS_API_KEY.strip())

    merged = hz.merge(acs[["zcta"] + var_codes], on="zcta", how="left")

    demo_rows = []
    for hosp, g in merged.groupby("hospital_name"):
        out = {"hospital_name": hosp}
        for code, nice in ACS_VARS.items():
            out[nice] = weighted_avg(g, code, "cases")
        demo_rows.append(out)

    demo = pd.DataFrame(demo_rows)

    # formatting 
    for code, nice in ACS_VARS.items():
        if code.endswith("PE") and nice in demo.columns:
            demo[nice] = demo[nice].round(2)
    if "median_household_income" in demo.columns:
        demo["median_household_income"] = demo["median_household_income"].round(0)

    # Final attributes table
    attrs = hosp_ms.merge(demo, on="hospital_name", how="left")

    return attrs


def main():
    # Build hospital-level attributes once
    attrs = build_hospital_attributes(CMS_HSA_PATH)

    # Find pricing files
    pattern = os.path.join(PRICING_DIR, "*_pricing_long_TOP10_PAYERS_ONLY.csv")
    pricing_files = sorted(glob.glob(pattern))
    if not pricing_files:
        raise FileNotFoundError(f"No pricing files found matching: {pattern}")

    # Validate expected hospitals exist 
    found_hospitals = {parse_hospital_from_filename(p) for p in pricing_files}
    expected = set(PROVIDERS.values())
    missing = expected - found_hospitals
    if missing:
        print(f"WARNING: Missing expected hospital pricing files for: {sorted(missing)}")
    extra = found_hospitals - expected
    if extra:
        print(f"WARNING: Found unexpected hospital names in filenames: {sorted(extra)}")

    # Write combined output
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)

    wrote_header = False

    for path in pricing_files:
        hospital_name = parse_hospital_from_filename(path)

        # One-row attribute record for this hospital
        arow = attrs[attrs["hospital_name"] == hospital_name]
        if arow.empty:
            raise ValueError(f"No attributes found for hospital_name={hospital_name}. Check PROVIDERS mapping.")
        # Convert to dict for fast broadcasting to chunks
        attr_dict = arow.iloc[0].to_dict()

        print(f"Processing: {os.path.basename(path)}  -> hospital_name={hospital_name}")

        if USE_CHUNKS:
            for chunk in pd.read_csv(path, low_memory=False, chunksize=CHUNK_SIZE):
                # Add hospital_name
                chunk["hospital_name"] = hospital_name
                # Add hospital level attributes,ame value for every row
                for k, v in attr_dict.items():
                    if k == "hospital_name":
                        continue
                    chunk[k] = v

                chunk.to_csv(OUT_CSV, index=False, mode="a", header=(not wrote_header))
                wrote_header = True
        else:
            dfp = pd.read_csv(path, low_memory=False)
            dfp["hospital_name"] = hospital_name
            for k, v in attr_dict.items():
                if k == "hospital_name":
                    continue
                dfp[k] = v
            dfp.to_csv(OUT_CSV, index=False, mode="a", header=(not wrote_header))
            wrote_header = True

    print(f"\nDone. Combined file saved to:\n{OUT_CSV}")
    print("\nHospital attributes used:")
    print(attrs.to_string(index=False))


if __name__ == "__main__":
    main()

