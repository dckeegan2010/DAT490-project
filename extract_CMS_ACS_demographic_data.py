import pandas as pd
import requests

# ========================
# ====== EDIT THESE ======
# ========================
cms_path = r"C:\Users\dckee\OneDrive\Documents\DAT490\Hospital_Service_Area_2024.csv"

# Using American Community Survey API (ACS) 

ACS_YEAR = 2023  
CENSUS_API_KEY = "97b91818f7a066db796faf5506722cde3271f8f9"  


providers = {
    "390111": "Upenn",
    "390174": "Jefferson",
    "390027": "Temple",
    "310017": "Cooper",
}



# Use DP (Data Profile) variables (percent variables end in PE) from ACS
ACS_VARS = {
    "DP05_0001E": "total_population",
    "DP03_0062E": "median_household_income",
    "DP03_0119PE": "pct_below_poverty",
    "DP05_0071PE": "pct_white_alone",
    "DP05_0072PE": "pct_black_alone",
    "DP05_0074PE": "pct_asian_alone",
    "DP05_0079PE": "pct_hispanic_any_race",
}

WEIGHT_COL = "TOTAL_CASES"


def fetch_acs_all_zctas(year, var_codes, api_key=""):
    """
    Pull ACS profile variables for all zip code tabulation areas (ZCTAs) in one request.
    Returns a DataFrame with 'zcta' plus requested variables
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5/profile"
    params = {
        "get": ",".join(var_codes),
        "for": "zip code tabulation area:*",
    }
    if api_key:
        params["key"] = api_key

    r = requests.get(base, params=params, timeout=120)

    # If it isn't JSON, print the first chunk to show the real error message
    if "application/json" not in r.headers.get("Content-Type", ""):
        print("Census API did not return JSON. First 500 chars of response:\n")
        print(r.text[:500])
        r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data[1:], columns=data[0])

    # Rename ZCTA field to 'zcta'
    df = df.rename(columns={"zip code tabulation area": "zcta"})

    # Convert numeric vars
    for c in var_codes:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Ensure zcta is 5-digit string
    df["zcta"] = df["zcta"].astype(str).str.zfill(5)

    return df


def weighted_avg(group, value_col, weight_col="cases"):
    w = group[weight_col]
    x = group[value_col]
    ok = x.notna() & (w > 0)
    if ok.sum() == 0:
        return float("nan")
    return (x[ok] * w[ok]).sum() / w[ok].sum()


def main():
    # Read CMS hospital service area data, downloaded from 2024
    df = pd.read_csv(cms_path, dtype=str, low_memory=False)
    needed = {"MEDICARE_PROV_NUM", "ZIP_CD_OF_RESIDENCE", WEIGHT_COL}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["MEDICARE_PROV_NUM"] = df["MEDICARE_PROV_NUM"].astype(str).str.strip()
    df["ZIP_CD_OF_RESIDENCE"] = df["ZIP_CD_OF_RESIDENCE"].astype(str).str.strip().str.zfill(5)
    df[WEIGHT_COL] = pd.to_numeric(df[WEIGHT_COL], errors="coerce").fillna(0)

    # Filter to the four hopitals of interest
    df = df[df["MEDICARE_PROV_NUM"].isin(providers.keys())].copy()
    df["hospital"] = df["MEDICARE_PROV_NUM"].map(providers)

    # Cases by hospital x ZIP
    hz = (
        df.groupby(["hospital", "ZIP_CD_OF_RESIDENCE"], as_index=False)[WEIGHT_COL]
        .sum()
        .rename(columns={"ZIP_CD_OF_RESIDENCE": "zcta", WEIGHT_COL: "cases"})
    )

    # Pull ACS for all ZCTAs
    var_codes = list(ACS_VARS.keys())
    acs = fetch_acs_all_zctas(ACS_YEAR, var_codes, api_key=CENSUS_API_KEY.strip())

    # Join service area ZIP weights to ACS
    merged = hz.merge(acs[["zcta"] + var_codes], on="zcta", how="left")

    # Weighted demographics per hospital
    rows = []
    for hosp, g in merged.groupby("hospital"):
        out = {
            "hospital": hosp,
            "total_cases": g["cases"].sum(),
            "unique_zctas": g["zcta"].nunique(),
        }
        for code, nice in ACS_VARS.items():
            out[nice] = weighted_avg(g, code, "cases")
        rows.append(out)

    out_df = pd.DataFrame(rows).sort_values("total_cases", ascending=False)

    # make the output look nice
    for code, nice in ACS_VARS.items():
        if code.endswith("PE"):
            out_df[nice] = out_df[nice].round(2)
    if "median_household_income" in out_df.columns:
        out_df["median_household_income"] = out_df["median_household_income"].round(0)

    print("\nWeighted ACS demographics by hospital (weights = CMS TOTAL_CASES by ZIP/ZCTA):\n")
    print(out_df.to_string(index=False))

    # Save output
    out_path = cms_path.replace(".csv", f"_weighted_acs_{ACS_YEAR}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
