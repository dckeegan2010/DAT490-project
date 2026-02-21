import pandas as pd

# set file path 
file_path = r"C:\Users\dckee\OneDrive\Documents\DAT490\Hospital_Service_Area_2024.csv"

# ====== 
# PROVIDER NUMBERS 
# ======
providers = {
    "390111": "Upenn",
    "390174": "Jefferson",
    "390027": "Temple",
  # "390195": "Mainline Lankenau",
    "310017": "Cooper"
}

# read files
df = pd.read_csv(file_path, dtype=str)

# make numeric
df["TOTAL_CASES"] = pd.to_numeric(df["TOTAL_CASES"], errors="coerce").fillna(0)

# filter
df4 = df[df["MEDICARE_PROV_NUM"].isin(providers.keys())].copy()
df4["hospital"] = df4["MEDICARE_PROV_NUM"].map(providers)

# sum by case hospital
totals = df4.groupby("hospital")["TOTAL_CASES"].sum()

# shares
shares = totals / totals.sum()

# COMPUTE HHI
hhi = (shares * 100).pow(2).sum()

# print results
print("\nCases by hospital:")
print(totals)

print("\nMarket shares:")
print((shares * 100).round(2).astype(str) + "%")

print(f"\nHHI = {hhi:.1f}  (0–10,000 scale)")

