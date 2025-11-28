import pandas as pd
import hashlib
from sqlalchemy import create_engine, text
import json

# ------------- CONFIG -------------
CSV_PATH = "ETL/dataset_ele_5_cleaned_adjusted.csv"

# change this to your own database URL
DATABASE_URL = "postgresql+psycopg2://postgres:admin@localhost:5432/ELE5"

REQUIRED_COLUMNS = [
    "product_id",
    "original_price",
    "current_price",
    "purchase_date",
    "total_sales",
    "country",
    "countrycode"
]

# ------------- HELPERS -------------

def compute_row_hash(row):
    """
    Create a stable hash of important fields to detect duplicates.
    """
    key_fields = [
        str(row.get("product_id", "")),
        str(row.get("purchase_date", "")),
        str(row.get("current_price", "")),
        str(row.get("total_sales", "")),
        str(row.get("country", "")),
        str(row.get("countrycode", ""))  # must match DF column name
    ]
    raw = "|".join(key_fields)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ------------- 1. EXTRACT -------------

df = pd.read_csv(CSV_PATH)

# 🔧 FIX: normalize the country code column name from CSV
# (your CSV likely has "countryCode" instead of "countrycode")
if "countryCode" in df.columns and "countrycode" not in df.columns:
    df = df.rename(columns={"countryCode": "countrycode"})

# (Optional) quick debug to verify columns
# print("Columns:", df.columns.tolist())

df["__row_number"] = df.index + 1

# ------------- 2. CLEAN / TRANSFORM (to landing) -------------

errors = []

# 2.1 standardize strings
for col in ["category", "brand", "season", "size", "color", "country", "countrycode"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# 2.2 parse date
df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")

# 2.3 convert numeric columns
numeric_cols = [
    "original_price", "markdown_percentage", "current_price",
    "stock_quantity", "total_sales", "customer_rating",
    "latitude", "longitude"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 2.4 derived metrics
df["discount_amount"] = df["original_price"] - df["current_price"]
df["discount_rate"] = df["markdown_percentage"] / 100.0
df["revenue"] = df["current_price"] * df["total_sales"]

# 2.5 missing required fields
mask_missing = df[REQUIRED_COLUMNS].isna().any(axis=1)
for _, row in df[mask_missing].iterrows():
    errors.append({
        "row_number": int(row["__row_number"]),
        "error_stage": "cleaning",
        "error_reason": "Missing required field(s)",
        "raw_payload": row.drop(labels="__row_number").to_dict()
    })
df_clean = df[~mask_missing].copy()

# 2.6 invalid dates
mask_bad_date = df_clean["purchase_date"].isna()
for _, row in df_clean[mask_bad_date].iterrows():
    errors.append({
        "row_number": int(row["__row_number"]),
        "error_stage": "cleaning",
        "error_reason": "Invalid purchase_date",
        "raw_payload": row.drop(labels="__row_number").to_dict()
    })
df_clean = df_clean[~mask_bad_date].copy()

# 2.7 create row_hash
df_clean["row_hash"] = df_clean.apply(compute_row_hash, axis=1)

# 2.8 duplicate detection within CSV batch
dup_mask = df_clean.duplicated(subset=["row_hash"], keep="first")
internal_dup_count = int(dup_mask.sum())

for _, row in df_clean[dup_mask].iterrows():
    errors.append({
        "row_number": int(row["__row_number"]),
        "error_stage": "de-duplication",
        "error_reason": "Duplicate row_hash within batch",
        "raw_payload": row.drop(labels="__row_number").to_dict()
    })

df_clean = df_clean[~dup_mask].copy()
df_landing_batch = df_clean.drop(columns=["__row_number"])

# ------------- 3. LOAD TO LANDING & ERROR TABLES (INCREMENTAL) -------------

engine = create_engine(DATABASE_URL)

# --- 3.1 load existing hashes ---
try:
    df_existing = pd.read_sql("SELECT row_hash FROM landing_product_sales", engine)
    existing_hashes = set(df_existing["row_hash"].tolist())
    print(f"Found {len(existing_hashes)} existing rows in landing_product_sales.")
except Exception:
    existing_hashes = set()
    print("landing_product_sales does not exist yet – will be created.")

# --- 3.2 split new vs existing duplicates ---
is_existing = df_landing_batch["row_hash"].isin(existing_hashes)
df_new_rows = df_landing_batch[~is_existing].copy()
df_dup_vs_landing = df_landing_batch[is_existing].copy()

# log landing duplicates
for _, row in df_dup_vs_landing.iterrows():
    errors.append({
        "row_number": None,
        "error_stage": "landing-duplicate",
        "error_reason": "Row with same hash already in landing_product_sales",
        "raw_payload": row.to_dict()
    })

# --- 3.3 Build error dataframe ---
if errors:
    df_errors = pd.DataFrame(errors)
    df_errors["raw_payload"] = df_errors["raw_payload"].apply(lambda x: json.dumps(x, default=str))
else:
    df_errors = pd.DataFrame(columns=["row_number", "error_stage", "error_reason", "raw_payload"])

# --- 3.4 append new rows into landing ---
df_new_rows.to_sql("landing_product_sales", engine, if_exists="append", index=False)

# --- 3.5 append errors ---
df_errors.to_sql("etl_errors_ele5", engine, if_exists="append", index=False)

print(f"New rows inserted: {len(df_new_rows)}")
print(f"Internal dupes: {internal_dup_count}")
print(f"Landing dupes skipped: {len(df_dup_vs_landing)}")
print(f"Errors logged: {len(df_errors)}")

# -------------------------------------------------------------
# ------------- 4. DROP AND REBUILD DIM/FACT TABLES -----------
# -------------------------------------------------------------

# IMPORTANT: drop fact first, then dims (fixes FK issue)
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS fact_product_sales CASCADE;"))
    conn.execute(text("DROP TABLE IF EXISTS dim_date CASCADE;"))
    conn.execute(text("DROP TABLE IF EXISTS dim_product CASCADE;"))
    conn.execute(text("DROP TABLE IF EXISTS dim_location CASCADE;"))

print("Dropped old dim/fact tables. Rebuilding...")

# ------------- 5. BUILD DIM/FACT FROM FULL LANDING -----------

df_landing_full = pd.read_sql("SELECT * FROM landing_product_sales", engine)

# DIM DATE
dim_date = (
    df_landing_full[["purchase_date"]]
    .drop_duplicates()
    .rename(columns={"purchase_date": "full_date"})
    .sort_values("full_date")
    .reset_index(drop=True)
)
dim_date.insert(0, "date_key", dim_date.index + 1)

# DIM PRODUCT
dim_product = (
    df_landing_full[["product_id", "category", "brand", "season", "size", "color"]]
    .drop_duplicates()
    .sort_values("product_id")
    .reset_index(drop=True)
)
dim_product.insert(0, "product_key", dim_product.index + 1)

# DIM LOCATION
dim_location = (
    df_landing_full[["country", "countrycode", "latitude", "longitude"]]
    .drop_duplicates()
    .sort_values(["country", "countrycode"])
    .reset_index(drop=True)
)
dim_location.insert(0, "location_key", dim_location.index + 1)

# FACT TABLE
fact = df_landing_full.merge(
    dim_product,
    on=["product_id", "category", "brand", "season", "size", "color"],
    how="left"
)

fact = fact.merge(
    dim_date[["date_key", "full_date"]],
    left_on="purchase_date",
    right_on="full_date",
    how="left"
)

fact = fact.merge(
    dim_location,
    on=["country", "countrycode", "latitude", "longitude"],
    how="left"
)

fact_product_sales = fact[[
    "product_key",
    "date_key",
    "location_key",
    "original_price",
    "markdown_percentage",
    "current_price",
    "discount_amount",
    "discount_rate",
    "stock_quantity",
    "total_sales",
    "revenue",
    "customer_rating",
    "is_returned",
    "return_reason"
]]

# write dims & fact
dim_date.to_sql("dim_date", engine, if_exists="replace", index=False)
dim_product.to_sql("dim_product", engine, if_exists="replace", index=False)
dim_location.to_sql("dim_location", engine, if_exists="replace", index=False)
fact_product_sales.to_sql("fact_product_sales", engine, if_exists="replace", index=False)

print("✔ Dimensions and fact table successfully rebuilt.")
print("✔ ETL COMPLETE.")
