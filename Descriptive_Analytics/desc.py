# ==========================================================
# COMBINED DESCRIPTIVE ANALYTICS
# - SALES DESCRIPTIVE (existing dashboard logic)
# - PRODUCT DEMAND DESCRIPTIVE (from predictive script)
# Focus: Category Performance, Seasonal Sales, Markdown Impact,
# Average Price by Category, Return Rate, Top 10 Products,
# + Product Demand by SKU, Category, Season
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Ensure UTF-8 output for Windows terminals
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------
# LOAD DATA (shared for sales + product demand)
# ----------------------------------------------------------
df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df = df.sort_values('purchase_date').reset_index(drop=True)

# ----------------------------------------------------------
# BASIC PREP (same as your current SALES code)
# ----------------------------------------------------------
df['has_discount'] = df['markdown_percentage'] > 0
df['revenue_without_discount'] = df['original_price']
df['discount_amount'] = df['original_price'] - df['current_price']

# Map months to seasons (this is what your SALES CSVs currently use)
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# This is the same logic your current desc.py uses for seasons
df['season'] = df['purchase_date'].dt.month.apply(get_season)

# ==========================================================
# PART A: SALES DESCRIPTIVE (your current code)
# ==========================================================

# ----------------------------------------------------------
# 1. TOTAL SALES BY CATEGORY
# ----------------------------------------------------------
category_sales = (
    df.groupby('category')['total_sales']
      .sum()
      .sort_values(ascending=False)
      .reset_index()
)

plt.figure(figsize=(10, 6))
plt.barh(category_sales['category'], category_sales['total_sales'], color='teal')
plt.xlabel('Total Sales ($)')
plt.ylabel('Category')
plt.title('Total Sales by Category')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 2. SEASONAL SALES
# ----------------------------------------------------------
seasonal_sales = (
    df.groupby('season')['total_sales']
      .sum()
      .reindex(['Spring', 'Summer', 'Fall', 'Winter'])
      .reset_index()
)

plt.figure(figsize=(8, 5))
plt.bar(
    seasonal_sales['season'],
    seasonal_sales['total_sales'],
    color=['#90EE90', '#FFD700', '#FF8C00', '#4682B4']
)
plt.title('Sales by Season')
plt.ylabel('Total Sales ($)')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 3. SALES WITH AND WITHOUT MARKDOWN
# ----------------------------------------------------------
markdown_sales = (
    df.groupby('has_discount')['total_sales']
      .sum()
      .reset_index()
)

labels = ['No Discount', 'With Discount']
plt.figure(figsize=(6, 6))
plt.pie(
    markdown_sales['total_sales'],
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#27ae60', '#e74c3c'],
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
plt.title('Sales With vs Without Markdown')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. RETURN RATE (Pie Chart)
# ----------------------------------------------------------
return_analysis = df.groupby('is_returned').agg(
    total_sales=('total_sales', 'sum'),
    transactions=('product_id', 'count')
).reset_index()

labels = ['Not Returned', 'Returned']
plt.figure(figsize=(6, 6))
plt.pie(
    return_analysis['transactions'],
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=['#2ecc71', '#e74c3c'],
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)
plt.title('Return Rate by Transaction Count')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. AVERAGE CURRENT PRICE BY CATEGORY
# ----------------------------------------------------------
avg_price = (
    df.groupby('category')['current_price']
      .mean()
      .sort_values(ascending=False)
      .reset_index()
)

plt.figure(figsize=(10, 6))
plt.barh(avg_price['category'], avg_price['current_price'], color='orange')
plt.xlabel('Average Current Price ($)')
plt.ylabel('Category')
plt.title('Average Current Price by Category')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 6. TOP 10 BEST-SELLING PRODUCTS (SALES SIDE)
# ----------------------------------------------------------
top_products = (
    df.groupby(['product_id', 'category'])['total_sales']
      .sum()
      .sort_values(ascending=False)
      .head(10)
      .reset_index()
)

plt.figure(figsize=(10, 6))
plt.barh(top_products['product_id'], top_products['total_sales'], color='royalblue')
plt.xlabel('Total Sales ($)')
plt.ylabel('Product ID')
plt.title('Top 10 Best-Selling Products (Sales)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ==========================================================
# PART B: PRODUCT DEMAND DESCRIPTIVE (from predictive file)
# ==========================================================

# Make sure optional columns exist (same defensive pattern)
optional_cols = {
    'customer_rating': np.nan,
    'stock_quantity': np.nan,
}
for col, default_val in optional_cols.items():
    if col not in df.columns:
        df[col] = default_val

# ----------------------------------------------------------
# B1. DAILY PRODUCT-LEVEL DEMAND (demand_df)
# ----------------------------------------------------------
demand_df = (
    df.groupby(['purchase_date', 'product_id', 'category'])
      .agg(
          daily_sales=('total_sales', 'sum'),
          num_transactions=('total_sales', 'count'),
          avg_rating=('customer_rating', 'mean'),
          avg_original_price=('original_price', 'mean'),
          avg_current_price=('current_price', 'mean'),
          avg_markdown=('markdown_percentage', 'mean'),
          avg_stock=('stock_quantity', 'mean')
      )
      .reset_index()
)
demand_df = demand_df.rename(columns={'purchase_date': 'date'})

# Attach season (using the same df['season'] we already created above)
season_map = df[['purchase_date', 'season']].drop_duplicates()
season_map = season_map.rename(columns={'purchase_date': 'date'})
demand_df = demand_df.merge(season_map, on='date', how='left')

# Fill some NAs for robustness (same style as predictive)
for col in ['avg_rating', 'avg_original_price', 'avg_current_price', 'avg_markdown', 'avg_stock']:
    if col in demand_df.columns:
        if demand_df[col].notna().any():
            demand_df[col] = demand_df[col].fillna(demand_df[col].median())
        else:
            demand_df[col] = demand_df[col].fillna(0)

# ----------------------------------------------------------
# B2. HISTORICAL PRODUCT DEMAND (Top SKUs)
# ----------------------------------------------------------
hist_product_demand = (
    demand_df.groupby('product_id')
             .agg(
                 total_hist_demand=('daily_sales', 'sum'),
                 avg_daily_demand=('daily_sales', 'mean'),
                 num_days_sold=('daily_sales', lambda x: (x > 0).sum()),
                 category=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
             )
             .reset_index()
)

# Top 10 by historical demand (this feeds product demand dashboard)
top_hist_products = hist_product_demand.sort_values(
    'total_hist_demand', ascending=False
).head(10)

# Plot: Top 10 Historical Best-Selling Products (Product Demand)
plt.figure(figsize=(10, 6))
plt.barh(top_hist_products['product_id'], top_hist_products['total_hist_demand'], color='purple')
plt.xlabel('Total Historical Demand (Sales Value)')
plt.ylabel('Product ID')
plt.title('Top 10 Historical Best-Selling Products (Demand View)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# B3. HISTORICAL DEMAND BY CATEGORY
# ----------------------------------------------------------
cat_hist = (
    demand_df.groupby('category')['daily_sales']
             .sum()
             .reset_index()
             .rename(columns={'daily_sales': 'total_hist_demand'})
)
cat_hist['share_pct'] = cat_hist['total_hist_demand'] / cat_hist['total_hist_demand'].sum() * 100

plt.figure(figsize=(10, 6))
plt.barh(cat_hist['category'], cat_hist['total_hist_demand'], color='darkcyan')
plt.xlabel('Total Historical Demand (Sales Value)')
plt.ylabel('Category')
plt.title('Historical Product Demand by Category')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# B4. HISTORICAL PRODUCT DEMAND BY SEASON
# ----------------------------------------------------------
hist_season_demand = (
    demand_df.groupby('season')['daily_sales']
             .sum()
             .reset_index()
             .rename(columns={'daily_sales': 'total_hist_demand'})
)

# Order seasons
hist_season_demand['season'] = pd.Categorical(
    hist_season_demand['season'],
    categories=['Spring', 'Summer', 'Fall', 'Winter'],
    ordered=True
)
hist_season_demand = hist_season_demand.sort_values('season')

plt.figure(figsize=(8, 5))
plt.plot(
    hist_season_demand['season'],
    hist_season_demand['total_hist_demand'],
    marker='o'
)
plt.title('Historical Product Demand by Season')
plt.ylabel('Total Historical Demand (Sales Value)')
plt.tight_layout()
plt.show()

# ==========================================================
# PART C: VALIDATION – TOTAL SALES VS PRODUCT DEMAND
# ==========================================================
sales_total = df['total_sales'].sum()
product_demand_total = demand_df['daily_sales'].sum()

print("\n" + "=" * 70)
print(f"CHECK — Total Sales from raw df:         {sales_total:,.2f}")
print(f"CHECK — Total Product Demand (daily_sales sum): {product_demand_total:,.2f}")
print(f"DIFFERENCE (should be 0):                {sales_total - product_demand_total:,.4f}")
print("=" * 70 + "\n")

# If DIFFERENCE = 0.0 → your Python side is consistent.
# If Power BI shows a different total for Product Demand,
# the problem is in relationships/measures there, NOT in this code.

# ==========================================================
# PART D: EXPORT FOR POWER BI (separate folders)
# ==========================================================
base_export_dir = os.path.join(os.path.dirname(__file__), "powerbi_exports")
sales_export_dir = os.path.join(base_export_dir, "sales")
product_export_dir = os.path.join(base_export_dir, "product_demand")

os.makedirs(sales_export_dir, exist_ok=True)
os.makedirs(product_export_dir, exist_ok=True)

# ---- SALES CSVs (same logic / filenames, just in /sales) ----
category_sales.to_csv(os.path.join(sales_export_dir, "sales_by_category.csv"), index=False)
seasonal_sales.to_csv(os.path.join(sales_export_dir, "sales_by_season.csv"), index=False)
markdown_sales.to_csv(os.path.join(sales_export_dir, "sales_markdown_comparison.csv"), index=False)
return_analysis.to_csv(os.path.join(sales_export_dir, "return_rate.csv"), index=False)
avg_price.to_csv(os.path.join(sales_export_dir, "avg_current_price_by_category.csv"), index=False)
top_products.to_csv(os.path.join(sales_export_dir, "top10_best_selling_products.csv"), index=False)

# ---- PRODUCT DEMAND CSVs (same structure as predictive descriptive) ----
demand_df.to_csv(os.path.join(product_export_dir, "sku_daily_demand.csv"), index=False)
top_hist_products.to_csv(os.path.join(product_export_dir, "top10_hist_products.csv"), index=False)
cat_hist.to_csv(os.path.join(product_export_dir, "hist_category_demand.csv"), index=False)
hist_season_demand.to_csv(os.path.join(product_export_dir, "hist_seasonal_demand.csv"), index=False)

print("✅ CSV files exported successfully.")
print("Sales CSVs in:", sales_export_dir)
for f in os.listdir(sales_export_dir):
    print(" -", f)
print("\nProduct Demand CSVs in:", product_export_dir)
for f in os.listdir(product_export_dir):
    print(" -", f)
