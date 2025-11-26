import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import matplotlib.ticker as ticker

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# XGBoost (main model for demand forecasting)
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost is not installed. Please run: pip install xgboost")


# =============================================================================
# PART 0: DATA INGESTION
# =============================================================================

df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")

df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df = df.sort_values('purchase_date').reset_index(drop=True)

# Ensure optional columns exist
optional_cols = {
    'customer_rating': np.nan,
    'original_price': np.nan,
    'current_price': np.nan,
    'markdown_percentage': 0.0,
    'stock_quantity': np.nan
}
for col, default_val in optional_cols.items():
    if col not in df.columns:
        df[col] = default_val

print("=" * 100)
print(" " * 20 + "SKU-LEVEL & WEEKLY CATEGORY-LEVEL DEMAND FORECASTING")
print("=" * 100)
print(f"\nDataset: {len(df):,} transactions")
print(f"Date Range: {df['purchase_date'].min().strftime('%Y-%m-%d')} "
      f"to {df['purchase_date'].max().strftime('%Y-%m-%d')}")
print(f"Total Sales: ${df['total_sales'].sum():,.2f}")
print(f"Unique Products: {df['product_id'].nunique():,}")
print(f"Categories: {df['category'].nunique():,}")

# =============================================================================
# PART 1: DAILY PRODUCT-LEVEL DATASET (SKU) + DESCRIPTIVE
# =============================================================================

print("\n" + "=" * 100)
print("PART 1: DAILY PRODUCT DEMAND (SKU-LEVEL) + DESCRIPTIVE")
print("=" * 100)

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

demand_df.rename(columns={'purchase_date': 'date'}, inplace=True)

# Attach season information from original df if available
if 'season' in df.columns:
    season_map = df[['purchase_date', 'season']].drop_duplicates()
    season_map = season_map.rename(columns={'purchase_date': 'date'})
    demand_df = demand_df.merge(season_map, on='date', how='left')
else:
    demand_df['season'] = np.nan

for col in ['avg_rating', 'avg_markdown', 'avg_stock']:
    demand_df[col] = demand_df[col].fillna(demand_df[col].median())

print(f"\nDaily product-level rows: {len(demand_df):,}")
print(f"Unique product_ids in demand dataset: {demand_df['product_id'].nunique():,}")

# ---- Historical best-selling products (SKU) ----
hist_product_demand = (
    demand_df.groupby('product_id')
             .agg(
                 total_hist_demand=('daily_sales', 'sum'),
                 avg_daily_demand=('daily_sales', 'mean'),
                 num_days_sold=('daily_sales', lambda x: (x > 0).sum()),
                 category=('category', lambda x: x.mode().iloc[0]
                           if not x.mode().empty else x.iloc[0])
             )
             .reset_index()
)

top_hist_n = 10
top_hist_products = hist_product_demand.sort_values(
    'total_hist_demand', ascending=False
).head(top_hist_n)

print(f"\nTOP {top_hist_n} HISTORICAL BEST-SELLING PRODUCTS (SKU):")
print("-" * 100)
print(top_hist_products[['product_id', 'category',
                         'total_hist_demand', 'avg_daily_demand',
                         'num_days_sold']])

# ---- Historical demand by category (for dashboard) ----
cat_hist = (
    demand_df.groupby('category')['daily_sales']
             .sum()
             .reset_index()
             .rename(columns={'daily_sales': 'total_hist_demand'})
)
cat_hist['share_pct'] = (cat_hist['total_hist_demand']
                         / cat_hist['total_hist_demand'].sum()) * 100

print("\nHISTORICAL DEMAND BY CATEGORY:")
print("-" * 100)
print(cat_hist.sort_values('total_hist_demand', ascending=False))

# ---- Historical demand by season (product demand) ----
if 'season' in demand_df.columns and demand_df['season'].notna().any():
    hist_season_demand = (
        demand_df.groupby('season')['daily_sales']
                 .sum()
                 .reset_index()
                 .rename(columns={'daily_sales': 'total_hist_demand'})
                 .sort_values('total_hist_demand', ascending=False)
    )
    print("\nHISTORICAL PRODUCT DEMAND BY SEASON:")
    print("-" * 100)
    print(hist_season_demand)
else:
    hist_season_demand = pd.DataFrame(columns=['season', 'total_hist_demand'])

min_date = demand_df['date'].min()

# Prepare month -> season mapping (for future dates)
if 'season' in df.columns:
    month_to_season = (
        df.assign(month=df['purchase_date'].dt.month)
          .groupby('month')['season']
          .agg(lambda x: x.mode().iloc[0])
          .to_dict()
    )
else:
    month_to_season = {}

# =============================================================================
# PART 2: FEATURE ENGINEERING & MODELING (SKU-LEVEL FORECAST, DAILY)
# =============================================================================

print("\n" + "=" * 100)
print("PART 2: SKU-LEVEL FEATURE ENGINEERING & MODELING (DAILY)")
print("=" * 100)

# Encodings
demand_df['product_code'], product_unique = pd.factorize(demand_df['product_id'])
demand_df['category_code'], category_unique = pd.factorize(demand_df['category'])

# Time features
demand_df['year'] = demand_df['date'].dt.year
demand_df['month'] = demand_df['date'].dt.month
demand_df['day'] = demand_df['date'].dt.day
demand_df['day_of_week'] = demand_df['date'].dt.dayofweek
demand_df['day_of_year'] = demand_df['date'].dt.dayofyear
demand_df['week_of_year'] = demand_df['date'].dt.isocalendar().week.astype(int)
demand_df['quarter'] = demand_df['date'].dt.quarter
demand_df['is_weekend'] = demand_df['day_of_week'].isin([5, 6]).astype(int)

demand_df['month_sin'] = np.sin(2 * np.pi * demand_df['month'] / 12)
demand_df['month_cos'] = np.cos(2 * np.pi * demand_df['month'] / 12)
demand_df['dow_sin'] = np.sin(2 * np.pi * demand_df['day_of_week'] / 7)
demand_df['dow_cos'] = np.cos(2 * np.pi * demand_df['day_of_week'] / 7)

demand_df['global_trend'] = (demand_df['date'] - min_date).dt.days

sku_target_col = 'daily_sales'
sku_feature_cols = [
    'product_code', 'category_code',
    'avg_original_price', 'avg_current_price', 'avg_markdown',
    'avg_stock', 'avg_rating',
    'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_weekend',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'global_trend'
]

demand_df = demand_df.sort_values('date').reset_index(drop=True)
X_sku = demand_df[sku_feature_cols]
y_sku = demand_df[sku_target_col]
dates_sku = demand_df['date']

split_idx_sku = int(len(demand_df) * 0.8)
X_sku_train, X_sku_test = X_sku.iloc[:split_idx_sku], X_sku.iloc[split_idx_sku:]
y_sku_train, y_sku_test = y_sku.iloc[:split_idx_sku], y_sku.iloc[split_idx_sku:]
dates_sku_train, dates_sku_test = dates_sku.iloc[:split_idx_sku], dates_sku.iloc[split_idx_sku:]

print(f"\nTraining Period (SKU): {dates_sku_train.min().strftime('%Y-%m-%d')} "
      f"to {dates_sku_train.max().strftime('%Y-%m-%d')}")
print(f"Testing Period  (SKU): {dates_sku_test.min().strftime('%Y-%m-%d')} "
      f"to {dates_sku_test.max().strftime('%Y-%m-%d')}")
print(f"Train samples (SKU): {len(X_sku_train):,}, Test samples (SKU): {len(X_sku_test):,}")

scaler_sku = StandardScaler()
X_sku_train_scaled = scaler_sku.fit_transform(X_sku_train)
X_sku_test_scaled = scaler_sku.transform(X_sku_test)

# ---- Train SKU-level models ----
print("\nTraining XGBoost Regressor (MAIN MODEL, SKU-LEVEL)...")
xgb_sku_model = XGBRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_sku_model.fit(X_sku_train, y_sku_train)

print("Training Random Forest Regressor (BENCHMARK 1, SKU-LEVEL)...")
rf_sku_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_sku_model.fit(X_sku_train, y_sku_train)

print("Training Linear Regression (BENCHMARK 2, SKU-LEVEL)...")
lin_sku_model = LinearRegression()
lin_sku_model.fit(X_sku_train_scaled, y_sku_train)

def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100  # +1 avoids div-by-zero
    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape}

print("\n" + "=" * 100)
print("SKU-LEVEL MODEL EVALUATION (DAILY)")
print("=" * 100)

y_sku_pred_xgb = xgb_sku_model.predict(X_sku_test)
y_sku_pred_rf = rf_sku_model.predict(X_sku_test)
y_sku_pred_lin = lin_sku_model.predict(X_sku_test_scaled)

metrics_xgb_sku = evaluate_model("XGBoost (MAIN MODEL, SKU)", y_sku_test, y_sku_pred_xgb)
metrics_rf_sku = evaluate_model("Random Forest (BENCHMARK 1, SKU)", y_sku_test, y_sku_pred_rf)
metrics_lin_sku = evaluate_model("Linear Regression (BENCHMARK 2, SKU)", y_sku_test, y_sku_pred_lin)

# =============================================================================
# PART 3: SKU-LEVEL FUTURE DEMAND FORECAST (NEXT 180 DAYS)
#       + SEASONAL FORECAST (PRODUCT DEMAND)
# =============================================================================

print("\n" + "=" * 100)
print("PART 3: SKU-LEVEL FUTURE DEMAND FORECAST (NEXT 180 DAYS)")
print("=" * 100)

last_date_sku = demand_df['date'].max()
forecast_horizon_days = 180

# 3.1 Create future dates with time features (vectorized)
future_dates = pd.date_range(
    start=last_date_sku + timedelta(days=1),
    periods=forecast_horizon_days,
    freq='D'
)

future_dates_df = pd.DataFrame({'date': future_dates})
future_dates_df['month'] = future_dates_df['date'].dt.month
future_dates_df['day_of_week'] = future_dates_df['date'].dt.dayofweek
future_dates_df['day_of_year'] = future_dates_df['date'].dt.dayofyear
future_dates_df['week_of_year'] = future_dates_df['date'].dt.isocalendar().week.astype(int)
future_dates_df['quarter'] = future_dates_df['month'].apply(lambda m: (m - 1) // 3 + 1)
future_dates_df['is_weekend'] = future_dates_df['day_of_week'].isin([5, 6]).astype(int)

future_dates_df['month_sin'] = np.sin(2 * np.pi * future_dates_df['month'] / 12)
future_dates_df['month_cos'] = np.cos(2 * np.pi * future_dates_df['month'] / 12)
future_dates_df['dow_sin'] = np.sin(2 * np.pi * future_dates_df['day_of_week'] / 7)
future_dates_df['dow_cos'] = np.cos(2 * np.pi * future_dates_df['day_of_week'] / 7)

future_dates_df['global_trend'] = (future_dates_df['date'] - min_date).dt.days

# Map month -> season (if mapping exists)
if month_to_season:
    future_dates_df['season'] = future_dates_df['month'].map(
        lambda m: month_to_season.get(m, "Unknown")
    )
else:
    future_dates_df['season'] = np.nan

# 3.2 Get last known product-level features per SKU (vectorized)
last_product_info = (
    demand_df.sort_values('date')
             .groupby('product_id')
             .agg(
                 product_code=('product_code', 'last'),
                 category_code=('category_code', 'last'),
                 avg_original_price=('avg_original_price', 'last'),
                 avg_current_price=('avg_current_price', 'last'),
                 avg_markdown=('avg_markdown', 'last'),
                 avg_stock=('avg_stock', 'last'),
                 avg_rating=('avg_rating', 'last')
             )
             .reset_index()
)

# 3.3 Cross-join dates x products using a key column
future_dates_df['key'] = 1
last_product_info['key'] = 1

future_df = future_dates_df.merge(last_product_info, on='key', how='outer').drop(columns='key')

# 3.4 Predict SKU-level demand for all date–product pairs
X_sku_future = future_df[sku_feature_cols]
future_df['predicted_demand'] = xgb_sku_model.predict(X_sku_future)

# 3.5 Aggregate to product-level totals
product_forecast = (
    future_df.groupby('product_id')
             .agg(
                 total_forecast_demand=('predicted_demand', 'sum'),
                 avg_daily_demand=('predicted_demand', 'mean'),
                 first_category_code=('category_code', 'first')
             )
             .reset_index()
)

product_forecast['category'] = product_forecast['first_category_code'].apply(
    lambda c: category_unique[c] if 0 <= c < len(category_unique) else "Unknown"
)
product_forecast = product_forecast.sort_values('total_forecast_demand', ascending=False)

top_n = 10
top_products_forecast = product_forecast.head(top_n)

print(f"\nTOP {top_n} PREDICTED IN-DEMAND PRODUCTS (NEXT {forecast_horizon_days} DAYS):")
print("-" * 100)
for _, row in top_products_forecast.iterrows():
    print(f"{row['product_id']:>10s} | {row['category']:<15s} | "
          f"Total Demand: {row['total_forecast_demand']:.2f} | "
          f"Avg Daily: {row['avg_daily_demand']:.2f}")

# 3.6 Seasonal forecast of product demand (next 6 months)
if 'season' in future_df.columns and future_df['season'].notna().any():
    seasonal_forecast = (
        future_df.groupby('season')['predicted_demand']
                 .sum()
                 .reset_index()
                 .rename(columns={'predicted_demand': 'total_forecast_demand'})
                 .sort_values('total_forecast_demand', ascending=False)
    )
    print(f"\nFORECASTED PRODUCT DEMAND BY SEASON (NEXT {forecast_horizon_days} DAYS):")
    print("-" * 100)
    print(seasonal_forecast)
else:
    seasonal_forecast = pd.DataFrame(columns=['season', 'total_forecast_demand'])

# =============================================================================
# PART 4: WEEKLY CATEGORY-LEVEL DATASET & MODELING
# =============================================================================

print("\n" + "=" * 100)
print("PART 4: WEEKLY CATEGORY-LEVEL DATASET & MODELING")
print("=" * 100)

# Build daily category dataset from demand_df
cat_daily_df = (
    demand_df.groupby(['date', 'category'])
             .agg(daily_sales=('daily_sales', 'sum'))
             .reset_index()
)

cat_daily_df['category_code'], category_codes_unique = pd.factorize(cat_daily_df['category'])
cat_codes = cat_daily_df[['category', 'category_code']].drop_duplicates()

# Weekly aggregation
cat_weekly_df = (
    cat_daily_df
      .set_index('date')
      .groupby('category')['daily_sales']
      .resample('W-MON')
      .sum()
      .reset_index()
      .rename(columns={'daily_sales': 'weekly_sales'})
)

cat_weekly_df = cat_weekly_df.merge(cat_codes, on='category', how='left')

# Weekly time features
cat_weekly_df['year'] = cat_weekly_df['date'].dt.year
cat_weekly_df['week_of_year'] = cat_weekly_df['date'].dt.isocalendar().week.astype(int)
cat_weekly_df['month'] = cat_weekly_df['date'].dt.month
cat_weekly_df['quarter'] = cat_weekly_df['date'].dt.quarter
cat_weekly_df['week_of_year_sin'] = np.sin(2 * np.pi * cat_weekly_df['week_of_year'] / 52)
cat_weekly_df['week_of_year_cos'] = np.cos(2 * np.pi * cat_weekly_df['week_of_year'] / 52)
cat_weekly_df['global_trend'] = (cat_weekly_df['date'] - min_date).dt.days

weekly_target_col = 'weekly_sales'
weekly_feature_cols = [
    'category_code',
    'year', 'month', 'week_of_year', 'quarter',
    'week_of_year_sin', 'week_of_year_cos',
    'global_trend'
]

cat_weekly_df = cat_weekly_df.sort_values('date').reset_index(drop=True)
X_week = cat_weekly_df[weekly_feature_cols]
y_week = cat_weekly_df[weekly_target_col]
dates_week = cat_weekly_df['date']

split_idx_week = int(len(cat_weekly_df) * 0.8)
X_week_train, X_week_test = X_week.iloc[:split_idx_week], X_week.iloc[split_idx_week:]
y_week_train, y_week_test = y_week.iloc[:split_idx_week], y_week.iloc[split_idx_week:]
dates_week_train, dates_week_test = dates_week.iloc[:split_idx_week], dates_week.iloc[split_idx_week:]

print(f"\nTraining Period (CATEGORY WEEKLY): {dates_week_train.min().strftime('%Y-%m-%d')} "
      f"to {dates_week_train.max().strftime('%Y-%m-%d')}")
print(f"Testing Period  (CATEGORY WEEKLY): {dates_week_test.min().strftime('%Y-%m-%d')} "
      f"to {dates_week_test.max().strftime('%Y-%m-%d')}")
print(f"Train samples (CATEGORY WEEKLY): {len(X_week_train):,}, "
      f"Test samples (CATEGORY WEEKLY): {len(X_week_test):,}")

scaler_week = StandardScaler()
X_week_train_scaled = scaler_week.fit_transform(X_week_train)
X_week_test_scaled = scaler_week.transform(X_week_test)

print("\nTraining XGBoost Regressor (MAIN MODEL, CATEGORY WEEKLY)...")
xgb_cat_week_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_cat_week_model.fit(X_week_train, y_week_train)

print("Training Random Forest Regressor (BENCHMARK 1, CATEGORY WEEKLY)...")
rf_cat_week_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_cat_week_model.fit(X_week_train, y_week_train)

print("Training Linear Regression (BENCHMARK 2, CATEGORY WEEKLY)...")
lin_cat_week_model = LinearRegression()
lin_cat_week_model.fit(X_week_train_scaled, y_week_train)


def compute_wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-8) * 100


def evaluate_weekly_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    wape = compute_wape(y_true, y_pred)
    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    print(f"WAPE:  {wape:.2f}%")
    return {'name': name, 'mae': mae, 'rmse': rmse, 'mape': mape, 'wape': wape}


print("\n" + "=" * 100)
print("CATEGORY-LEVEL MODEL EVALUATION (WEEKLY)")
print("=" * 100)

y_week_pred_xgb = xgb_cat_week_model.predict(X_week_test)
y_week_pred_rf = rf_cat_week_model.predict(X_week_test)
y_week_pred_lin = lin_cat_week_model.predict(X_week_test_scaled)

metrics_xgb_week = evaluate_weekly_model(
    "XGBoost (MAIN MODEL, CATEGORY WEEKLY)", y_week_test, y_week_pred_xgb
)
metrics_rf_week = evaluate_weekly_model(
    "Random Forest (BENCHMARK 1, CATEGORY WEEKLY)", y_week_test, y_week_pred_rf
)
metrics_lin_week = evaluate_weekly_model(
    "Linear Regression (BENCHMARK 2, CATEGORY WEEKLY)", y_week_test, y_week_pred_lin
)

# =============================================================================
# PART 5: FUTURE WEEKLY CATEGORY FORECAST (NEXT 26 WEEKS)
# =============================================================================

print("\n" + "=" * 100)
print("PART 5: FUTURE WEEKLY CATEGORY FORECAST (NEXT 26 WEEKS)")
print("=" * 100)

forecast_weeks = 26
last_week_date = cat_weekly_df['date'].max()
future_week_starts = pd.date_range(
    start=last_week_date + pd.Timedelta(weeks=1),
    periods=forecast_weeks,
    freq='W-MON'
)

cat_week_list = cat_weekly_df[['category', 'category_code']].drop_duplicates()

future_week_rows = []
for d in future_week_starts:
    year = d.year
    month = d.month
    week_of_year = d.isocalendar().week
    quarter = (month - 1) // 3 + 1
    week_sin = np.sin(2 * np.pi * week_of_year / 52)
    week_cos = np.cos(2 * np.pi * week_of_year / 52)
    global_trend = (d - min_date).days

    for _, row in cat_week_list.iterrows():
        cat_name = row['category']
        cat_code = row['category_code']
        future_week_rows.append({
            'date': d,
            'category': cat_name,
            'category_code': cat_code,
            'year': year,
            'month': month,
            'week_of_year': week_of_year,
            'quarter': quarter,
            'week_of_year_sin': week_sin,
            'week_of_year_cos': week_cos,
            'global_trend': global_trend
        })

future_week_df = pd.DataFrame(future_week_rows)
X_week_future = future_week_df[weekly_feature_cols]
future_week_df['predicted_weekly_demand'] = xgb_cat_week_model.predict(X_week_future)

cat_forecast_weekly = (
    future_week_df.groupby('category')
                  .agg(
                      total_forecast_demand=('predicted_weekly_demand', 'sum'),
                      avg_weekly_demand=('predicted_weekly_demand', 'mean')
                  )
                  .reset_index()
                  .sort_values('total_forecast_demand', ascending=False)
)

print(f"\nFORECASTED CATEGORY DEMAND (NEXT {forecast_weeks} WEEKS):")
print("-" * 100)
print(cat_forecast_weekly)

# =============================================================================
# PART 5.5: EXPORT TABLES FOR POWER BI
# =============================================================================

# Create a subfolder for Power BI exports (beside this .py file)
base_dir = os.path.dirname(__file__)
export_dir = os.path.join(base_dir, "powerbi_exports")
os.makedirs(export_dir, exist_ok=True)

print("\nSaving Power BI export CSVs to:", export_dir)

# 1) SKU-level daily demand (historical)
#    One row per product per day
sku_daily_path = os.path.join(export_dir, "sku_daily_demand.csv")
demand_df.to_csv(sku_daily_path, index=False)
print(" - Saved SKU daily demand to", sku_daily_path)

# 2) Top 10 historical best-selling products (SKU)
top_hist_path = os.path.join(export_dir, "top10_hist_products.csv")
top_hist_products.to_csv(top_hist_path, index=False)
print(" - Saved Top 10 historical products to", top_hist_path)

# 3) Historical demand by category
cat_hist_path = os.path.join(export_dir, "hist_category_demand.csv")
cat_hist.to_csv(cat_hist_path, index=False)
print(" - Saved historical category demand to", cat_hist_path)

# 4) Historical product demand by season (if available)
if not hist_season_demand.empty:
    hist_season_path = os.path.join(export_dir, "hist_seasonal_demand.csv")
    hist_season_demand.to_csv(hist_season_path, index=False)
    print(" - Saved historical seasonal demand to", hist_season_path)

# 5) Full SKU-level future forecast (next 180 days)
#    One row per product per forecast date
sku_future_path = os.path.join(export_dir, "sku_future_forecast_180d.csv")
future_df.to_csv(sku_future_path, index=False)
print(" - Saved SKU future forecast (180 days) to", sku_future_path)

# 6) Top 10 forecasted in-demand products
top_forecast_path = os.path.join(export_dir, "top10_sku_future_forecast.csv")
top_products_forecast.to_csv(top_forecast_path, index=False)
print(" - Saved Top 10 future in-demand products to", top_forecast_path)

# 7) Weekly category-level historical data
cat_weekly_hist_path = os.path.join(export_dir, "category_weekly_hist_demand.csv")
cat_weekly_df.to_csv(cat_weekly_hist_path, index=False)
print(" - Saved weekly category historical demand to", cat_weekly_hist_path)

# 8) Weekly category-level future forecast (next 26 weeks)
cat_weekly_future_path = os.path.join(export_dir, "category_weekly_future_forecast_26w.csv")
future_week_df.to_csv(cat_weekly_future_path, index=False)
print(" - Saved weekly category future forecast (26 weeks) to", cat_weekly_future_path)

# 9) Aggregated category forecast (total over next 26 weeks)
cat_forecast_totals_path = os.path.join(export_dir, "category_forecast_totals_26w.csv")
cat_forecast_weekly.to_csv(cat_forecast_totals_path, index=False)
print(" - Saved category forecast totals to", cat_forecast_totals_path)

# 10) Seasonal forecast for product demand (next 180 days)
if not seasonal_forecast.empty:
    seasonal_forecast_path = os.path.join(export_dir, "seasonal_forecast_180d.csv")
    seasonal_forecast.to_csv(seasonal_forecast_path, index=False)
    print(" - Saved seasonal product forecast to", seasonal_forecast_path)

# 11) Model metrics table (SKU + weekly category)
# metrics_xxx_* are dicts created earlier in the script
metrics_rows = [
    ["XGBoost_SKU_Daily",          metrics_xgb_sku["mae"],  metrics_xgb_sku["rmse"],  metrics_xgb_sku["mape"],  None],
    ["RandomForest_SKU_Daily",     metrics_rf_sku["mae"],   metrics_rf_sku["rmse"],   metrics_rf_sku["mape"],   None],
    ["LinearReg_SKU_Daily",        metrics_lin_sku["mae"],  metrics_lin_sku["rmse"],  metrics_lin_sku["mape"],  None],
    ["XGBoost_Category_Weekly",    metrics_xgb_week["mae"], metrics_xgb_week["rmse"], metrics_xgb_week["mape"], metrics_xgb_week["wape"]],
    ["RandomForest_Category_Weekly", metrics_rf_week["mae"], metrics_rf_week["rmse"], metrics_rf_week["mape"], metrics_rf_week["wape"]],
    ["LinearReg_Category_Weekly",  metrics_lin_week["mae"], metrics_lin_week["rmse"], metrics_lin_week["mape"], metrics_lin_week["wape"]],
]

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["model", "mae", "rmse", "mape", "wape"]
)

metrics_path = os.path.join(export_dir, "model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)
print(" - Saved model metrics to", metrics_path)

# =============================================================================
# PART 6: VISUALIZATIONS (SEPARATE FIGURES)
# =============================================================================

print("\n" + "=" * 100)
print("PART 6: VISUALIZATIONS (SEPARATE CHARTS)")
print("=" * 100)

base_dir = os.path.dirname(__file__)

# 1) Top 10 Historical Best-Selling Products (SKU) – horizontal bar
fig1, ax1 = plt.subplots(figsize=(10, 6))
top_hist_plot = top_hist_products.sort_values('total_hist_demand', ascending=True)
ax1.barh(top_hist_plot['product_id'], top_hist_plot['total_hist_demand'])
ax1.set_title(f"Top {top_hist_n} Historical Best-Selling Products (SKU)")
ax1.set_xlabel("Total Historical Demand")
ax1.set_ylabel("Product ID")
ax1.grid(axis='x', alpha=0.3)
plt.tight_layout()
out1 = os.path.join(base_dir, "viz1_top_hist_sku.png")
plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out1}")

# 2) Historical Demand by Category – vertical bar
fig2, ax2 = plt.subplots(figsize=(10, 6))
cat_hist_plot = cat_hist.sort_values('total_hist_demand', ascending=False)
ax2.bar(cat_hist_plot['category'], cat_hist_plot['total_hist_demand'])
ax2.set_title("Historical Demand by Category")
ax2.set_xlabel("Category")
ax2.set_ylabel("Total Historical Demand")
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
out2 = os.path.join(base_dir, "viz2_hist_demand_by_category.png")
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out2}")

# 3) Top 10 Predicted In-Demand Products (Next 180 Days) – standard bar chart
fig3, ax3 = plt.subplots(figsize=(10, 6))
future_top_plot = top_products_forecast.sort_values('total_forecast_demand', ascending=False)
ax3.bar(future_top_plot['product_id'], future_top_plot['total_forecast_demand'])
ax3.set_title(f"Top {top_n} Forecasted In-Demand Products (Next {forecast_horizon_days} Days)")
ax3.set_xlabel("Product ID")
ax3.set_ylabel("Forecast Total Demand")
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
out3 = os.path.join(base_dir, "viz3_top_forecast_sku.png")
plt.savefig(out3, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out3}")

# 4A) Forecasted Weekly Demand by Category (Next 26 Weeks) – COMBINED MULTI-LINE
weekly_pivot = future_week_df.pivot_table(
    index='date',
    columns='category',
    values='predicted_weekly_demand',
    aggfunc='sum'
).sort_index()

fig4a, ax4a = plt.subplots(figsize=(10, 6))
for cat in weekly_pivot.columns:
    ax4a.plot(weekly_pivot.index, weekly_pivot[cat], label=cat)

ax4a.set_title(f"Forecasted Weekly Demand by Category (Next {forecast_weeks} Weeks) – Combined")
ax4a.set_xlabel("Week (Date)")
ax4a.set_ylabel("Weekly Demand")
ax4a.grid(axis='y', alpha=0.3)
ax4a.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Category")
plt.tight_layout()
out4a = os.path.join(base_dir, "viz4a_weekly_forecast_category_combined.png")
plt.savefig(out4a, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out4a}")

# 4B) Forecasted Weekly Demand by Category (Next 26 Weeks) – SMALL MULTIPLES
categories = weekly_pivot.columns.tolist()
n_cats = len(categories)

rows, cols = 3, 2  # for 6 categories
fig4b, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True, sharey=True)

for ax, cat in zip(axes.flatten(), categories):
    ax.plot(weekly_pivot.index, weekly_pivot[cat])
    ax.set_title(cat)
    ax.grid(axis='y', alpha=0.3)

# Hide any unused subplots
for ax in axes.flatten()[n_cats:]:
    ax.axis('off')

fig4b.suptitle(f"Forecasted Weekly Demand by Category (Next {forecast_weeks} Weeks)", y=0.95)
for ax in axes[-1, :]:
    ax.set_xlabel("Week (Date)")
axes[0, 0].set_ylabel("Weekly Demand")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out4b = os.path.join(base_dir, "viz4b_weekly_forecast_category_small_multiples.png")
plt.savefig(out4b, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out4b}")

# 5) Historical Average Weekly Demand by Category – vertical bar
fig5, ax5 = plt.subplots(figsize=(10, 6))
cat_weekly_avg = (
    cat_weekly_df.groupby('category')['weekly_sales']
                 .mean()
                 .reset_index()
                 .rename(columns={'weekly_sales': 'avg_weekly_sales'})
                 .sort_values('avg_weekly_sales', ascending=False)
)
ax5.bar(cat_weekly_avg['category'], cat_weekly_avg['avg_weekly_sales'])
ax5.set_title("Historical Average Weekly Demand by Category")
ax5.set_xlabel("Category")
ax5.set_ylabel("Average Weekly Demand")
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)
plt.tight_layout()
out5 = os.path.join(base_dir, "viz5_hist_avg_weekly_category.png")
plt.savefig(out5, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out5}")

# 6) Historical Product Demand by Season – line graph
fig6, ax6 = plt.subplots(figsize=(8, 5))
hist_season_plot = hist_season_demand.copy()
if not hist_season_plot.empty:
    x_idx = np.arange(len(hist_season_plot))
    ax6.plot(x_idx, hist_season_plot['total_hist_demand'], marker='o')
    ax6.set_xticks(x_idx)
    ax6.set_xticklabels(hist_season_plot['season'])
    ax6.set_title("Historical Product Demand by Season")
    ax6.set_xlabel("Season")
    ax6.set_ylabel("Total Historical Demand")
    ax6.grid(axis='y', alpha=0.3)
else:
    ax6.text(0.5, 0.5, "No season data available", ha='center', va='center')
plt.tight_layout()
out6 = os.path.join(base_dir, "viz6_hist_demand_by_season.png")
plt.savefig(out6, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out6}")

# 7) Forecasted Product Demand by Season (Next 180 Days) – line graph, no scientific notation
fig7, ax7 = plt.subplots(figsize=(8, 5))
seasonal_forecast_plot = seasonal_forecast.copy()
if not seasonal_forecast_plot.empty:
    x_idx2 = np.arange(len(seasonal_forecast_plot))
    ax7.plot(x_idx2, seasonal_forecast_plot['total_forecast_demand'], marker='o')
    ax7.set_xticks(x_idx2)
    ax7.set_xticklabels(seasonal_forecast_plot['season'])
    ax7.set_title(f"Forecasted Product Demand by Season (Next {forecast_horizon_days} Days)")
    ax7.set_xlabel("Season")
    ax7.set_ylabel("Forecast Total Demand")

    # Disable scientific notation, show full values
    ax7.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    ax7.grid(axis='y', alpha=0.3)
else:
    ax7.text(0.5, 0.5, "No season forecast available", ha='center', va='center')
plt.tight_layout()
out7 = os.path.join(base_dir, "viz7_forecast_demand_by_season.png")
plt.savefig(out7, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {out7}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 100)
print("FORECASTING SUMMARY")
print("=" * 100)

def short_sku_summary(m):
    return f"MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, MAPE={m['mape']:.1f}%"

def short_week_summary(m):
    return (f"MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, "
            f"MAPE={m['mape']:.1f}%, WAPE={m['wape']:.1f}%")

print("\nSKU-LEVEL MODELS (DAILY):")
print(f"MAIN MODEL (XGBoost):       {short_sku_summary(metrics_xgb_sku)}")
print(f"BENCHMARK 1 (RandomForest): {short_sku_summary(metrics_rf_sku)}")
print(f"BENCHMARK 2 (LinearReg):    {short_sku_summary(metrics_lin_sku)}")

print("\nCATEGORY-LEVEL MODELS (WEEKLY):")
print(f"MAIN MODEL (XGBoost):       {short_week_summary(metrics_xgb_week)}")
print(f"BENCHMARK 1 (RandomForest): {short_week_summary(metrics_rf_week)}")
print(f"BENCHMARK 2 (LinearReg):    {short_week_summary(metrics_lin_week)}")

print(f"\nTop product by forecasted demand (SKU): "
      f"{top_products_forecast.iloc[0]['product_id']} "
      f"({top_products_forecast.iloc[0]['category']}) with "
      f"{top_products_forecast.iloc[0]['total_forecast_demand']:.2f} "
      f"forecast demand over the next {forecast_horizon_days} days.")

print(f"Top category by historical demand: "
      f"{cat_hist_plot.iloc[0]['category']} with "
      f"{cat_hist_plot.iloc[0]['total_hist_demand']:.2f} total sales.")

print(f"Top category by forecasted weekly demand (next {forecast_weeks} weeks): "
      f"{cat_forecast_weekly.iloc[0]['category']} with "
      f"{cat_forecast_weekly.iloc[0]['total_forecast_demand']:.2f} forecast demand.")

if not seasonal_forecast.empty:
    print(f"\nTop season by forecasted product demand (next {forecast_horizon_days} days): "
          f"{seasonal_forecast.iloc[0]['season']} with "
          f"{seasonal_forecast.iloc[0]['total_forecast_demand']:.2f} total forecast demand.")

print("\n" + "=" * 100)
print("SKU & CATEGORY DEMAND FORECASTING COMPLETE")
print("=" * 100)
