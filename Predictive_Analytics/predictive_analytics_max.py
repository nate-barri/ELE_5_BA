import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

# XGBoost (main model for demand forecasting)
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost is not installed. Please run: pip install xgboost")

# =============================================================================
# PART 0: DATA INGESTION (SAME STYLE AS EXISTING CODE)
# =============================================================================

df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")

df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df = df.sort_values('purchase_date').reset_index(drop=True)

# Ensure optional columns exist (for safety; if missing, create them)
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
print(" " * 25 + "PRODUCT DEMAND FORECASTING (SKU-LEVEL)")
print("=" * 100)
print(f"\nDataset: {len(df):,} transactions")
print(f"Date Range: {df['purchase_date'].min().strftime('%Y-%m-%d')} to {df['purchase_date'].max().strftime('%Y-%m-%d')}")
print(f"Total Sales: ${df['total_sales'].sum():,.2f}")
print(f"Unique Products: {df['product_id'].nunique():,}")
print(f"Categories: {df['category'].nunique():,}")

# Treat "demand" as DAILY PRODUCT-LEVEL SALES (sum of total_sales per product per day)

# =============================================================================
# PART 1: BUILD DAILY PRODUCT-LEVEL DEMAND TABLE
# =============================================================================
print("\n" + "=" * 100)
print("PART 1: BUILDING DAILY PRODUCT DEMAND DATASET")
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

print(f"\nDaily product-level rows: {len(demand_df):,}")
print(f"Unique product_ids in demand dataset: {demand_df['product_id'].nunique():,}")

for col in ['avg_rating', 'avg_markdown', 'avg_stock']:
    demand_df[col] = demand_df[col].fillna(demand_df[col].median())

# =============================================================================
# PART 1B: HISTORICAL DEMAND DESCRIPTIVE INSIGHTS (OPTIONAL BUT ADDED)
# =============================================================================
print("\n" + "=" * 100)
print("PART 1B: HISTORICAL DEMAND DESCRIPTIVE INSIGHTS")
print("=" * 100)

# 1) TOP-SELLING PRODUCTS HISTORICALLY
hist_product_demand = (
    demand_df.groupby('product_id')
             .agg(
                 total_hist_demand=('daily_sales', 'sum'),
                 avg_daily_demand=('daily_sales', 'mean'),
                 num_days_sold=('daily_sales', lambda x: (x > 0).sum()),
                 category=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
             )
             .reset_index()
)

top_hist_n = 10
top_hist_products = hist_product_demand.sort_values('total_hist_demand', ascending=False).head(top_hist_n)

print(f"\nTOP {top_hist_n} HISTORICAL BEST-SELLING PRODUCTS:")
print("-" * 100)
print(top_hist_products[['product_id', 'category', 'total_hist_demand', 'avg_daily_demand', 'num_days_sold']])

# 2) DEMAND DISTRIBUTION PER CATEGORY (HISTORICAL)
cat_demand = (
    demand_df.groupby('category')['daily_sales']
             .sum()
             .reset_index()
             .rename(columns={'daily_sales': 'total_hist_demand'})
)
cat_demand['share_pct'] = (cat_demand['total_hist_demand'] / cat_demand['total_hist_demand'].sum()) * 100

print("\nDEMAND DISTRIBUTION BY CATEGORY (HISTORICAL):")
print("-" * 100)
print(cat_demand.sort_values('total_hist_demand', ascending=False))

# 3) AVERAGE MONTHLY DEMAND PER PRODUCT
demand_df['year_month'] = demand_df['date'].dt.to_period('M')

monthly_product_demand = (
    demand_df.groupby(['product_id', 'year_month'])['daily_sales']
             .sum()
             .reset_index()
             .rename(columns={'daily_sales': 'monthly_demand'})
)

avg_monthly_demand = (
    monthly_product_demand.groupby('product_id')['monthly_demand']
                          .mean()
                          .reset_index()
                          .rename(columns={'monthly_demand': 'avg_monthly_demand'})
)

avg_monthly_demand = avg_monthly_demand.merge(
    hist_product_demand[['product_id', 'category']],
    on='product_id',
    how='left'
)

top_avg_monthly = avg_monthly_demand.sort_values('avg_monthly_demand', ascending=False).head(top_hist_n)

print(f"\nTOP {top_hist_n} PRODUCTS BY AVERAGE MONTHLY DEMAND:")
print("-" * 100)
print(top_avg_monthly[['product_id', 'category', 'avg_monthly_demand']])

# 4) PRODUCTS WITH MOST SPIKY / VARIABLE DEMAND
spike_df = (
    demand_df.groupby('product_id')
             .agg(
                 mean_daily=('daily_sales', 'mean'),
                 std_daily=('daily_sales', 'std'),
                 category=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
             )
             .reset_index()
)

spike_df['cv'] = spike_df['std_daily'] / (spike_df['mean_daily'] + 1e-6)  # coefficient of variation
spikiest_products = spike_df.sort_values('cv', ascending=False).head(top_hist_n)

print(f"\nTOP {top_hist_n} PRODUCTS WITH MOST SPIKY DEMAND (BY CV):")
print("-" * 100)
print(spikiest_products[['product_id', 'category', 'mean_daily', 'std_daily', 'cv']])

# =============================================================================
# PART 2: FEATURE ENGINEERING FOR DEMAND FORECASTING
# =============================================================================
print("\n" + "=" * 100)
print("PART 2: FEATURE ENGINEERING")
print("=" * 100)

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

demand_df['global_trend'] = (demand_df['date'] - demand_df['date'].min()).dt.days

demand_df['product_code'], product_unique = pd.factorize(demand_df['product_id'])
demand_df['category_code'], category_unique = pd.factorize(demand_df['category'])

print(f"\nEncoded product_id into 'product_code' (0 to {demand_df['product_code'].max()})")
print(f"Encoded category into 'category_code' (0 to {demand_df['category_code'].max()})")

target_col = 'daily_sales'

feature_cols = [
    'product_code', 'category_code',
    'avg_original_price', 'avg_current_price', 'avg_markdown',
    'avg_stock', 'avg_rating',
    'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_weekend',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'global_trend'
]

X = demand_df[feature_cols].copy()
y = demand_df[target_col].copy()

# =============================================================================
# PART 3: TRAIN-TEST SPLIT (TIME-BASED)
# =============================================================================
print("\n" + "=" * 100)
print("PART 3: TRAIN / TEST SPLIT")
print("=" * 100)

demand_df = demand_df.sort_values('date').reset_index(drop=True)
X = demand_df[feature_cols]
y = demand_df[target_col]
dates = demand_df['date']
products = demand_df['product_id']

split_idx = int(len(demand_df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]
products_test = products.iloc[split_idx:]

print(f"\nTraining Period: {dates_train.min().strftime('%Y-%m-%d')} to {dates_train.max().strftime('%Y-%m-%d')}")
print(f"Testing Period:  {dates_test.min().strftime('%Y-%m-%d')} to {dates_test.max().strftime('%Y-%m-%d')}")
print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# PART 4: TRAINING MODELS (XGBOOST + BENCHMARKS)
# =============================================================================
print("\n" + "=" * 100)
print("PART 4: TRAINING DEMAND FORECASTING MODELS")
print("=" * 100)

print("\nTraining XGBoost Regressor (MAIN demand model)...")
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

print("Training Random Forest Regressor (BENCHMARK 1)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print("Training Linear Regression (BENCHMARK 2)...")
lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

# =============================================================================
# PART 5: MODEL EVALUATION
# =============================================================================
print("\n" + "=" * 100)
print("PART 5: MODEL EVALUATION (PRODUCT DEMAND)")
print("=" * 100)

def evaluate_model(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    print(f"\n{name}")
    print("-" * 60)
    print(f"R²:    {r2:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    return {'name': name, 'r2': r2, 'mae': mae, 'rmse': rmse, 'mape': mape}

y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_lin = lin_model.predict(X_test_scaled)

metrics_xgb = evaluate_model("XGBoost (MAIN MODEL)", y_test, y_pred_xgb)
metrics_rf = evaluate_model("Random Forest (BENCHMARK 1)", y_test, y_pred_rf)
metrics_lin = evaluate_model("Linear Regression (BENCHMARK 2)", y_test, y_pred_lin)

# =============================================================================
# PART 6: FUTURE DEMAND FORECAST (NEXT 30 DAYS) + TOP PRODUCTS
# =============================================================================
print("\n" + "=" * 100)
print("PART 6: FUTURE DEMAND FORECAST (NEXT 30 DAYS)")
print("=" * 100)

last_date = demand_df['date'].max()
forecast_horizon = 30
future_dates = pd.date_range(start=last_date + timedelta(days=1),
                             periods=forecast_horizon, freq='D')

all_products = demand_df[['product_id', 'product_code', 'category', 'category_code']].drop_duplicates()

future_rows = []

for d in future_dates:
    year = d.year
    month = d.month
    day = d.day
    day_of_week = d.weekday()
    day_of_year = d.timetuple().tm_yday
    week_of_year = d.isocalendar().week
    quarter = (month - 1) // 3 + 1
    is_weekend = 1 if day_of_week in [5, 6] else 0

    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)

    global_trend = (d - demand_df['date'].min()).days

    for _, row in all_products.iterrows():
        pid = row['product_id']
        pcode = row['product_code']
        ccode = row['category_code']

        hist_rows = demand_df[demand_df['product_id'] == pid]
        if len(hist_rows) > 0:
            avg_original_price = hist_rows['avg_original_price'].iloc[-1]
            avg_current_price = hist_rows['avg_current_price'].iloc[-1]
            avg_markdown = hist_rows['avg_markdown'].iloc[-1]
            avg_stock = hist_rows['avg_stock'].iloc[-1]
            avg_rating = hist_rows['avg_rating'].iloc[-1]
        else:
            avg_original_price = demand_df['avg_original_price'].median()
            avg_current_price = demand_df['avg_current_price'].median()
            avg_markdown = demand_df['avg_markdown'].median()
            avg_stock = demand_df['avg_stock'].median()
            avg_rating = demand_df['avg_rating'].median()

        future_rows.append({
            'date': d,
            'product_id': pid,
            'product_code': pcode,
            'category_code': ccode,
            'avg_original_price': avg_original_price,
            'avg_current_price': avg_current_price,
            'avg_markdown': avg_markdown,
            'avg_stock': avg_stock,
            'avg_rating': avg_rating,
            'month': month,
            'day_of_week': day_of_week,
            'day_of_year': day_of_year,
            'week_of_year': week_of_year,
            'quarter': quarter,
            'is_weekend': is_weekend,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'global_trend': global_trend
        })

future_df = pd.DataFrame(future_rows)

X_future = future_df[feature_cols]
future_df['predicted_demand'] = xgb_model.predict(X_future)

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
top_products = product_forecast.head(top_n)

print(f"\nTOP {top_n} PREDICTED IN-DEMAND PRODUCTS (NEXT {forecast_horizon} DAYS):")
print("-" * 100)
for _, row in top_products.iterrows():
    print(f"{row['product_id']:>10s} | {row['category']:<15s} | "
          f"Total Demand: {row['total_forecast_demand']:.2f} | "
          f"Avg Daily: {row['avg_daily_demand']:.2f}")

# =============================================================================
# PART 7: DASHBOARD VISUALIZATIONS (ONE PNG WITH ALL CHARTS)
# =============================================================================
print("\n" + "=" * 100)
print("PART 7: VISUALIZATIONS DASHBOARD")
print("=" * 100)

# Prepare subsets for plotting
top_hist_plot = top_hist_products.copy()
top_hist_plot = top_hist_plot.sort_values('total_hist_demand', ascending=False)

cat_plot = cat_demand.sort_values('total_hist_demand', ascending=False)

spiky_plot = spikiest_products.copy()
spiky_plot = spiky_plot.sort_values('cv', ascending=False)

future_top_plot = top_products.copy()
future_top_plot = future_top_plot.sort_values('total_forecast_demand', ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
(fig_ax1, fig_ax2), (fig_ax3, fig_ax4) = axes

# 1) Historical Top-Selling Products
fig_ax1.bar(top_hist_plot['product_id'], top_hist_plot['total_hist_demand'])
fig_ax1.set_title(f"Top {top_hist_n} Historical Best-Selling Products")
fig_ax1.set_xlabel("Product ID")
fig_ax1.set_ylabel("Total Historical Demand")
fig_ax1.tick_params(axis='x', rotation=45)
fig_ax1.grid(axis='y', alpha=0.3)

# 2) Category Demand Distribution
fig_ax2.bar(cat_plot['category'], cat_plot['total_hist_demand'])
fig_ax2.set_title("Historical Demand by Category")
fig_ax2.set_xlabel("Category")
fig_ax2.set_ylabel("Total Historical Demand")
fig_ax2.tick_params(axis='x', rotation=45)
fig_ax2.grid(axis='y', alpha=0.3)

# 3) Spikiest Products (by CV)
fig_ax3.bar(spiky_plot['product_id'], spiky_plot['cv'])
fig_ax3.set_title(f"Top {top_hist_n} Spikiest Products (Coefficient of Variation)")
fig_ax3.set_xlabel("Product ID")
fig_ax3.set_ylabel("CV (std / mean)")
fig_ax3.tick_params(axis='x', rotation=45)
fig_ax3.grid(axis='y', alpha=0.3)

# 4) Future Top Products (Next 30 Days)
fig_ax4.bar(future_top_plot['product_id'], future_top_plot['total_forecast_demand'])
fig_ax4.set_title(f"Top {top_n} Predicted In-Demand Products (Next {forecast_horizon} Days)")
fig_ax4.set_xlabel("Product ID")
fig_ax4.set_ylabel("Forecast Total Demand")
fig_ax4.tick_params(axis='x', rotation=45)
fig_ax4.grid(axis='y', alpha=0.3)

fig.suptitle("Product Demand – Historical & Forecasting Dashboard", fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save as ONE PNG file with all visualizations
output_file = "product_demand_forecasting_dashboard.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Dashboard saved as: {output_file}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("DEMAND FORECASTING SUMMARY")
print("=" * 100)

def short_model_summary(m):
    return f"R²={m['r2']:.3f}, MAPE={m['mape']:.1f}%, MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}"

print(f"\nMAIN MODEL (XGBoost):       {short_model_summary(metrics_xgb)}")
print(f"BENCHMARK 1 (RandomForest): {short_model_summary(metrics_rf)}")
print(f"BENCHMARK 2 (LinReg):       {short_model_summary(metrics_lin)}")

print(f"\nTop product by forecasted demand: {top_products.iloc[0]['product_id']} "
      f"({top_products.iloc[0]['category']}) with "
      f"{top_products.iloc[0]['total_forecast_demand']:.2f} units of forecast demand "
      f"over the next {forecast_horizon} days.")

print("\n" + "=" * 100)
print("PRODUCT DEMAND FORECASTING COMPLETE")
print("=" * 100)
