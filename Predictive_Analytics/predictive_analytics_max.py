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

# XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("XGBoost is not installed. Please run: pip install xgboost")

# =====================================================================
# FIXED SEASON MAPPING
# =====================================================================
def assign_season(month: int) -> str:
    """Map calendar month to season."""
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"
    else:
        return "Winter"


# =====================================================================
# ABC ANALYSIS FOR PRODUCT SEGMENTATION
# =====================================================================
def perform_abc_analysis(df, value_col='total_sales', id_col='product_id'):
    """
    Classify products into A, B, C categories based on sales contribution.
    A: Top 20% of products contributing 80% of sales
    B: Next 30% contributing 15% of sales
    C: Remaining 50% contributing 5% of sales
    """
    product_sales = df.groupby(id_col)[value_col].sum().reset_index()
    product_sales = product_sales.sort_values(value_col, ascending=False)
    product_sales['cumulative_sales'] = product_sales[value_col].cumsum()
    product_sales['cumulative_pct'] = (product_sales['cumulative_sales'] / 
                                       product_sales[value_col].sum() * 100)
    
    product_sales['abc_category'] = 'C'
    product_sales.loc[product_sales['cumulative_pct'] <= 80, 'abc_category'] = 'A'
    product_sales.loc[(product_sales['cumulative_pct'] > 80) & 
                     (product_sales['cumulative_pct'] <= 95), 'abc_category'] = 'B'
    
    return product_sales[[id_col, 'abc_category']]


# =====================================================================
# METRICS
# =====================================================================
def compute_mase(y_true, y_pred, seasonality=1):
    """Mean Absolute Scaled Error (MASE)"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if len(y_true) <= seasonality + 1:
        return np.nan
    
    naive_actual = y_true[seasonality:]
    naive_forecast = y_true[:-seasonality]
    mae_naive = np.mean(np.abs(naive_actual - naive_forecast)) + 1e-8
    mae_model = np.mean(np.abs(y_true[seasonality:] - y_pred[seasonality:]))
    
    return mae_model / mae_naive




def compute_mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    # Avoid division by zero - only calculate MAPE for non-zero actuals
    mask = y_true != 0
    if not mask.any():
        return np.nan
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(name, y_true, y_pred):
    """Evaluate model with multiple metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mase = compute_mase(y_true, y_pred, seasonality=1)
    mape = compute_mape(y_true, y_pred)  # Add MAPE
    
    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MASE:  {mase:.4f}")
    print(f"MAPE:  {mape:.2f}%")  # Display MAPE
    return {'name': name, 'mae': mae, 'rmse': rmse, 'mase': mase,'mape': mape}

# =============================================================================
# PART 0: DATA INGESTION
# =============================================================================
print("=" * 100)
print(" " * 15 + "ENHANCED SKU-LEVEL & WEEKLY CATEGORY-LEVEL DEMAND FORECASTING")
print("=" * 100)

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

print(f"\nDataset: {len(df):,} transactions")
print(f"Date Range: {df['purchase_date'].min().strftime('%Y-%m-%d')} "
      f"to {df['purchase_date'].max().strftime('%Y-%m-%d')}")
print(f"Total Sales: ${df['total_sales'].sum():,.2f}")
print(f"Unique Products: {df['product_id'].nunique():,}")
print(f"Categories: {df['category'].nunique():,}")

# Perform ABC Analysis
abc_classification = perform_abc_analysis(df)
df = df.merge(abc_classification, on='product_id', how='left')

print("\n" + "=" * 100)
print("ABC ANALYSIS RESULTS")
print("=" * 100)
abc_summary = df.groupby('abc_category').agg({
    'product_id': 'nunique',
    'total_sales': 'sum'
}).reset_index()
abc_summary['sales_pct'] = (abc_summary['total_sales'] / 
                             abc_summary['total_sales'].sum() * 100)
print(abc_summary)


# =============================================================================
# PART 1: MONTHLY PRODUCT-LEVEL DATASET (SKU) + DESCRIPTIVE
# =============================================================================
print("\n" + "=" * 100)
print("PART 1: MONTHLY PRODUCT DEMAND (SKU-LEVEL) - ENHANCED")
print("=" * 100)

df['month_start'] = df['purchase_date'].values.astype('datetime64[M]')

demand_df_raw = (
    df.groupby(['month_start', 'product_id', 'category', 'abc_category'])
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

demand_df_raw.rename(columns={'month_start': 'date'}, inplace=True)

# Build full grid
full_months = pd.date_range(
    start=demand_df_raw['date'].min(),
    end=demand_df_raw['date'].max(),
    freq='MS'
)
calendar_df = pd.DataFrame({'date': full_months})

product_cat = demand_df_raw[['product_id', 'category', 'abc_category']].drop_duplicates()
product_cat['key'] = 1
calendar_df['key'] = 1

full_grid = calendar_df.merge(product_cat, on='key', how='outer').drop(columns='key')
demand_df = full_grid.merge(
    demand_df_raw,
    on=['date', 'product_id', 'category', 'abc_category'],
    how='left'
)

# Fill missing values
demand_df['daily_sales'] = demand_df['daily_sales'].fillna(0)
demand_df['num_transactions'] = demand_df['num_transactions'].fillna(0)

for col in ['avg_rating', 'avg_original_price', 'avg_current_price',
            'avg_markdown', 'avg_stock']:
    if col in demand_df.columns:
        demand_df[col] = (
            demand_df
            .sort_values(['product_id', 'date'])
            .groupby('product_id')[col]
            .ffill()
        )
        demand_df[col] = demand_df[col].fillna(demand_df[col].median())

demand_df['month'] = demand_df['date'].dt.month
demand_df['season'] = demand_df['month'].apply(assign_season)
month_to_season = {m: assign_season(m) for m in range(1, 13)}

# =============================================================================
# ENHANCED FEATURE ENGINEERING: ADD LAG FEATURES
# =============================================================================
print("\nAdding lag features and rolling statistics...")

demand_df = demand_df.sort_values(['product_id', 'date']).reset_index(drop=True)

# Add lag features per product
for product in demand_df['product_id'].unique():
    mask = demand_df['product_id'] == product
    demand_df.loc[mask, 'lag_1_month'] = demand_df.loc[mask, 'daily_sales'].shift(1)
    demand_df.loc[mask, 'lag_2_month'] = demand_df.loc[mask, 'daily_sales'].shift(2)
    demand_df.loc[mask, 'lag_3_month'] = demand_df.loc[mask, 'daily_sales'].shift(3)
    demand_df.loc[mask, 'rolling_mean_3m'] = demand_df.loc[mask, 'daily_sales'].rolling(3, min_periods=1).mean()
    demand_df.loc[mask, 'rolling_std_3m'] = demand_df.loc[mask, 'daily_sales'].rolling(3, min_periods=1).std()

# Fill NaN values
demand_df['lag_1_month'] = demand_df['lag_1_month'].fillna(0)
demand_df['lag_2_month'] = demand_df['lag_2_month'].fillna(0)
demand_df['lag_3_month'] = demand_df['lag_3_month'].fillna(0)
demand_df['rolling_mean_3m'] = demand_df['rolling_mean_3m'].fillna(0)
demand_df['rolling_std_3m'] = demand_df['rolling_std_3m'].fillna(0)

print(f"Monthly product-level rows: {len(demand_df):,}")
print(f"Unique product_ids: {demand_df['product_id'].nunique():,}")

# Historical analysis
hist_product_demand = (
    demand_df.groupby('product_id')
             .agg(
                 total_hist_demand=('daily_sales', 'sum'),
                 avg_daily_demand=('daily_sales', 'mean'),
                 num_days_sold=('daily_sales', lambda x: (x > 0).sum()),
                 category=('category', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
                 abc_category=('abc_category', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
             )
             .reset_index()
)

top_hist_n = 10
top_hist_products = hist_product_demand.sort_values(
    'total_hist_demand', ascending=False
).head(top_hist_n)

print(f"\nTOP {top_hist_n} HISTORICAL BEST-SELLING PRODUCTS (SKU):")
print("-" * 100)
print(top_hist_products[['product_id', 'category', 'abc_category',
                         'total_hist_demand', 'avg_daily_demand', 'num_days_sold']])

min_date = demand_df['date'].min()

# =============================================================================
# PART 2: ENHANCED MODELING WITH REGULARIZED XGBOOST
# =============================================================================
print("\n" + "=" * 100)
print("PART 2: SKU-LEVEL MODELING - ENHANCED")
print("=" * 100)

# Encodings
demand_df['product_code'], product_unique = pd.factorize(demand_df['product_id'])
demand_df['category_code'], category_unique = pd.factorize(demand_df['category'])
demand_df['abc_code'], abc_unique = pd.factorize(demand_df['abc_category'])

# Time features
demand_df['year'] = demand_df['date'].dt.year
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

# Enhanced feature set with lags
sku_target_col = 'daily_sales'
sku_feature_cols = [
    'product_code', 'category_code', 'abc_code',
    'avg_original_price', 'avg_current_price', 'avg_markdown',
    'avg_stock', 'avg_rating',
    'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_weekend',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
    'global_trend',
    'lag_1_month', 'lag_2_month', 'lag_3_month',
    'rolling_mean_3m', 'rolling_std_3m'
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
print(f"Train samples: {len(X_sku_train):,}, Test samples: {len(X_sku_test):,}")

scaler_sku = StandardScaler()
X_sku_train_scaled = scaler_sku.fit_transform(X_sku_train)
X_sku_test_scaled = scaler_sku.transform(X_sku_test)

# Train models with improved hyperparameters
print("\nTraining Enhanced XGBoost (with regularization)...")
xgb_sku_model = XGBRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1
)
xgb_sku_model.fit(X_sku_train, y_sku_train)

print("Training Random Forest...")
rf_sku_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf_sku_model.fit(X_sku_train, y_sku_train)

print("Training Linear Regression (PRIMARY MODEL)...")
lin_sku_model = LinearRegression()
lin_sku_model.fit(X_sku_train_scaled, y_sku_train)

print("\n" + "=" * 100)
print("SKU-LEVEL MODEL EVALUATION")
print("=" * 100)

y_sku_pred_xgb = xgb_sku_model.predict(X_sku_test)
y_sku_pred_rf = rf_sku_model.predict(X_sku_test)
y_sku_pred_lin = lin_sku_model.predict(X_sku_test_scaled)

metrics_xgb_sku = evaluate_model("Enhanced XGBoost", y_sku_test, y_sku_pred_xgb)
metrics_rf_sku = evaluate_model("Random Forest", y_sku_test, y_sku_pred_rf)
metrics_lin_sku = evaluate_model("Linear Regression (PRIMARY)", y_sku_test, y_sku_pred_lin)

# Ensemble model (average of all three)
y_sku_pred_ensemble = (y_sku_pred_xgb + y_sku_pred_rf + y_sku_pred_lin) / 3
metrics_ensemble_sku = evaluate_model("Ensemble (Average)", y_sku_test, y_sku_pred_ensemble)

# Select best model
best_model_sku = min([metrics_xgb_sku, metrics_rf_sku, metrics_lin_sku, metrics_ensemble_sku],
                     key=lambda x: x['mae'])
print(f"\n*** BEST SKU MODEL: {best_model_sku['name']} ***")

# =============================================================================
# PART 3: FUTURE FORECASTING WITH BEST MODEL
# =============================================================================
print("\n" + "=" * 100)
print("PART 3: SKU-LEVEL FUTURE DEMAND FORECAST (NEXT 6 MONTHS)")
print("=" * 100)

last_date_sku = demand_df['date'].max()
forecast_horizon_months = 6

future_dates = pd.date_range(
    start=last_date_sku + pd.offsets.MonthBegin(1),
    periods=forecast_horizon_months,
    freq='MS'
)

# Prepare future features with lags
last_product_info = (
    demand_df.sort_values('date')
             .groupby('product_id')
             .tail(3)  # Get last 3 months for lag calculation
)

future_rows = []
for date in future_dates:
    month = date.month
    for product in demand_df['product_id'].unique():
        product_data = last_product_info[last_product_info['product_id'] == product]
        if len(product_data) > 0:
            last_row = product_data.iloc[-1]
            
            # Calculate lags from recent history
            recent_sales = product_data['daily_sales'].values
            lag_1 = recent_sales[-1] if len(recent_sales) >= 1 else 0
            lag_2 = recent_sales[-2] if len(recent_sales) >= 2 else 0
            lag_3 = recent_sales[-3] if len(recent_sales) >= 3 else 0
            rolling_mean = np.mean(recent_sales) if len(recent_sales) > 0 else 0
            rolling_std = np.std(recent_sales) if len(recent_sales) > 0 else 0
            
            future_rows.append({
                'date': date,
                'product_id': product,
                'product_code': last_row['product_code'],
                'category_code': last_row['category_code'],
                'abc_code': last_row['abc_code'],
                'avg_original_price': last_row['avg_original_price'],
                'avg_current_price': last_row['avg_current_price'],
                'avg_markdown': last_row['avg_markdown'],
                'avg_stock': last_row['avg_stock'],
                'avg_rating': last_row['avg_rating'],
                'month': month,
                'day_of_week': date.dayofweek,
                'day_of_year': date.dayofyear,
                'week_of_year': date.isocalendar().week,
                'quarter': (month - 1) // 3 + 1,
                'is_weekend': 1 if date.dayofweek in [5, 6] else 0,
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'dow_sin': np.sin(2 * np.pi * date.dayofweek / 7),
                'dow_cos': np.cos(2 * np.pi * date.dayofweek / 7),
                'global_trend': (date - min_date).days,
                'lag_1_month': lag_1,
                'lag_2_month': lag_2,
                'lag_3_month': lag_3,
                'rolling_mean_3m': rolling_mean,
                'rolling_std_3m': rolling_std,
                'season': month_to_season[month],
                'category': last_row['category'],
                'abc_category': last_row['abc_category']
            })

future_df = pd.DataFrame(future_rows)
X_sku_future = future_df[sku_feature_cols]

# Use best model for predictions
if best_model_sku['name'] == 'Linear Regression (PRIMARY)':
    X_sku_future_scaled = scaler_sku.transform(X_sku_future)
    future_df['predicted_demand'] = lin_sku_model.predict(X_sku_future_scaled)
elif best_model_sku['name'] == 'Ensemble (Average)':
    X_sku_future_scaled = scaler_sku.transform(X_sku_future)
    pred_xgb = xgb_sku_model.predict(X_sku_future)
    pred_rf = rf_sku_model.predict(X_sku_future)
    pred_lin = lin_sku_model.predict(X_sku_future_scaled)
    future_df['predicted_demand'] = (pred_xgb + pred_rf + pred_lin) / 3
else:
    future_df['predicted_demand'] = (xgb_sku_model if 'XGBoost' in best_model_sku['name'] 
                                     else rf_sku_model).predict(X_sku_future)

# Aggregate forecasts
product_forecast = (
    future_df.groupby(['product_id', 'abc_category', 'category'])
             .agg(
                 total_forecast_demand=('predicted_demand', 'sum'),
                 avg_monthly_demand=('predicted_demand', 'mean')
             )
             .reset_index()
             .sort_values('total_forecast_demand', ascending=False)
)

top_n = 10
top_products_forecast = product_forecast.head(top_n)

print(f"\nTOP {top_n} PREDICTED IN-DEMAND PRODUCTS (NEXT {forecast_horizon_months} MONTHS):")
print("-" * 100)
for _, row in top_products_forecast.iterrows():
    print(f"{row['product_id']:>10s} | {row['abc_category']:>3s} | "
          f"{row['category']:<15s} | Total: ${row['total_forecast_demand']:>8.2f} | "
          f"Avg Monthly: ${row['avg_monthly_demand']:>6.2f}")

# Seasonal forecast
seasonal_forecast = (
    future_df.groupby('season')['predicted_demand']
             .sum()
             .reset_index()
             .rename(columns={'predicted_demand': 'total_forecast_demand'})
             .sort_values('total_forecast_demand', ascending=False)
)

print(f"\nFORECASTED PRODUCT DEMAND BY SEASON (NEXT {forecast_horizon_months} MONTHS):")
print("-" * 100)
print(seasonal_forecast)

# Category forecast
category_forecast = (
    future_df.groupby('category')['predicted_demand']
             .sum()
             .reset_index()
             .rename(columns={'predicted_demand': 'total_forecast_demand'})
             .sort_values('total_forecast_demand', ascending=False)
)

print(f"\nFORECASTED DEMAND BY CATEGORY (NEXT {forecast_horizon_months} MONTHS):")
print("-" * 100)
print(category_forecast)

# ABC forecast
abc_forecast = (
    future_df.groupby('abc_category')['predicted_demand']
             .sum()
             .reset_index()
             .rename(columns={'predicted_demand': 'total_forecast_demand'})
             .sort_values('total_forecast_demand', ascending=False)
)

print(f"\nFORECASTED DEMAND BY ABC CATEGORY (NEXT {forecast_horizon_months} MONTHS):")
print("-" * 100)
print(abc_forecast)

# =============================================================================
# EXPORT ENHANCED RESULTS
# =============================================================================
base_dir = os.path.dirname(__file__)
export_dir = os.path.join(base_dir, "powerbi_exports_enhanced")
os.makedirs(export_dir, exist_ok=True)

print("\n" + "=" * 100)
print("EXPORTING ENHANCED RESULTS")
print("=" * 100)
print(f"Saving to: {export_dir}")

# Export key tables
demand_df.to_csv(os.path.join(export_dir, "sku_monthly_demand_enhanced.csv"), index=False)
top_hist_products.to_csv(os.path.join(export_dir, "top10_hist_products.csv"), index=False)
product_forecast.to_csv(os.path.join(export_dir, "sku_future_forecast_6months.csv"), index=False)
future_df.to_csv(os.path.join(export_dir, "detailed_future_forecast.csv"), index=False)
seasonal_forecast.to_csv(os.path.join(export_dir, "seasonal_forecast.csv"), index=False)
category_forecast.to_csv(os.path.join(export_dir, "category_forecast.csv"), index=False)
abc_forecast.to_csv(os.path.join(export_dir, "abc_forecast.csv"), index=False)

# Model comparison
model_comparison = pd.DataFrame([
    metrics_xgb_sku,
    metrics_rf_sku,
    metrics_lin_sku,
    metrics_ensemble_sku
])
model_comparison.to_csv(os.path.join(export_dir, "model_comparison.csv"), index=False)

print("✓ All enhanced exports saved")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 100)
print("ENHANCED FORECASTING SUMMARY")
print("=" * 100)

print(f"\nBest Model: {best_model_sku['name']}")
print(f"  MAE:  {best_model_sku['mae']:.2f}")
print(f"  RMSE: {best_model_sku['rmse']:.2f}")
print(f"  MASE: {best_model_sku['mase']:.3f}")
print(f"  MAPE: {best_model_sku['mape']:.2f}%")

print(f"\nTop Product (Next 6 Months): {top_products_forecast.iloc[0]['product_id']}")
print(f"  Category: {top_products_forecast.iloc[0]['category']}")
print(f"  ABC Class: {top_products_forecast.iloc[0]['abc_category']}")
print(f"  Forecast: ${top_products_forecast.iloc[0]['total_forecast_demand']:.2f}")

print(f"\nTop Season: {seasonal_forecast.iloc[0]['season']} "
      f"(${seasonal_forecast.iloc[0]['total_forecast_demand']:,.2f})")

print(f"\nTop Category: {category_forecast.iloc[0]['category']} "
      f"(${category_forecast.iloc[0]['total_forecast_demand']:,.2f})")

# =============================================================================
# PART 4: VISUALIZATIONS (DISPLAY ONLY) + CSV EXPORTS FOR POWER BI
# =============================================================================
print("\n" + "=" * 100)
print("PART 4: DISPLAYING VISUALIZATIONS AND GENERATING CSV FILES FOR POWER BI")
print("=" * 100)

# Set style for matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.Set3(np.linspace(0, 1, 10))

# =============================================================================
# DISPLAY VISUALIZATIONS (NO PNG SAVING)
# =============================================================================
print("\nDisplaying visualizations...")

# 1) Model Performance Comparison
fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
fig1.suptitle('Model Performance Comparison - SKU Level', fontsize=16, fontweight='bold')

metrics_data = [metrics_xgb_sku, metrics_rf_sku, metrics_lin_sku, metrics_ensemble_sku]
model_names = ['XGBoost\n(Enhanced)', 'Random\nForest', 'Linear\nRegression', 'Ensemble']

axes1[0, 0].bar(model_names, [m['mae'] for m in metrics_data], color=colors[:4])
axes1[0, 0].set_title('Mean Absolute Error (MAE)', fontweight='bold')
axes1[0, 0].set_ylabel('MAE')
axes1[0, 0].grid(axis='y', alpha=0.3)

axes1[0, 1].bar(model_names, [m['rmse'] for m in metrics_data], color=colors[:4])
axes1[0, 1].set_title('Root Mean Squared Error (RMSE)', fontweight='bold')
axes1[0, 1].set_ylabel('RMSE')
axes1[0, 1].grid(axis='y', alpha=0.3)

axes1[0, 2].bar(model_names, [m['mase'] for m in metrics_data], color=colors[:4])
axes1[0, 2].set_title('Mean Absolute Scaled Error (MASE)', fontweight='bold')
axes1[0, 2].set_ylabel('MASE')
axes1[0, 2].axhline(y=1, color='r', linestyle='--', label='Naive Forecast')
axes1[0, 2].legend()
axes1[0, 2].grid(axis='y', alpha=0.3)

axes1[1, 0].bar(model_names, [m['mape'] for m in metrics_data], color=colors[:4])
axes1[1, 0].set_title('Mean Absolute Percentage Error (MAPE)', fontweight='bold')
axes1[1, 0].set_ylabel('MAPE (%)')
axes1[1, 0].grid(axis='y', alpha=0.3)

axes1[1, 1].axis('off')
axes1[1, 2].axis('off')

plt.tight_layout()
plt.show()
print("✓ Displayed: Model Comparison")

# 2) ABC Analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('ABC Analysis - Product Classification', fontsize=16, fontweight='bold')

abc_counts = abc_summary.set_index('abc_category')['product_id']
axes2[0].pie(abc_counts, labels=abc_counts.index, autopct='%1.1f%%', 
             colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
axes2[0].set_title('Product Distribution by ABC Category', fontweight='bold')

abc_sales = abc_summary.set_index('abc_category')['total_sales']
axes2[1].bar(abc_sales.index, abc_sales.values, color=['#ff9999', '#66b3ff', '#99ff99'])
axes2[1].set_title('Sales Distribution by ABC Category', fontweight='bold')
axes2[1].set_ylabel('Total Sales ($)')
axes2[1].yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
axes2[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Displayed: ABC Analysis")

# 3) Top 10 Historical vs Forecasted
fig3, ax3 = plt.subplots(figsize=(12, 8))

top_hist_ids = top_hist_products['product_id'].values
top_forecast_matched = product_forecast[product_forecast['product_id'].isin(top_hist_ids)].set_index('product_id')
top_hist_matched = top_hist_products.set_index('product_id')

x = np.arange(len(top_hist_ids))
width = 0.35

ax3.bar(x - width/2, [top_hist_matched.loc[pid, 'total_hist_demand'] for pid in top_hist_ids],
        width, label='Historical', color='steelblue', alpha=0.8)
ax3.bar(x + width/2, [top_forecast_matched.loc[pid, 'total_forecast_demand'] 
                       if pid in top_forecast_matched.index else 0 for pid in top_hist_ids],
        width, label='Forecast (6mo)', color='coral', alpha=0.8)

ax3.set_xlabel('Product ID', fontweight='bold')
ax3.set_ylabel('Demand ($)', fontweight='bold')
ax3.set_title('Top 10 Historical Products: Historical vs Forecasted Demand', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(top_hist_ids, rotation=45, ha='right')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))

plt.tight_layout()
plt.show()
print("✓ Displayed: Historical vs Forecast")

# 4) Seasonal Forecast
fig4, ax4 = plt.subplots(figsize=(10, 6))

season_order = ['Spring', 'Summer', 'Fall', 'Winter']
seasonal_forecast_sorted = seasonal_forecast.set_index('season').reindex(season_order).reset_index()
seasonal_forecast_sorted = seasonal_forecast_sorted.dropna()

bars = ax4.bar(seasonal_forecast_sorted['season'], 
               seasonal_forecast_sorted['total_forecast_demand'],
               color=['#90EE90', '#FFD700', '#FF8C00', '#87CEEB'])

ax4.set_xlabel('Season', fontweight='bold')
ax4.set_ylabel('Forecast Demand ($)', fontweight='bold')
ax4.set_title('Seasonal Demand Forecast (Next 6 Months)', fontsize=14, fontweight='bold')
ax4.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax4.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
print("✓ Displayed: Seasonal Forecast")

# 5) Category Forecast
fig5, ax5 = plt.subplots(figsize=(12, 6))

ax5.barh(category_forecast['category'], category_forecast['total_forecast_demand'], 
         color=colors[:len(category_forecast)])
ax5.set_xlabel('Forecast Demand ($)', fontweight='bold')
ax5.set_title('Category Demand Forecast (Next 6 Months)', fontsize=14, fontweight='bold')
ax5.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax5.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Displayed: Category Forecast")

# 6) ABC Forecast Breakdown
fig6, ax6 = plt.subplots(figsize=(10, 6))

abc_colors_map = {'A': '#ff9999', 'B': '#66b3ff', 'C': '#99ff99'}
colors_abc = [abc_colors_map.get(cat, '#cccccc') for cat in abc_forecast['abc_category']]

ax6.bar(abc_forecast['abc_category'], abc_forecast['total_forecast_demand'], color=colors_abc)
ax6.set_xlabel('ABC Category', fontweight='bold')
ax6.set_ylabel('Forecast Demand ($)', fontweight='bold')
ax6.set_title('ABC Category Forecast (Next 6 Months)', fontsize=14, fontweight='bold')
ax6.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax6.grid(axis='y', alpha=0.3)

total_forecast = abc_forecast['total_forecast_demand'].sum()
for i, (cat, val) in enumerate(zip(abc_forecast['abc_category'], abc_forecast['total_forecast_demand'])):
    pct = (val / total_forecast) * 100
    ax6.text(i, val, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
print("✓ Displayed: ABC Forecast")

# 7) Top 10 Forecast Products
fig7, ax7 = plt.subplots(figsize=(12, 8))

top_10_forecast = product_forecast.head(10).copy()
bar_colors = [abc_colors_map.get(abc, '#cccccc') for abc in top_10_forecast['abc_category']]

ax7.barh(range(len(top_10_forecast)), top_10_forecast['total_forecast_demand'], color=bar_colors)
ax7.set_yticks(range(len(top_10_forecast)))
ax7.set_yticklabels([f"{row['product_id']} ({row['abc_category']})" 
                      for _, row in top_10_forecast.iterrows()])
ax7.set_xlabel('Forecast Demand ($)', fontweight='bold')
ax7.set_title('Top 10 Forecasted Products (Next 6 Months)', fontsize=14, fontweight='bold')
ax7.xaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
ax7.grid(axis='x', alpha=0.3)
ax7.invert_yaxis()

plt.tight_layout()
plt.show()
print("✓ Displayed: Top 10 Forecast Products")

# 8) Monthly Category Trend
fig8, ax8 = plt.subplots(figsize=(14, 7))

monthly_category = future_df.groupby(['date', 'category'])['predicted_demand'].sum().reset_index()

for category in monthly_category['category'].unique():
    cat_data = monthly_category[monthly_category['category'] == category]
    ax8.plot(cat_data['date'], cat_data['predicted_demand'], marker='o', label=category, linewidth=2)

ax8.set_xlabel('Month', fontweight='bold')
ax8.set_ylabel('Forecast Demand ($)', fontweight='bold')
ax8.set_title('Monthly Forecast Trend by Category (Next 6 Months)', fontsize=14, fontweight='bold')
ax8.legend(loc='best')
ax8.grid(alpha=0.3)
ax8.yaxis.set_major_formatter(ticker.StrMethodFormatter('${x:,.0f}'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
print("✓ Displayed: Monthly Category Trend")

# 9) Actual vs Predicted
fig9, ax9 = plt.subplots(figsize=(12, 8))

sample_size = min(500, len(y_sku_test))
sample_indices = np.random.choice(len(y_sku_test), sample_size, replace=False)

if best_model_sku['name'] == 'Linear Regression (PRIMARY)':
    y_pred_best = y_sku_pred_lin[sample_indices]
elif best_model_sku['name'] == 'Ensemble (Average)':
    y_pred_best = y_sku_pred_ensemble[sample_indices]
elif 'XGBoost' in best_model_sku['name']:
    y_pred_best = y_sku_pred_xgb[sample_indices]
else:
    y_pred_best = y_sku_pred_rf[sample_indices]

y_test_sample = y_sku_test.values[sample_indices]

ax9.scatter(y_test_sample, y_pred_best, alpha=0.5, s=30)
ax9.plot([y_test_sample.min(), y_test_sample.max()], 
         [y_test_sample.min(), y_test_sample.max()], 
         'r--', lw=2, label='Perfect Prediction')

ax9.set_xlabel('Actual Demand ($)', fontweight='bold')
ax9.set_ylabel('Predicted Demand ($)', fontweight='bold')
ax9.set_title(f'Actual vs Predicted - {best_model_sku["name"]} (Test Set Sample)', 
              fontsize=14, fontweight='bold')
ax9.legend()
ax9.grid(alpha=0.3)

textstr = f"MAE: ${best_model_sku['mae']:.2f}\nRMSE: ${best_model_sku['rmse']:.2f}\nMAPE: {best_model_sku['mape']:.2f}%"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax9.text(0.05, 0.95, textstr, transform=ax9.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
print("✓ Displayed: Actual vs Predicted")

# 10) Feature Importance
if hasattr(xgb_sku_model, 'feature_importances_'):
    fig10, ax10 = plt.subplots(figsize=(10, 8))
    
    feature_importance_df = pd.DataFrame({
        'feature': sku_feature_cols,
        'importance': xgb_sku_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    ax10.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='steelblue')
    ax10.set_xlabel('Importance', fontweight='bold')
    ax10.set_title('Top 15 Feature Importance - Enhanced XGBoost Model', fontsize=14, fontweight='bold')
    ax10.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Feature Importance")

print("\n✓ All visualizations displayed!")

# =============================================================================
# CSV EXPORTS FOR POWER BI
# =============================================================================
print("\nGenerating CSV files for Power BI...")

# 1) Historical vs Forecast Comparison
hist_vs_forecast = pd.DataFrame({
    'product_id': top_hist_ids
})

hist_vs_forecast = hist_vs_forecast.merge(
    top_hist_products[['product_id', 'total_hist_demand', 'category', 'abc_category']], 
    on='product_id', 
    how='left'
)
hist_vs_forecast = hist_vs_forecast.merge(
    top_forecast_matched.reset_index()[['product_id', 'total_forecast_demand']], 
    on='product_id', 
    how='left'
)
hist_vs_forecast['total_forecast_demand'] = hist_vs_forecast['total_forecast_demand'].fillna(0)

hist_vs_forecast.to_csv(os.path.join(export_dir, "viz_hist_vs_forecast_comparison.csv"), index=False)
print("✓ Saved: viz_hist_vs_forecast_comparison.csv")

# 2) Monthly Forecast Trend by Category
monthly_category.rename(columns={'predicted_demand': 'forecast_demand'}, inplace=True)
monthly_category.to_csv(os.path.join(export_dir, "viz_monthly_category_trend.csv"), index=False)
print("✓ Saved: viz_monthly_category_trend.csv")

# 3) Actual vs Predicted
actual_vs_predicted = pd.DataFrame({
    'actual_demand': y_sku_test.values,
    'predicted_demand_xgb': y_sku_pred_xgb,
    'predicted_demand_rf': y_sku_pred_rf,
    'predicted_demand_linear': y_sku_pred_lin,
    'predicted_demand_ensemble': y_sku_pred_ensemble,
    'date': dates_sku_test.values
})
actual_vs_predicted.to_csv(os.path.join(export_dir, "viz_actual_vs_predicted.csv"), index=False)
print("✓ Saved: viz_actual_vs_predicted.csv")

# 4) Feature Importance
if hasattr(xgb_sku_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': sku_feature_cols,
        'importance': xgb_sku_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv(os.path.join(export_dir, "viz_feature_importance.csv"), index=False)
    print("✓ Saved: viz_feature_importance.csv")

# 5) ABC Summary with Percentages
abc_summary_pct = abc_summary.copy()
abc_summary_pct['sales_percentage'] = abc_summary_pct['sales_pct']
abc_summary_pct['product_percentage'] = (abc_summary_pct['product_id'] / 
                                          abc_summary_pct['product_id'].sum() * 100)
abc_summary_pct.to_csv(os.path.join(export_dir, "viz_abc_summary.csv"), index=False)
print("✓ Saved: viz_abc_summary.csv")

# 6) ABC Forecast with Percentages
abc_forecast_pct = abc_forecast.copy()
total_abc_forecast = abc_forecast_pct['total_forecast_demand'].sum()
abc_forecast_pct['forecast_percentage'] = (abc_forecast_pct['total_forecast_demand'] / 
                                            total_abc_forecast * 100)
abc_forecast_pct.to_csv(os.path.join(export_dir, "viz_abc_forecast_detailed.csv"), index=False)
print("✓ Saved: viz_abc_forecast_detailed.csv")

# 7) Category Forecast with Percentages
category_forecast_pct = category_forecast.copy()
total_cat_forecast = category_forecast_pct['total_forecast_demand'].sum()
category_forecast_pct['forecast_percentage'] = (category_forecast_pct['total_forecast_demand'] / 
                                                 total_cat_forecast * 100)
category_forecast_pct.to_csv(os.path.join(export_dir, "viz_category_forecast_detailed.csv"), index=False)
print("✓ Saved: viz_category_forecast_detailed.csv")

# 8) Seasonal Forecast with Percentages
seasonal_forecast_pct = seasonal_forecast.copy()
total_seasonal_forecast = seasonal_forecast_pct['total_forecast_demand'].sum()
seasonal_forecast_pct['forecast_percentage'] = (seasonal_forecast_pct['total_forecast_demand'] / 
                                                 total_seasonal_forecast * 100)

season_order_map = {'Spring': 1, 'Summer': 2, 'Fall': 3, 'Winter': 4}
seasonal_forecast_pct['season_order'] = seasonal_forecast_pct['season'].map(season_order_map)
seasonal_forecast_pct = seasonal_forecast_pct.sort_values('season_order')

seasonal_forecast_pct.to_csv(os.path.join(export_dir, "viz_seasonal_forecast_detailed.csv"), index=False)
print("✓ Saved: viz_seasonal_forecast_detailed.csv")

# 9) Top 10 Forecast Products with Rankings
top_10_forecast_viz = product_forecast.head(10).copy()
top_10_forecast_viz['rank'] = range(1, len(top_10_forecast_viz) + 1)
top_10_forecast_viz.to_csv(os.path.join(export_dir, "viz_top10_forecast_products.csv"), index=False)
print("✓ Saved: viz_top10_forecast_products.csv")

# 10) Time Series Historical Demand
historical_product_timeseries = demand_df[demand_df['product_id'].isin(top_hist_ids)].copy()
historical_product_timeseries = historical_product_timeseries[['date', 'product_id', 'category', 
                                                                 'abc_category', 'daily_sales']]
historical_product_timeseries.to_csv(os.path.join(export_dir, "viz_historical_timeseries.csv"), index=False)
print("✓ Saved: viz_historical_timeseries.csv")

# 11) Combined Historical + Forecast Timeline
historical_timeline = demand_df.groupby('date')['daily_sales'].sum().reset_index()
historical_timeline.rename(columns={'daily_sales': 'demand'}, inplace=True)
historical_timeline['type'] = 'Historical'

forecast_timeline = future_df.groupby('date')['predicted_demand'].sum().reset_index()
forecast_timeline.rename(columns={'predicted_demand': 'demand'}, inplace=True)
forecast_timeline['type'] = 'Forecast'

combined_timeline = pd.concat([historical_timeline, forecast_timeline], ignore_index=True)
combined_timeline = combined_timeline.sort_values('date')
combined_timeline.to_csv(os.path.join(export_dir, "viz_combined_timeline.csv"), index=False)
print("✓ Saved: viz_combined_timeline.csv")

# 12) Product Performance Matrix
product_performance = demand_df.groupby(['product_id', 'category', 'abc_category']).agg({
    'daily_sales': ['sum', 'mean', 'std'],
    'num_transactions': 'sum'
}).reset_index()

product_performance.columns = ['product_id', 'category', 'abc_category', 
                                'total_sales', 'avg_sales', 'sales_volatility', 'total_transactions']

product_performance = product_performance.merge(
    product_forecast[['product_id', 'total_forecast_demand']], 
    on='product_id', 
    how='left'
)
product_performance['total_forecast_demand'] = product_performance['total_forecast_demand'].fillna(0)

product_performance.to_csv(os.path.join(export_dir, "viz_product_performance_matrix.csv"), index=False)
print("✓ Saved: viz_product_performance_matrix.csv")

print("\n✓ All CSV files for Power BI generated successfully!")

print("\n" + "=" * 100)
print("ENHANCED FORECASTING COMPLETE")
print("=" * 100)
