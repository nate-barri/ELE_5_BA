import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df = df.sort_values('purchase_date').reset_index(drop=True)

print("="*100)
print(" " * 30 + "SALES & SEASONALITY FORECASTING")
print("="*100)
print(f"\nDataset: {len(df):,} transactions")
print(f"Date Range: {df['purchase_date'].min().strftime('%Y-%m-%d')} to {df['purchase_date'].max().strftime('%Y-%m-%d')}")
print(f"Total Sales: ${df['total_sales'].sum():,.2f}")

# ============================================================================
# PART 1: HISTORICAL SEASONALITY ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("PART 1: HISTORICAL SEASONALITY PATTERNS")
print("="*100)

# Aggregate by day
daily_sales = df.groupby('purchase_date')['total_sales'].sum().reset_index()
daily_sales.columns = ['date', 'total_sales']

# Create time features
daily_sales['year'] = daily_sales['date'].dt.year
daily_sales['month'] = daily_sales['date'].dt.month
daily_sales['day_of_week'] = daily_sales['date'].dt.dayofweek
daily_sales['week_of_year'] = daily_sales['date'].dt.isocalendar().week
daily_sales['quarter'] = daily_sales['date'].dt.quarter
daily_sales['day_of_year'] = daily_sales['date'].dt.dayofyear

# Month names for display
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# ============================================================================
# MONTHLY SEASONALITY
# ============================================================================
print("\n" + "-"*100)
print("MONTHLY SEASONALITY PATTERN")
print("-"*100)

monthly_stats = df.groupby(df['purchase_date'].dt.month).agg({
    'total_sales': ['sum', 'mean', 'count', 'std']
}).round(2)

monthly_stats.columns = ['Total_Sales', 'Avg_Per_Transaction', 'Num_Transactions', 'Std_Dev']
monthly_stats.index = month_names

# Calculate seasonality index (average = 100)
monthly_stats['Seasonality_Index'] = (
    (monthly_stats['Total_Sales'] / monthly_stats['Total_Sales'].mean()) * 100
).round(1)

print("\n", monthly_stats)

# Identify patterns
peak_month = monthly_stats['Seasonality_Index'].idxmax()
low_month = monthly_stats['Seasonality_Index'].idxmin()
peak_index = monthly_stats.loc[peak_month, 'Seasonality_Index']
low_index = monthly_stats.loc[low_month, 'Seasonality_Index']

print(f"\n📈 PEAK MONTH: {peak_month} (Index: {peak_index}) - {(peak_index-100):.1f}% above average")
print(f"📉 LOW MONTH: {low_month} (Index: {low_index}) - {(100-low_index):.1f}% below average")
print(f"💫 SEASONALITY STRENGTH: {(peak_index/low_index - 1)*100:.1f}% variation between peak and low")

# ============================================================================
# QUARTERLY SEASONALITY
# ============================================================================
print("\n" + "-"*100)
print("QUARTERLY SEASONALITY PATTERN")
print("-"*100)

quarterly_stats = df.groupby(df['purchase_date'].dt.quarter).agg({
    'total_sales': ['sum', 'mean', 'count']
}).round(2)

quarterly_stats.columns = ['Total_Sales', 'Avg_Per_Transaction', 'Num_Transactions']
quarterly_stats.index = ['Q1 (Jan-Mar)', 'Q2 (Apr-Jun)', 'Q3 (Jul-Sep)', 'Q4 (Oct-Dec)']
quarterly_stats['Seasonality_Index'] = (
    (quarterly_stats['Total_Sales'] / quarterly_stats['Total_Sales'].mean()) * 100
).round(1)

print("\n", quarterly_stats)

# ============================================================================
# DAY OF WEEK PATTERN
# ============================================================================
print("\n" + "-"*100)
print("DAY OF WEEK PATTERN")
print("-"*100)

dow_stats = df.groupby(df['purchase_date'].dt.dayofweek).agg({
    'total_sales': ['sum', 'mean', 'count']
}).round(2)

dow_stats.columns = ['Total_Sales', 'Avg_Per_Transaction', 'Num_Transactions']
dow_stats.index = dow_names
dow_stats['Seasonality_Index'] = (
    (dow_stats['Total_Sales'] / dow_stats['Total_Sales'].mean()) * 100
).round(1)

print("\n", dow_stats)

best_day = dow_stats['Seasonality_Index'].idxmax()
worst_day = dow_stats['Seasonality_Index'].idxmin()
print(f"\n📊 BEST DAY: {best_day} ({dow_stats.loc[best_day, 'Seasonality_Index']:.1f} index)")
print(f"📊 WORST DAY: {worst_day} ({dow_stats.loc[worst_day, 'Seasonality_Index']:.1f} index)")

# ============================================================================
# PART 2: TIME SERIES FORECASTING MODEL
# ============================================================================
print("\n" + "="*100)
print("PART 2: BUILDING TIME SERIES FORECASTING MODEL")
print("="*100)

# Aggregate to daily level for forecasting
daily_agg = df.groupby('purchase_date').agg({
    'total_sales': 'sum',
    'product_id': 'count'  # number of transactions
}).reset_index()

daily_agg.columns = ['date', 'total_sales', 'num_transactions']

# Create full date range (fill missing dates)
date_range = pd.date_range(start=daily_agg['date'].min(), end=daily_agg['date'].max(), freq='D')
daily_full = pd.DataFrame({'date': date_range})
daily_full = daily_full.merge(daily_agg, on='date', how='left')
daily_full['total_sales'] = daily_full['total_sales'].fillna(0)
daily_full['num_transactions'] = daily_full['num_transactions'].fillna(0)

# Feature engineering for time series
daily_full['year'] = daily_full['date'].dt.year
daily_full['month'] = daily_full['date'].dt.month
daily_full['day'] = daily_full['date'].dt.day
daily_full['day_of_week'] = daily_full['date'].dt.dayofweek
daily_full['day_of_year'] = daily_full['date'].dt.dayofyear
daily_full['week_of_year'] = daily_full['date'].dt.isocalendar().week
daily_full['quarter'] = daily_full['date'].dt.quarter
daily_full['is_weekend'] = daily_full['day_of_week'].isin([5, 6]).astype(int)
daily_full['is_month_start'] = daily_full['date'].dt.is_month_start.astype(int)
daily_full['is_month_end'] = daily_full['date'].dt.is_month_end.astype(int)

# Cyclical features (CRITICAL for seasonality)
daily_full['month_sin'] = np.sin(2 * np.pi * daily_full['month'] / 12)
daily_full['month_cos'] = np.cos(2 * np.pi * daily_full['month'] / 12)
daily_full['day_of_week_sin'] = np.sin(2 * np.pi * daily_full['day_of_week'] / 7)
daily_full['day_of_week_cos'] = np.cos(2 * np.pi * daily_full['day_of_week'] / 7)
daily_full['day_of_year_sin'] = np.sin(2 * np.pi * daily_full['day_of_year'] / 365)
daily_full['day_of_year_cos'] = np.cos(2 * np.pi * daily_full['day_of_year'] / 365)

# Lag features (past sales)
for lag in [1, 7, 14, 30]:
    daily_full[f'sales_lag_{lag}'] = daily_full['total_sales'].shift(lag).fillna(0)

# Rolling averages
for window in [7, 14, 30]:
    daily_full[f'sales_ma_{window}'] = daily_full['total_sales'].rolling(
        window=window, min_periods=1
    ).mean()

# Trend
daily_full['trend'] = range(len(daily_full))

# Prepare features
feature_cols = [
    'month', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
    'is_weekend', 'is_month_start', 'is_month_end',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos',
    'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_30',
    'sales_ma_7', 'sales_ma_14', 'sales_ma_30',
    'trend'
]

X = daily_full[feature_cols].copy()
y = daily_full['total_sales'].copy()

# Train-test split (80-20, preserving time order)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

train_dates = daily_full['date'].iloc[:split_idx]
test_dates = daily_full['date'].iloc[split_idx:]

print(f"\nTraining Period: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')}")
print(f"Testing Period: {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Gradient Boosting model
print("\nTraining Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Metrics
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mape_test = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100

print("\n" + "-"*100)
print("MODEL PERFORMANCE")
print("-"*100)
print(f"Training R²:   {r2_train:.4f}")
print(f"Testing R²:    {r2_test:.4f} {'✓ EXCELLENT' if r2_test > 0.7 else '✓ GOOD' if r2_test > 0.5 else '⚠ MODERATE'}")
print(f"MAE:           ${mae_test:.2f}")
print(f"RMSE:          ${rmse_test:.2f}")
print(f"MAPE:          {mape_test:.2f}%")
print(f"\nAvg Daily Sales (Test): ${y_test.mean():.2f}")
print(f"Prediction Accuracy:     {(1 - mape_test/100)*100:.1f}%")

# ============================================================================
# PART 3: FUTURE FORECASTING
# ============================================================================
print("\n" + "="*100)
print("PART 3: FUTURE SALES FORECAST")
print("="*100)

# Generate future dates (next 90 days)
last_date = daily_full['date'].max()
future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=90, freq='D')

# Create future dataframe
future_df = pd.DataFrame({'date': future_dates})

# Engineer features for future dates
future_df['year'] = future_df['date'].dt.year
future_df['month'] = future_df['date'].dt.month
future_df['day'] = future_df['date'].dt.day
future_df['day_of_week'] = future_df['date'].dt.dayofweek
future_df['day_of_year'] = future_df['date'].dt.dayofyear
future_df['week_of_year'] = future_df['date'].dt.isocalendar().week
future_df['quarter'] = future_df['date'].dt.quarter
future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
future_df['is_month_start'] = future_df['date'].dt.is_month_start.astype(int)
future_df['is_month_end'] = future_df['date'].dt.is_month_end.astype(int)

# Cyclical features
future_df['month_sin'] = np.sin(2 * np.pi * future_df['month'] / 12)
future_df['month_cos'] = np.cos(2 * np.pi * future_df['month'] / 12)
future_df['day_of_week_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
future_df['day_of_week_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
future_df['day_of_year_sin'] = np.sin(2 * np.pi * future_df['day_of_year'] / 365)
future_df['day_of_year_cos'] = np.cos(2 * np.pi * future_df['day_of_year'] / 365)

# For lag features, use recent history
recent_sales = daily_full['total_sales'].tail(30).values
for i in range(len(future_df)):
    if i == 0:
        future_df.loc[i, 'sales_lag_1'] = recent_sales[-1]
        future_df.loc[i, 'sales_lag_7'] = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales.mean()
        future_df.loc[i, 'sales_lag_14'] = recent_sales[-14] if len(recent_sales) >= 14 else recent_sales.mean()
        future_df.loc[i, 'sales_lag_30'] = recent_sales[-30] if len(recent_sales) >= 30 else recent_sales.mean()
        future_df.loc[i, 'sales_ma_7'] = recent_sales[-7:].mean()
        future_df.loc[i, 'sales_ma_14'] = recent_sales[-14:].mean()
        future_df.loc[i, 'sales_ma_30'] = recent_sales.mean()
    else:
        # Use previously predicted values for lags
        if i >= 1:
            future_df.loc[i, 'sales_lag_1'] = future_df.loc[i-1, 'predicted_sales'] if i > 0 and 'predicted_sales' in future_df.columns else recent_sales[-1]
        if i >= 7:
            future_df.loc[i, 'sales_lag_7'] = future_df.loc[i-7, 'predicted_sales'] if 'predicted_sales' in future_df.columns else recent_sales.mean()
        else:
            future_df.loc[i, 'sales_lag_7'] = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales.mean()
        
        future_df.loc[i, 'sales_lag_14'] = recent_sales.mean()
        future_df.loc[i, 'sales_lag_30'] = recent_sales.mean()
        future_df.loc[i, 'sales_ma_7'] = recent_sales[-7:].mean()
        future_df.loc[i, 'sales_ma_14'] = recent_sales[-14:].mean()
        future_df.loc[i, 'sales_ma_30'] = recent_sales.mean()

future_df['trend'] = range(len(daily_full), len(daily_full) + len(future_df))

# Make predictions
X_future = future_df[feature_cols]
X_future_scaled = scaler.transform(X_future)
future_df['predicted_sales'] = model.predict(X_future_scaled)

# Aggregate by month
future_monthly = future_df.groupby(future_df['date'].dt.month).agg({
    'predicted_sales': 'sum',
    'date': 'count'
}).round(2)
future_monthly.columns = ['Forecasted_Sales', 'Num_Days']
future_monthly['Avg_Daily_Sales'] = (future_monthly['Forecasted_Sales'] / future_monthly['Num_Days']).round(2)
future_monthly.index = [month_names[i-1] for i in future_monthly.index]

print("\n" + "-"*100)
print("NEXT 90 DAYS FORECAST BY MONTH")
print("-"*100)
print("\n", future_monthly)

# Calculate vs historical average
historical_monthly_avg = df.groupby(df['purchase_date'].dt.month)['total_sales'].sum().mean()
print(f"\nHistorical Monthly Average: ${historical_monthly_avg:,.2f}")

for month in future_monthly.index:
    forecast = future_monthly.loc[month, 'Forecasted_Sales']
    diff_pct = ((forecast / historical_monthly_avg) - 1) * 100
    trend_symbol = "📈" if diff_pct > 0 else "📉"
    print(f"  {month}: ${forecast:,.2f} ({trend_symbol} {diff_pct:+.1f}% vs avg)")

# ============================================================================
# PART 4: CATEGORY-LEVEL SEASONALITY FORECAST
# ============================================================================
print("\n" + "="*100)
print("PART 4: CATEGORY-LEVEL SEASONALITY FORECAST")
print("="*100)

category_seasonal = df.groupby([df['purchase_date'].dt.month, 'category'])['total_sales'].sum().unstack(fill_value=0)
category_seasonal.index = month_names
category_seasonal_pct = (category_seasonal / category_seasonal.sum(axis=1).values.reshape(-1, 1) * 100).round(1)

print("\nCATEGORY CONTRIBUTION BY MONTH (% of monthly sales):")
print(category_seasonal_pct)

print("\nCATEGORY FORECAST FOR NEXT 3 MONTHS:")
current_month_idx = datetime.now().month
for i in range(3):
    future_month_idx = ((current_month_idx + i - 1) % 12) + 1
    month_name = month_names[future_month_idx - 1]
    
    print(f"\n{month_name}:")
    for cat in category_seasonal.columns:
        historical_avg = category_seasonal.loc[month_name, cat]
        contribution_pct = category_seasonal_pct.loc[month_name, cat]
        print(f"  {cat:15s}: ${historical_avg:8,.2f} ({contribution_pct:5.1f}% of month)")

# ============================================================================
# VISUALIZATION DASHBOARD
# ============================================================================
print("\n" + "="*100)
print("GENERATING FORECAST VISUALIZATION DASHBOARD")
print("="*100)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Historical + Forecast Time Series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(train_dates, y_train, label='Training Data', color='blue', alpha=0.6, linewidth=1.5)
ax1.plot(test_dates, y_test, label='Test Data (Actual)', color='green', alpha=0.6, linewidth=1.5)
ax1.plot(test_dates, y_pred_test, label='Test Data (Predicted)', color='red', linestyle='--', linewidth=2)
ax1.plot(future_df['date'], future_df['predicted_sales'], label='Future Forecast', color='orange', linewidth=2.5)
ax1.axvline(test_dates.min(), color='black', linestyle=':', linewidth=2, alpha=0.5, label='Train/Test Split')
ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
ax1.set_ylabel('Daily Sales ($)', fontsize=11, fontweight='bold')
ax1.set_title(f'Sales Forecast - Historical & Future (R² = {r2_test:.3f})', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Monthly Seasonality
ax2 = fig.add_subplot(gs[1, 0])
monthly_stats['Seasonality_Index'].plot(kind='bar', ax=ax2, color='skyblue', edgecolor='navy')
ax2.axhline(100, color='red', linestyle='--', linewidth=2, label='Average (100)')
ax2.set_xlabel('Month', fontsize=10, fontweight='bold')
ax2.set_ylabel('Seasonality Index', fontsize=10, fontweight='bold')
ax2.set_title('Monthly Seasonality Pattern', fontsize=11, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Day of Week Pattern
ax3 = fig.add_subplot(gs[1, 1])
dow_stats['Seasonality_Index'].plot(kind='bar', ax=ax3, color='lightgreen', edgecolor='darkgreen')
ax3.axhline(100, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('Day of Week', fontsize=10, fontweight='bold')
ax3.set_ylabel('Seasonality Index', fontsize=10, fontweight='bold')
ax3.set_title('Day of Week Pattern', fontsize=11, fontweight='bold')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Quarterly Trend
ax4 = fig.add_subplot(gs[1, 2])
quarterly_stats['Seasonality_Index'].plot(kind='bar', ax=ax4, color='coral', edgecolor='darkred')
ax4.axhline(100, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Quarter', fontsize=10, fontweight='bold')
ax4.set_ylabel('Seasonality Index', fontsize=10, fontweight='bold')
ax4.set_title('Quarterly Seasonality', fontsize=11, fontweight='bold')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Actual vs Predicted (Test Set)
ax5 = fig.add_subplot(gs[2, 0])
ax5.scatter(y_test, y_pred_test, alpha=0.5, s=50, c='steelblue', edgecolors='navy')
ax5.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax5.set_xlabel('Actual Sales ($)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Predicted Sales ($)', fontsize=10, fontweight='bold')
ax5.set_title(f'Prediction Accuracy\nR² = {r2_test:.3f}, MAPE = {mape_test:.1f}%', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Category Seasonality Heatmap
ax6 = fig.add_subplot(gs[2, 1:])
sns.heatmap(category_seasonal_pct.T, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax6, cbar_kws={'label': '% of Monthly Sales'})
ax6.set_xlabel('Month', fontsize=10, fontweight='bold')
ax6.set_ylabel('Category', fontsize=10, fontweight='bold')
ax6.set_title('Category Contribution by Month (%)', fontsize=11, fontweight='bold')

plt.suptitle('Sales & Seasonality Forecasting Dashboard', fontsize=16, fontweight='bold', y=0.995)
print("\n✓ Displaying interactive dashboard...")
plt.show()  # This will display the plots instead of saving

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*100)
print("FORECAST SUMMARY")
print("="*100)

print(f"\n📊 MODEL QUALITY: R² = {r2_test:.3f}, MAPE = {mape_test:.1f}%")
print(f"\n📅 SEASONALITY INSIGHTS:")
print(f"   Peak Month: {peak_month} ({peak_index:.0f} index)")
print(f"   Low Month: {low_month} ({low_index:.0f} index)")
print(f"   Best Day: {best_day}")
print(f"   Worst Day: {worst_day}")

print(f"\n📈 NEXT 90 DAYS OUTLOOK:")
total_forecast = future_monthly['Forecasted_Sales'].sum()
avg_daily_forecast = total_forecast / 90
print(f"   Total Forecasted Sales: ${total_forecast:,.2f}")
print(f"   Average Daily Sales: ${avg_daily_forecast:,.2f}")

print(f"\n🎯 KEY TAKEAWAYS:")
print(f"   1. Your sales show {(peak_index/low_index - 1)*100:.1f}% seasonal variation")
print(f"   2. {peak_month} is consistently your best month")
print(f"   3. {best_day} generates {dow_stats.loc[best_day, 'Seasonality_Index']:.0f}% of average daily sales")
print(f"   4. Model can predict sales with {(1-mape_test/100)*100:.1f}% accuracy")

print("\n" + "="*100)
print("FORECASTING COMPLETE")
print("="*100)