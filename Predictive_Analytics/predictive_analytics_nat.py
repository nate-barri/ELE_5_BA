import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df = df.sort_values('purchase_date').reset_index(drop=True)
raw_last_date = df['purchase_date'].max()

print("="*100)
print(" " * 25 + "IMPROVED SALES FORECASTING ANALYSIS")
print("="*100)

# ============================================================================
# PART 1: DATA QUALITY DIAGNOSTICS
# ============================================================================

print("\n" + "="*100)
print("PART 1: DATA QUALITY DIAGNOSTICS")
print("="*100)

# Aggregate to daily level
daily_agg = df.groupby('purchase_date').agg({
    'total_sales': 'sum',
    'product_id': 'count'
}).reset_index()
daily_agg.columns = ['date', 'total_sales', 'num_transactions']

# Create full date range
date_range = pd.date_range(start=daily_agg['date'].min(), end=daily_agg['date'].max(), freq='D')
daily_full = pd.DataFrame({'date': date_range})
daily_full = daily_full.merge(daily_agg, on='date', how='left')
daily_full['total_sales'] = daily_full['total_sales'].fillna(0)
daily_full['num_transactions'] = daily_full['num_transactions'].fillna(0)
daily_full_all = daily_full.copy()

print(f"\nDataset Overview:")
print(f"  Total Days: {len(daily_full)}")
print(f"  Days with Sales: {(daily_full['total_sales'] > 0).sum()}")
print(f"  Days with Zero Sales: {(daily_full['total_sales'] == 0).sum()}")
print(f"  Date Range: {daily_full['date'].min().date()} to {daily_full['date'].max().date()}")
print(f"  Total Sales: ${daily_full['total_sales'].sum():,.2f}")
print(f"  Avg Daily Sales: ${daily_full['total_sales'].mean():.2f}")
print(f"  Median Daily: ${daily_full['total_sales'].median():.2f}")
print(f"  Std Dev: ${daily_full['total_sales'].std():.2f}")

# Check for data quality issues by period
daily_full['year_month'] = daily_full['date'].dt.to_period('M')
monthly_stats = daily_full.groupby('year_month')['total_sales'].agg(['sum', 'mean', 'count', 'std'])
monthly_stats.columns = ['Total', 'Avg_Daily', 'Days', 'Std_Dev']

print("\n" + "-"*100)
print("MONTHLY PATTERNS")
print("-"*100)
print(monthly_stats.tail(12).to_string())

# CRITICAL FIX: Remove incomplete months
last_month_days = monthly_stats.iloc[-1]['Days']
expected_days = pd.Period(daily_full['year_month'].iloc[-1], freq='M').days_in_month

if last_month_days < expected_days * 0.8:
    print(f"\n⚠️  WARNING: Last month has only {int(last_month_days)}/{expected_days} days - removing it!")
    last_complete_date = daily_full[daily_full['year_month'] != daily_full['year_month'].iloc[-1]]['date'].max()
    daily_full = daily_full[daily_full['date'] <= last_complete_date].copy()
    print(f"   New end date: {last_complete_date.date()}")
    
    daily_full['year_month'] = daily_full['date'].dt.to_period('M')
    monthly_stats = daily_full.groupby('year_month')['total_sales'].agg(['sum', 'mean', 'count', 'std'])
    monthly_stats.columns = ['Total', 'Avg_Daily', 'Days', 'Std_Dev']

# ============================================================================
# PART 2: HANDLE ZERO SALES DAYS
# ============================================================================

print("\n" + "="*100)
print("PART 2: DATA PREPROCESSING")
print("="*100)

daily_full['total_sales_original'] = daily_full['total_sales'].copy()

# IMPROVED: Handle zeros more intelligently
zero_mask = daily_full['total_sales'] == 0
if zero_mask.sum() > 0:
    print(f"Found {zero_mask.sum()} zero sales days ({zero_mask.sum()/len(daily_full)*100:.1f}%)")
    
    # Option 1: Use day-of-week specific median (better than global median)
    dow_medians = daily_full[daily_full['total_sales'] > 0].groupby(
        daily_full[daily_full['total_sales'] > 0]['date'].dt.dayofweek
    )['total_sales'].median()
    
    for idx in daily_full[zero_mask].index:
        dow = daily_full.loc[idx, 'date'].dayofweek
        daily_full.loc[idx, 'total_sales'] = dow_medians.get(dow, 
                                                               daily_full[daily_full['total_sales'] > 0]['total_sales'].median())
    
    print(f"✓ Filled zero days with day-of-week specific medians")
    print(f"  Example: Monday median=${dow_medians.get(0, 0):.2f}, Friday median=${dow_medians.get(4, 0):.2f}")

print(f"  New avg daily sales: ${daily_full['total_sales'].mean():.2f}")

# Add volatility flag for high-variance periods
daily_full['high_volatility'] = 0
rolling_std = daily_full['total_sales'].rolling(30, min_periods=1).std()
volatility_threshold = rolling_std.quantile(0.75)
daily_full.loc[rolling_std > volatility_threshold, 'high_volatility'] = 1
print(f"  Identified {daily_full['high_volatility'].sum()} high-volatility days")

# ============================================================================
# PART 3: FEATURE ENGINEERING
# ============================================================================

def create_base_features(df_input):
    """Create time-based features (no lags)"""
    df = df_input.copy()
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    
    # IMPROVED: Add more business-relevant features
    df['is_month_first_week'] = (df['day'] <= 7).astype(int)
    df['is_month_last_week'] = (df['day'] >= 22).astype(int)
    df['week_of_month'] = (df['day'] - 1) // 7 + 1
    
    # Cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    
    df['days_from_start'] = (df['date'] - df['date'].min()).dt.days
    
    return df

daily_full = create_base_features(daily_full)

# ============================================================================
# PART 4: PROPER TRAIN/TEST SPLIT
# ============================================================================

print("\n" + "="*100)
print("PART 3: TRAIN/TEST SPLIT")
print("="*100)

test_days = 90
train_df = daily_full.iloc[:-test_days].copy()
test_df = daily_full.iloc[-test_days:].copy()

print(f"\nTime-Based Split:")
print(f"  Training: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} days)")
print(f"  Testing:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} days)")

def add_lag_features(df, target_col='total_sales', lags=[1, 7, 14, 30]):
    """Add lag features with exponential weighting for recent values"""
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # IMPROVED: Add exponentially weighted moving averages (recent data matters more)
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df[target_col].rolling(window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = df[target_col].rolling(window, min_periods=1).std()
        df[f'rolling_min_{window}'] = df[target_col].rolling(window, min_periods=1).min()
        df[f'rolling_max_{window}'] = df[target_col].rolling(window, min_periods=1).max()
        # Exponential weighted mean (alpha=0.3 gives more weight to recent)
        df[f'ewm_{window}'] = df[target_col].ewm(span=window, adjust=False).mean()
    
    # Add momentum features (rate of change)
    df['momentum_7d'] = df[target_col] - df[target_col].shift(7)
    df['momentum_30d'] = df[target_col] - df[target_col].shift(30)
    
    return df

daily_full_with_lags = add_lag_features(daily_full.copy())

train_df = daily_full_with_lags.iloc[:-test_days].copy()
test_df = daily_full_with_lags.iloc[-test_days:].copy()

# Calculate seasonal patterns from TRAINING data only
train_monthly_avg = train_df.groupby('month')['total_sales'].mean()
train_dow_avg = train_df.groupby('day_of_week')['total_sales'].mean()
train_overall = train_df['total_sales'].mean()

train_df['month_pattern'] = train_df['month'].map(train_monthly_avg) / train_overall
train_df['dow_pattern'] = train_df['day_of_week'].map(train_dow_avg) / train_overall
test_df['month_pattern'] = test_df['month'].map(train_monthly_avg) / train_overall
test_df['dow_pattern'] = test_df['day_of_week'].map(train_dow_avg) / train_overall

train_df = train_df.fillna(method='bfill')
test_df = test_df.fillna(method='bfill')

# ============================================================================
# PART 5: BASELINE FORECASTS
# ============================================================================

print("\n" + "="*100)
print("PART 4: BASELINE FORECASTS")
print("="*100)

baseline_naive = test_df['total_sales'].shift(1).fillna(train_df['total_sales'].iloc[-1])
baseline_naive_mae = mean_absolute_error(test_df['total_sales'], baseline_naive)
baseline_naive_r2 = r2_score(test_df['total_sales'], baseline_naive)

baseline_dow_pred = test_df['day_of_week'].map(train_dow_avg).values
baseline_dow_mae = mean_absolute_error(test_df['total_sales'], baseline_dow_pred)
baseline_dow_r2 = r2_score(test_df['total_sales'], baseline_dow_pred)

baseline_ma7 = train_df['total_sales'].tail(7).mean()
baseline_ma7_pred = np.full(len(test_df), baseline_ma7)
baseline_ma7_mae = mean_absolute_error(test_df['total_sales'], baseline_ma7_pred)
baseline_ma7_r2 = r2_score(test_df['total_sales'], baseline_ma7_pred)

print("\nBaseline Performance:")
print(f"  Naive (Last Value):    MAE=${baseline_naive_mae:.2f},    R²={baseline_naive_r2:.4f}")
print(f"  Day-of-Week Average:   MAE=${baseline_dow_mae:.2f},    R²={baseline_dow_r2:.4f}")
print(f"  7-Day Moving Avg:      MAE=${baseline_ma7_mae:.2f},    R²={baseline_ma7_r2:.4f}")

best_baseline_mae = min(baseline_naive_mae, baseline_dow_mae, baseline_ma7_mae)
best_baseline_name = ['Naive', 'DOW', 'MA7'][[baseline_naive_mae, baseline_dow_mae, baseline_ma7_mae].index(best_baseline_mae)]
print(f"\n✓ Best baseline: {best_baseline_name} with MAE=${best_baseline_mae:.2f}")

# ============================================================================
# PART 6: MODEL TRAINING WITH IMPROVED EVALUATION
# ============================================================================

print("\n" + "="*100)
print("PART 5: MODEL TRAINING WITH COMPREHENSIVE EVALUATION")
print("="*100)

feature_cols = [
    'month', 'day_of_week', 'quarter', 'is_weekend',
    'is_month_first_week', 'is_month_last_week', 'week_of_month',
    'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
    'day_of_year_sin', 'day_of_year_cos', 'week_sin', 'week_cos',
    'days_from_start', 'month_pattern', 'dow_pattern',
    'lag_1', 'lag_7', 'lag_14', 'lag_30',
    'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30',
    'rolling_std_7', 'rolling_std_14', 'rolling_std_30',
    'rolling_min_7', 'rolling_min_14', 'rolling_min_30',
    'rolling_max_7', 'rolling_max_14', 'rolling_max_30',
    'ewm_7', 'ewm_14', 'ewm_30',
    'momentum_7d', 'momentum_30d'
]

X_train = train_df[feature_cols]
y_train = train_df['total_sales']
X_test = test_df[feature_cols]
y_test = test_df['total_sales']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("  Training Ensemble components...")
base_models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=500,  # Increased from 300
        max_depth=15,      # Increased from 10
        min_samples_split=5,  # Decreased from 10
        min_samples_leaf=2,   # Decreased from 5
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,  # Increased from 150
        learning_rate=0.03,  # Decreased for better generalization
        max_depth=7,      # Increased from 5
        min_samples_split=10,  # Decreased from 15
        min_samples_leaf=4,    # Decreased from 8
        subsample=0.8,    # Increased from 0.7
        random_state=42
    ),
    'Ridge': Ridge(alpha=1.0),  # Decreased from 10.0 for less regularization
    'Lasso': Lasso(alpha=0.5)   # Added for feature selection
}

ensemble_predictions = []
for name, model in base_models.items():
    print(f"    - Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    ensemble_predictions.append(y_pred)

print(f"  Creating Ensemble prediction...")
ensemble_pred = np.maximum(np.mean(ensemble_predictions, axis=0), 0)

# ============================================================================
# COMPREHENSIVE EVALUATION METRICS
# ============================================================================

def calculate_comprehensive_metrics(y_true, y_pred, baseline_mae, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Percentage metrics
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100
    
    # Scaled metrics
    mase = mae / baseline_mae  # Mean Absolute Scaled Error
    
    # Directional accuracy
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    # Bias metrics
    mean_error = np.mean(y_pred - y_true)
    mean_percentage_error = np.mean((y_pred - y_true) / (y_true + 1e-10)) * 100
    
    # Coverage metrics (within X% of actual)
    within_5pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) <= 0.05) * 100
    within_10pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) <= 0.10) * 100
    within_20pct = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)) <= 0.20) * 100
    
    # Residual analysis
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    residual_skew = pd.Series(residuals).skew()
    residual_kurt = pd.Series(residuals).kurtosis()
    
    # Threshold-based accuracy (for business decisions)
    threshold = np.median(y_true)
    actual_above = y_true >= threshold
    pred_above = y_pred >= threshold
    threshold_accuracy = np.mean(actual_above == pred_above) * 100
    
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'sMAPE': smape,
        'MASE': mase,
        'Direction_Accuracy_%': direction_accuracy,
        'Mean_Error': mean_error,
        'Mean_%_Error': mean_percentage_error,
        'Within_5%': within_5pct,
        'Within_10%': within_10pct,
        'Within_20%': within_20pct,
        'Residual_Std': residual_std,
        'Residual_Skew': residual_skew,
        'Residual_Kurt': residual_kurt,
        'Threshold_Accuracy_%': threshold_accuracy,
        'vs_Baseline_MAE': baseline_mae - mae
    }

# Calculate metrics for ensemble
ensemble_metrics = calculate_comprehensive_metrics(
    y_test, ensemble_pred, best_baseline_mae, "Ensemble"
)

# Calculate metrics for baselines
baseline_naive_metrics = calculate_comprehensive_metrics(
    y_test, baseline_naive, best_baseline_mae, "Naive"
)
baseline_dow_metrics = calculate_comprehensive_metrics(
    y_test, baseline_dow_pred, best_baseline_mae, "DOW_Avg"
)
baseline_ma7_metrics = calculate_comprehensive_metrics(
    y_test, baseline_ma7_pred, best_baseline_mae, "MA7"
)

# Combine all metrics
all_metrics = pd.DataFrame([
    ensemble_metrics,
    baseline_naive_metrics,
    baseline_dow_metrics,
    baseline_ma7_metrics
])

print("\n" + "="*100)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*100)

# Display core metrics
core_metrics = ['Model', 'R²', 'MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE']
print("\n📊 CORE PERFORMANCE METRICS:")
print("-" * 100)
print(all_metrics[core_metrics].to_string(index=False))

# Display accuracy metrics
accuracy_metrics = ['Model', 'Direction_Accuracy_%', 'Within_10%', 'Within_20%', 'Threshold_Accuracy_%']
print("\n🎯 ACCURACY METRICS:")
print("-" * 100)
print(all_metrics[accuracy_metrics].to_string(index=False))

# Display bias metrics
bias_metrics = ['Model', 'Mean_Error', 'Mean_%_Error', 'Residual_Skew']
print("\n⚖️  BIAS METRICS:")
print("-" * 100)
print(all_metrics[bias_metrics].to_string(index=False))

# Model interpretation
best_mae = ensemble_metrics['MAE']
best_r2 = ensemble_metrics['R²']

print("\n" + "="*100)
print("MODEL INTERPRETATION")
print("="*100)

print(f"\n🏆 ENSEMBLE MODEL PERFORMANCE:")
print(f"   R² = {best_r2:.4f} → Model explains {best_r2*100:.1f}% of variance")
print(f"   MAE = ${best_mae:.2f} → Average error per prediction")
print(f"   MAPE = {ensemble_metrics['MAPE']:.2f}% → Average percentage error")
print(f"   MASE = {ensemble_metrics['MASE']:.3f} → {'✅ Better' if ensemble_metrics['MASE'] < 1 else '❌ Worse'} than best baseline")

print(f"\n📈 PRACTICAL ACCURACY:")
print(f"   {ensemble_metrics['Within_10%']:.1f}% of predictions within 10% of actual")
print(f"   {ensemble_metrics['Within_20%']:.1f}% of predictions within 20% of actual")
print(f"   {ensemble_metrics['Direction_Accuracy_%']:.1f}% correct trend direction")

print(f"\n⚖️  BIAS ANALYSIS:")
if abs(ensemble_metrics['Mean_%_Error']) < 5:
    print(f"   ✅ Low bias: {ensemble_metrics['Mean_%_Error']:.2f}% (well-calibrated)")
elif ensemble_metrics['Mean_%_Error'] > 0:
    print(f"   ⚠️  Positive bias: {ensemble_metrics['Mean_%_Error']:.2f}% (tends to over-predict)")
else:
    print(f"   ⚠️  Negative bias: {ensemble_metrics['Mean_%_Error']:.2f}% (tends to under-predict)")

if abs(ensemble_metrics['Residual_Skew']) < 0.5:
    print(f"   ✅ Residuals normally distributed (skew={ensemble_metrics['Residual_Skew']:.2f})")
else:
    print(f"   ⚠️  Residuals skewed (skew={ensemble_metrics['Residual_Skew']:.2f})")

print(f"\n💡 BUSINESS INTERPRETATION:")
if best_r2 > 0.7:
    print(f"   ✅ EXCELLENT: Model is highly reliable for forecasting")
elif best_r2 > 0.5:
    print(f"   ✓ GOOD: Model captures major patterns, use with confidence intervals")
elif best_r2 > 0.3:
    print(f"   ⚠️  MODERATE: High volatility in data, use prediction ranges")
else:
    print(f"   ❌ POOR: Model performance suggests:")
    print(f"      1. Sales have high random variation not captured by temporal patterns")
    print(f"      2. Missing important external factors (promotions, events, weather)")
    print(f"      3. Customer behavior is irregular or influenced by unmeasured factors")
    
    print(f"\n🔧 RECOMMENDED ACTIONS:")
    print(f"   • Collect external data: promotions, holidays, competitor actions")
    print(f"   • Use prediction intervals (±${std_error*1.96:.2f}) instead of point estimates")
    print(f"   • Consider using weighted ensemble (recent data matters more)")
    print(f"   • Monitor actual vs forecast weekly and retrain monthly")
    
    if ensemble_metrics['Mean_%_Error'] > 10:
        print(f"   • Model over-predicts by {ensemble_metrics['Mean_%_Error']:.1f}%")
        print(f"     → Apply correction factor: forecast × {1 - ensemble_metrics['Mean_%_Error']/100:.2f}")
    
    if ensemble_metrics['Within_20%'] < 50:
        print(f"   • Only {ensemble_metrics['Within_20%']:.0f}% predictions within 20%")
        print(f"     → Use ranges: Best Case (+20%), Expected, Worst Case (-20%)")

# Add bias correction if significant
if abs(ensemble_metrics['Mean_%_Error']) > 10:
    print(f"\n⚙️  BIAS CORRECTION:")
    correction_factor = 1 - (ensemble_metrics['Mean_%_Error'] / 100)
    print(f"   Apply correction: Adjusted Forecast = Raw Forecast × {correction_factor:.3f}")
    ensemble_pred_corrected = ensemble_pred * correction_factor
    mae_corrected = mean_absolute_error(y_test, ensemble_pred_corrected)
    print(f"   Corrected MAE: ${mae_corrected:.2f} (vs ${best_mae:.2f} before)")

# Prediction intervals
residuals = y_test - ensemble_pred
std_error = np.std(residuals)
print(f"\n📊 PREDICTION UNCERTAINTY:")
print(f"  Standard Error: ±${std_error:.2f}")
print(f"  68% Confidence: ±${std_error:.2f} (1 std)")
print(f"  95% Confidence: ±${std_error*1.96:.2f} (1.96 std)")

# Export comprehensive metrics
all_metrics.to_csv("comprehensive_model_metrics.csv", index=False)
print("\n✅ Comprehensive metrics exported to 'comprehensive_model_metrics.csv'")

# Store for future predictions
models = base_models
y_pred_test = ensemble_pred
pred_std = std_error

# ============================================================================
# RESIDUAL ANALYSIS VISUALIZATION
# ============================================================================

print("\n" + "="*100)
print("GENERATING RESIDUAL ANALYSIS")
print("="*100)

fig_residuals, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Residuals over time
ax1 = axes[0, 0]
residuals_series = pd.Series(residuals, index=test_df['date'].values)
ax1.plot(residuals_series.index, residuals_series.values, marker='o', linestyle='-', alpha=0.6)
ax1.axhline(0, color='red', linestyle='--', linewidth=2)
ax1.axhline(std_error, color='orange', linestyle=':', linewidth=1, label='+1 std')
ax1.axhline(-std_error, color='orange', linestyle=':', linewidth=1, label='-1 std')
ax1.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Residual ($)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residual distribution
ax2 = axes[0, 1]
ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax2.axvline(np.mean(residuals), color='orange', linestyle='--', linewidth=2, label=f'Mean={np.mean(residuals):.2f}')
ax2.set_title(f'Residual Distribution (Skew={ensemble_metrics["Residual_Skew"]:.2f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('Residual ($)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Q-Q plot
ax3 = axes[1, 0]
from scipy import stats
stats.probplot(residuals, dist="norm", plot=ax3)
ax3.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Actual vs Predicted with perfect prediction line
ax4 = axes[1, 1]
ax4.scatter(y_test, ensemble_pred, alpha=0.6, s=50, c='steelblue', edgecolors='navy')
ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Sales ($)', fontsize=11)
ax4.set_ylabel('Predicted Sales ($)', fontsize=11)
ax4.set_title(f'Actual vs Predicted (R²={best_r2:.3f})', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# CONTINUE WITH FUTURE FORECAST (SAME AS BEFORE)
# ============================================================================

print("\n" + "="*100)
print("PART 6: 6-MONTH FUTURE FORECAST")
print("="*100)

forecast_start = pd.Timestamp("2025-09-01")
future_dates = pd.date_range(start=forecast_start, periods=180, freq='D')
future_df = pd.DataFrame({'date': future_dates})
future_df = create_base_features(future_df)

future_df['month_pattern'] = future_df['month'].map(train_monthly_avg) / train_overall
future_df['dow_pattern'] = future_df['day_of_week'].map(train_dow_avg) / train_overall

recent_sales = daily_full['total_sales'].tail(60).values.tolist()  # Increased to 60 for more history
predictions = []
prediction_lower = []
prediction_upper = []

for i in range(len(future_df)):
    # Basic lag features
    future_df.loc[i, 'lag_1'] = recent_sales[-1]
    future_df.loc[i, 'lag_7'] = recent_sales[-7] if len(recent_sales) >= 7 else recent_sales[-1]
    future_df.loc[i, 'lag_14'] = recent_sales[-14] if len(recent_sales) >= 14 else recent_sales[-1]
    future_df.loc[i, 'lag_30'] = recent_sales[-30] if len(recent_sales) >= 30 else recent_sales[-1]
    
    # Rolling statistics
    recent_7 = recent_sales[-7:]
    recent_14 = recent_sales[-14:] if len(recent_sales) >= 14 else recent_sales
    recent_30 = recent_sales[-30:] if len(recent_sales) >= 30 else recent_sales
    
    future_df.loc[i, 'rolling_mean_7'] = np.mean(recent_7)
    future_df.loc[i, 'rolling_mean_14'] = np.mean(recent_14)
    future_df.loc[i, 'rolling_mean_30'] = np.mean(recent_30)
    
    future_df.loc[i, 'rolling_std_7'] = np.std(recent_7) if len(recent_7) > 1 else 0
    future_df.loc[i, 'rolling_std_14'] = np.std(recent_14) if len(recent_14) > 1 else 0
    future_df.loc[i, 'rolling_std_30'] = np.std(recent_30) if len(recent_30) > 1 else 0
    
    # NEW: Rolling min/max features
    future_df.loc[i, 'rolling_min_7'] = np.min(recent_7)
    future_df.loc[i, 'rolling_min_14'] = np.min(recent_14)
    future_df.loc[i, 'rolling_min_30'] = np.min(recent_30)
    
    future_df.loc[i, 'rolling_max_7'] = np.max(recent_7)
    future_df.loc[i, 'rolling_max_14'] = np.max(recent_14)
    future_df.loc[i, 'rolling_max_30'] = np.max(recent_30)
    
    # NEW: Exponentially weighted mean (manually calculated)
    # EWM gives more weight to recent values
    alpha_7 = 2 / (7 + 1)
    alpha_14 = 2 / (14 + 1)
    alpha_30 = 2 / (30 + 1)
    
    ewm_7 = recent_7[0]
    for val in recent_7[1:]:
        ewm_7 = alpha_7 * val + (1 - alpha_7) * ewm_7
    future_df.loc[i, 'ewm_7'] = ewm_7
    
    ewm_14 = recent_14[0]
    for val in recent_14[1:]:
        ewm_14 = alpha_14 * val + (1 - alpha_14) * ewm_14
    future_df.loc[i, 'ewm_14'] = ewm_14
    
    ewm_30 = recent_30[0]
    for val in recent_30[1:]:
        ewm_30 = alpha_30 * val + (1 - alpha_30) * ewm_30
    future_df.loc[i, 'ewm_30'] = ewm_30
    
    # NEW: Momentum features (rate of change)
    if len(recent_sales) >= 7:
        future_df.loc[i, 'momentum_7d'] = recent_sales[-1] - recent_sales[-7]
    else:
        future_df.loc[i, 'momentum_7d'] = 0
    
    if len(recent_sales) >= 30:
        future_df.loc[i, 'momentum_30d'] = recent_sales[-1] - recent_sales[-30]
    else:
        future_df.loc[i, 'momentum_30d'] = 0
    
    # Make prediction
    X_future_row = future_df.loc[i:i, feature_cols]
    X_future_scaled = scaler.transform(X_future_row)
    
    preds = []
    for name, model in models.items():
        preds.append(model.predict(X_future_scaled)[0])
    pred = max(np.mean(preds), 0)
    
    predictions.append(pred)
    prediction_lower.append(max(pred - 1.96 * pred_std, 0))
    prediction_upper.append(pred + 1.96 * pred_std)
    recent_sales.append(pred)
    if len(recent_sales) > 60:
        recent_sales.pop(0)

future_df['predicted_sales'] = predictions
future_df['pred_lower'] = prediction_lower
future_df['pred_upper'] = prediction_upper

print(f"\n📅 FORECAST DATE RANGE:")
print(f"   Start: {future_df['date'].min().strftime('%B %d, %Y')}")
print(f"   End:   {future_df['date'].max().strftime('%B %d, %Y')}")
print(f"   Total Days: {len(future_df)}")

print(f"\n📊 FORECAST SUMMARY:")
print(f"   Total: ${future_df['predicted_sales'].sum():,.2f}")
print(f"   Avg Daily: ${future_df['predicted_sales'].mean():.2f}")
print(f"   95% CI: ${future_df['pred_lower'].mean():.2f} - ${future_df['pred_upper'].mean():.2f}")

# ============================================================================
# VISUALIZATION: 6-MONTH FORECAST (INDIVIDUAL GRAPHS)
# ============================================================================

print("\n" + "="*100)
print("GENERATING 6-MONTH FORECAST VISUALIZATIONS (INDIVIDUAL GRAPHS)")
print("="*100)

# Prepare data
historical_last_6m = daily_full_all[daily_full_all['date'] >= (daily_full_all['date'].max() - timedelta(days=180))].copy()

# ============================================================================
# GRAPH 1: Full Timeline - Historical (6M) + Forecast (6M) ONLY
# ============================================================================
print("\n📊 Graph 1: 6-Month Historical + 6-Month Forecast")
fig1, ax1 = plt.subplots(figsize=(16, 6))

# Prepare EXACTLY 6 months of historical data (180 days before forecast start)
forecast_start_date = future_df['date'].min()
historical_6m_start = forecast_start_date - timedelta(days=180)
historical_last_6m = daily_full_all[
    (daily_full_all['date'] >= historical_6m_start) & 
    (daily_full_all['date'] < forecast_start_date)
].copy()

print(f"  Historical period: {historical_last_6m['date'].min().date()} to {historical_last_6m['date'].max().date()} ({len(historical_last_6m)} days)")
print(f"  Forecast period:   {future_df['date'].min().date()} to {future_df['date'].max().date()} ({len(future_df)} days)")

# Plot historical (exactly 6 months before forecast)
ax1.plot(historical_last_6m['date'], historical_last_6m['total_sales'], 
         label='Historical (Last 6 Months)', linewidth=2, color='steelblue', alpha=0.8)

# Connect historical to forecast - add last historical point to forecast
last_historical_date = historical_last_6m['date'].iloc[-1]
last_historical_value = historical_last_6m['total_sales'].iloc[-1]
first_forecast_date = future_df['date'].iloc[0]
first_forecast_value = future_df['predicted_sales'].iloc[0]

# Plot connecting line
ax1.plot([last_historical_date, first_forecast_date], 
         [last_historical_value, first_forecast_value],
         linewidth=2.5, color='orange', alpha=0.9)

# Plot forecast (6 months)
ax1.plot(future_df['date'], future_df['predicted_sales'], 
         label='Forecast (Next 6 Months)', linewidth=2.5, color='orange', alpha=0.9)

# Add confidence interval
ax1.fill_between(future_df['date'], 
                  future_df['pred_lower'], 
                  future_df['pred_upper'],
                  alpha=0.2, color='orange', label='95% Confidence Interval')

# Add vertical line at forecast start
ax1.axvline(forecast_start_date, color='red', linestyle='--', linewidth=2, 
            alpha=0.7, label='Forecast Start')

# Add horizontal lines for averages
hist_avg = historical_last_6m['total_sales'].mean()
forecast_avg = future_df['predicted_sales'].mean()
ax1.axhline(hist_avg, color='steelblue', linestyle=':', linewidth=2, 
            alpha=0.5, label=f'Historical Avg: ${hist_avg:.0f}')
ax1.axhline(forecast_avg, color='orange', linestyle=':', linewidth=2, 
            alpha=0.5, label=f'Forecast Avg: ${forecast_avg:.0f}')

ax1.set_title('6-Month Sales Forecast: Historical vs Predicted', 
              fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
ax1.set_ylabel('Daily Sales ($)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=10, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Set x-axis limits to show exactly 12 months (6M historical + 6M forecast)
ax1.set_xlim(historical_6m_start, future_df['date'].max())

plt.tight_layout()
plt.show()

# ============================================================================
# SEASONAL ANALYSIS OF FORECAST
# ============================================================================
print("\n" + "="*100)
print("SEASONAL PERFORMANCE ANALYSIS (FORECAST PERIOD)")
print("="*100)

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:  # 9, 10, 11
        return 'Fall'

future_df['season'] = future_df['date'].dt.month.apply(get_season)

# Calculate seasonal statistics
seasonal_stats = future_df.groupby('season')['predicted_sales'].agg([
    ('Total', 'sum'),
    ('Avg_Daily', 'mean'),
    ('Median', 'median'),
    ('Days', 'count'),
    ('Min', 'min'),
    ('Max', 'max'),
    ('Std_Dev', 'std')
]).round(2)

# Sort by total sales descending
seasonal_stats = seasonal_stats.sort_values('Total', ascending=False)

print("\n📊 FORECAST SEASONAL BREAKDOWN:")
print("-" * 100)
print(seasonal_stats.to_string())

# Identify best season
best_season = seasonal_stats.index[0]
best_total = seasonal_stats.loc[best_season, 'Total']
best_avg = seasonal_stats.loc[best_season, 'Avg_Daily']

print(f"\n🏆 BEST PERFORMING SEASON: {best_season.upper()}")
print(f"   Total Sales: ${best_total:,.0f}")
print(f"   Avg Daily: ${best_avg:,.0f}")
print(f"   Days in Forecast: {int(seasonal_stats.loc[best_season, 'Days'])}")

# Calculate percentage of total forecast
total_forecast_sales = future_df['predicted_sales'].sum()
pct_of_total = (best_total / total_forecast_sales) * 100
print(f"   % of Total Forecast: {pct_of_total:.1f}%")

# Compare all seasons
print(f"\n📈 SEASONAL COMPARISON:")
for season in ['Winter', 'Spring', 'Summer', 'Fall']:
    if season in seasonal_stats.index:
        total = seasonal_stats.loc[season, 'Total']
        avg = seasonal_stats.loc[season, 'Avg_Daily']
        days = int(seasonal_stats.loc[season, 'Days'])
        pct = (total / total_forecast_sales) * 100
        
        # Visual indicator
        if season == best_season:
            indicator = "🥇"
        elif seasonal_stats['Total'].rank(ascending=False)[season] == 2:
            indicator = "🥈"
        elif seasonal_stats['Total'].rank(ascending=False)[season] == 3:
            indicator = "🥉"
        else:
            indicator = "  "
        
        print(f"   {indicator} {season:8s}: ${total:>10,.0f} total | ${avg:>6,.0f}/day | {days:2d} days | {pct:5.1f}%")

# Visualize seasonal performance
print("\n" + "="*100)
print("GENERATING SEASONAL PERFORMANCE VISUALIZATION")
print("="*100)

fig_seasonal, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart - Total sales by season
seasons_ordered = ['Winter', 'Spring', 'Summer', 'Fall']
colors_season = ['#87CEEB', '#90EE90', '#FFD700', '#FF8C00']
season_totals = [seasonal_stats.loc[s, 'Total'] if s in seasonal_stats.index else 0 
                 for s in seasons_ordered]

bars = ax_bar.bar(seasons_ordered, season_totals, color=colors_season, 
                   alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, total in zip(bars, season_totals):
    if total > 0:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                   f'${height/1000:.1f}K',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

# Highlight best season
max_idx = season_totals.index(max(season_totals))
bars[max_idx].set_edgecolor('red')
bars[max_idx].set_linewidth(3)

ax_bar.set_title('Forecast: Total Sales by Season', fontsize=14, fontweight='bold')
ax_bar.set_xlabel('Season', fontsize=12, fontweight='bold')
ax_bar.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
ax_bar.grid(True, alpha=0.3, axis='y')
ax_bar.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# Pie chart - Percentage distribution
season_pcts = [(seasonal_stats.loc[s, 'Total'] / total_forecast_sales * 100) 
               if s in seasonal_stats.index else 0 
               for s in seasons_ordered]

wedges, texts, autotexts = ax_pie.pie(season_pcts, labels=seasons_ordered, 
                                        colors=colors_season, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'},
                                        explode=[0.05 if i == max_idx else 0 for i in range(4)])

# Highlight best season slice
wedges[max_idx].set_edgecolor('red')
wedges[max_idx].set_linewidth(3)

ax_pie.set_title('Forecast: Sales Distribution by Season', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✅ Seasonal analysis complete")

# ============================================================================
# DIAGNOSTIC: Why is Winter so low?
# ============================================================================
print("\n" + "="*100)
print("DIAGNOSTIC: INVESTIGATING LOW WINTER FORECAST")
print("="*100)

# Check Winter months in detail
winter_forecast = future_df[future_df['season'] == 'Winter'].copy()
print(f"\n📅 Winter Period in Forecast:")
print(f"   Dates: {winter_forecast['date'].min().date()} to {winter_forecast['date'].max().date()}")
print(f"   Months: {winter_forecast['date'].dt.strftime('%B %Y').unique()}")

# Compare with historical Winter performance
historical_winter = daily_full_all[daily_full_all['date'].dt.month.isin([12, 1, 2])].copy()
hist_winter_avg = historical_winter['total_sales'].mean()
hist_winter_median = historical_winter['total_sales'].median()

print(f"\n📊 Historical Winter Performance:")
print(f"   Historical Winter Avg: ${hist_winter_avg:,.2f}/day")
print(f"   Historical Winter Median: ${hist_winter_median:,.2f}/day")
print(f"   Forecast Winter Avg: ${winter_forecast['predicted_sales'].mean():,.2f}/day")
print(f"   Difference: {((winter_forecast['predicted_sales'].mean() / hist_winter_avg - 1) * 100):.1f}%")

# Check if there's a trend issue (declining predictions over time)
print(f"\n🔍 Forecast Trend Analysis:")
future_df['month_year'] = future_df['date'].dt.to_period('M').astype(str)
monthly_avg = future_df.groupby('month_year')['predicted_sales'].mean()
print("\nAverage Daily Sales by Month in Forecast:")
print(monthly_avg.to_string())

# Check lag features for Winter period
print(f"\n🔍 Checking Lag Features for First Winter Days:")
winter_sample = winter_forecast.head(5)[['date', 'predicted_sales', 'lag_1', 'lag_7', 'lag_30', 
                                          'rolling_mean_7', 'rolling_mean_30']].copy()
print(winter_sample.to_string(index=False))

# Check if predictions are degrading over time
print(f"\n⚠️  POTENTIAL ISSUE DETECTED:")
first_month_avg = future_df.head(30)['predicted_sales'].mean()
last_month_avg = future_df.tail(30)['predicted_sales'].mean()
degradation_pct = ((last_month_avg / first_month_avg - 1) * 100)

if degradation_pct < -20:
    print(f"   ❌ SEVERE DEGRADATION: Predictions drop {abs(degradation_pct):.1f}% from start to end")
    print(f"   First 30 days avg: ${first_month_avg:,.2f}/day")
    print(f"   Last 30 days avg: ${last_month_avg:,.2f}/day")
    print(f"\n   ROOT CAUSE: Multi-step forecast error accumulation")
    print(f"   - Predictions use previous predictions as lag features")
    print(f"   - Errors compound over 180 days")
    print(f"   - Model wasn't trained to handle 6-month horizon")
    print(f"\n   💡 SOLUTIONS:")
    print(f"   1. Use actual historical data as anchors (hybrid approach)")
    print(f"   2. Limit forecast to 30-60 days instead of 180")
    print(f"   3. Retrain model on longer sequences")
    print(f"   4. Use seasonal averages for distant predictions")
    print(f"   5. Apply minimum threshold based on historical winter performance")
elif degradation_pct < -10:
    print(f"   ⚠️  MODERATE DEGRADATION: Predictions drop {abs(degradation_pct):.1f}% over time")
else:
    print(f"   ✅ Predictions stable: {degradation_pct:+.1f}% change")

# Visualize the degradation
fig_diag, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Full forecast with trend line
ax1.plot(future_df['date'], future_df['predicted_sales'], 
         linewidth=2, color='orange', alpha=0.7, label='Daily Forecast')

# Add 7-day moving average to show trend
ma7 = future_df['predicted_sales'].rolling(7).mean()
ax1.plot(future_df['date'], ma7, 
         linewidth=3, color='red', alpha=0.8, label='7-Day Moving Average')

# Highlight Winter period
winter_mask = future_df['season'] == 'Winter'
ax1.fill_between(future_df['date'], 0, future_df['predicted_sales'].max() * 1.1,
                 where=winter_mask, alpha=0.2, color='blue', label='Winter Period')

ax1.axhline(hist_winter_avg, color='green', linestyle='--', linewidth=2,
            label=f'Historical Winter Avg: ${hist_winter_avg:.0f}')

ax1.set_title('DIAGNOSTIC: Forecast Degradation Over Time', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=11)
ax1.set_ylabel('Predicted Daily Sales ($)', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Compare by season with historical baseline
ax2_data = []
for season in ['Fall', 'Winter', 'Spring', 'Summer']:
    forecast_avg = future_df[future_df['season'] == season]['predicted_sales'].mean()
    
    # Get historical average for same season
    season_months = {'Winter': [12,1,2], 'Spring': [3,4,5], 
                     'Summer': [6,7,8], 'Fall': [9,10,11]}
    hist_season = daily_full_all[daily_full_all['date'].dt.month.isin(season_months[season])]
    hist_avg = hist_season['total_sales'].mean()
    
    ax2_data.append({
        'Season': season,
        'Historical': hist_avg,
        'Forecast': forecast_avg
    })

df_compare = pd.DataFrame(ax2_data)
x_pos = np.arange(len(df_compare))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, df_compare['Historical'], width, 
                label='Historical Avg', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x_pos + width/2, df_compare['Forecast'], width,
                label='Forecast Avg', color='orange', alpha=0.7, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.0f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Season', fontsize=11, fontweight='bold')
ax2.set_ylabel('Average Daily Sales ($)', fontsize=11, fontweight='bold')
ax2.set_title('Historical vs Forecast by Season', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_compare['Season'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n✅ Diagnostic analysis complete")
# ============================================================================
# GRAPH 2: Monthly Forecast Breakdown
# ============================================================================
print("\n📊 Graph 2: Monthly Forecast Breakdown")
fig2, ax2 = plt.subplots(figsize=(14, 6))

# Aggregate by month
future_df['year_month'] = future_df['date'].dt.to_period('M')
future_df['month_name'] = future_df['date'].dt.strftime('%b %Y')

monthly_forecast = future_df.groupby('month_name').agg({
    'predicted_sales': ['sum', 'mean', 'count'],
    'pred_lower': 'sum',
    'pred_upper': 'sum'
}).reset_index()

monthly_forecast.columns = ['Month', 'Total', 'Avg_Daily', 'Days', 'Lower', 'Upper']

# Bar chart
x_pos = np.arange(len(monthly_forecast))
bars = ax2.bar(x_pos, monthly_forecast['Total'], color='orange', alpha=0.7, 
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, row) in enumerate(zip(bars, monthly_forecast.itertuples())):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'${height/1000:.1f}K\n({row.Days}d)',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_xlabel('Month', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Sales ($)', fontsize=11, fontweight='bold')
ax2.set_title('Monthly Forecast Breakdown', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(monthly_forecast['Month'], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 3: Day of Week Pattern (Historical vs Forecast)
# ============================================================================
print("\n📊 Graph 3: Day of Week Pattern")
fig3, ax3 = plt.subplots(figsize=(14, 6))

dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Calculate averages
hist_dow = historical_last_6m.groupby(historical_last_6m['date'].dt.dayofweek)['total_sales'].mean()
future_df['day_of_week_num'] = future_df['date'].dt.dayofweek
forecast_dow = future_df.groupby('day_of_week_num')['predicted_sales'].mean()

x_pos = np.arange(7)
width = 0.35

bars1 = ax3.bar(x_pos - width/2, [hist_dow.get(i, 0) for i in range(7)], 
                width, label='Historical', color='steelblue', alpha=0.7, edgecolor='black')
bars2 = ax3.bar(x_pos + width/2, [forecast_dow.get(i, 0) for i in range(7)], 
                width, label='Forecast', color='orange', alpha=0.7, edgecolor='black')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                     f'${height:.0f}',
                     ha='center', va='bottom', fontsize=8)

ax3.set_xlabel('Day of Week', fontsize=11, fontweight='bold')
ax3.set_ylabel('Average Daily Sales ($)', fontsize=11, fontweight='bold')
ax3.set_title('Day of Week Pattern: Historical vs Forecast', fontsize=13, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(dow_names)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 4: Forecast Distribution & Statistics
# ============================================================================
print("\n📊 Graph 4: Forecast Distribution")
fig4, ax4 = plt.subplots(figsize=(14, 6))

# Histogram of forecast values
ax4.hist(future_df['predicted_sales'], bins=30, color='orange', alpha=0.7, 
         edgecolor='black', linewidth=1)
ax4.axvline(future_df['predicted_sales'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Mean: ${future_df['predicted_sales'].mean():.0f}")
ax4.axvline(future_df['predicted_sales'].median(), color='green', linestyle='--', 
            linewidth=2, label=f"Median: ${future_df['predicted_sales'].median():.0f}")

ax4.set_xlabel('Predicted Daily Sales ($)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Forecast Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# GRAPH 5: Key Metrics Summary
# ============================================================================
print("\n📊 Graph 5: Key Metrics Summary")
fig5, ax5 = plt.subplots(figsize=(14, 10))
ax5.axis('off')

# Calculate key metrics
total_forecast = future_df['predicted_sales'].sum()
avg_daily = future_df['predicted_sales'].mean()
total_hist_6m = historical_last_6m['total_sales'].sum()
avg_hist_6m = historical_last_6m['total_sales'].mean()
pct_change = ((avg_daily / avg_hist_6m) - 1) * 100

# Apply bias correction
bias_corrected_total = total_forecast * 0.908
bias_corrected_avg = avg_daily * 0.908

summary_text = f"""
╔════════════════════════════════════════════════════════════╗
║          6-MONTH FORECAST SUMMARY                          ║
╚════════════════════════════════════════════════════════════╝

FORECAST TOTALS (Raw)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total Forecast:           ${total_forecast:>15,.0f}
  Average Daily:            ${avg_daily:>15,.0f}
  Median Daily:             ${future_df['predicted_sales'].median():>15,.0f}
  
BIAS-CORRECTED FORECAST (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Corrected Total:          ${bias_corrected_total:>15,.0f}
  Corrected Avg Daily:      ${bias_corrected_avg:>15,.0f}
  
CONFIDENCE INTERVALS (95%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Lower Bound (Daily):      ${future_df['pred_lower'].mean():>15,.0f}
  Upper Bound (Daily):      ${future_df['pred_upper'].mean():>15,.0f}
  Range:                    ±${pred_std * 1.96:>14,.0f}

HISTORICAL COMPARISON (Last 6 Months)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Historical Total:         ${total_hist_6m:>15,.0f}
  Historical Avg Daily:     ${avg_hist_6m:>15,.0f}
  Change:                   {pct_change:>14.1f}%
  
MODEL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  R² Score:                 {best_r2:>20.3f}
  MAPE:                     {ensemble_metrics['MAPE']:>19.1f}%
  Within 20%:               {ensemble_metrics['Within_20%']:>19.1f}%
  Direction Accuracy:       {ensemble_metrics['Direction_Accuracy_%']:>19.1f}%

BUSINESS PLANNING SCENARIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Conservative (-20%):      ${bias_corrected_avg * 0.8:>15,.0f}/day
  Expected (Corrected):     ${bias_corrected_avg:>15,.0f}/day
  Optimistic (+20%):        ${bias_corrected_avg * 1.2:>15,.0f}/day
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

plt.tight_layout()
plt.show()

print("\n✅ All 5 forecast visualizations displayed individually")

# ======================================================================
# POWER BI EXPORTS
#  - Graph 1: 6M DAILY historical + 6M MONTHLY cumulative forecast
#  - Still keep daily and monthly forecast CSVs for other visuals
# ======================================================================

# 1) GRAPH 1 DATA  ------------------------------------------------------
# Historical part: use the same 6-month window used in Graph 1
hist_part = historical_last_6m[['date', 'total_sales']].copy()
hist_part.rename(columns={'total_sales': 'historical_daily_sales'}, inplace=True)
hist_part['cumulative_forecast'] = np.nan
hist_part['cum_lower'] = np.nan
hist_part['cum_upper'] = np.nan
hist_part['segment'] = 'historical'

# Forecast part: use MONTHLY totals and build cumulative forecast
monthly_export = monthly_forecast.copy()          # has Month, Total, Lower, Upper
monthly_export['date'] = pd.to_datetime('1 ' + monthly_export['Month'])
monthly_export = monthly_export.sort_values('date').reset_index(drop=True)

monthly_export['cumulative_forecast'] = monthly_export['Total'].cumsum()
monthly_export['cum_lower'] = monthly_export['Lower'].cumsum()
monthly_export['cum_upper'] = monthly_export['Upper'].cumsum()
monthly_export['historical_daily_sales'] = np.nan
monthly_export['segment'] = 'forecast'

fc_part = monthly_export[['date',
                          'historical_daily_sales',
                          'cumulative_forecast',
                          'cum_lower',
                          'cum_upper',
                          'segment']]

# Combine historical daily + monthly cumulative forecast
powerbi_graph1_data = pd.concat([
    hist_part[['date', 'historical_daily_sales',
               'cumulative_forecast', 'cum_lower', 'cum_upper', 'segment']],
    fc_part
], ignore_index=True)

powerbi_graph1_data.to_csv("powerbi_graph1_data.csv", index=False)
print("✅ Graph 1 data exported to 'powerbi_graph1_data.csv'")

# 2) DAILY FORECAST DETAILS (still daily for 180 days) ------------------
future_df[['date', 'predicted_sales', 'pred_lower', 'pred_upper', 'month_name']].to_csv(
    "forecast_6months_detailed.csv", index=False
)
print("✅ Detailed daily forecast exported to 'forecast_6months_detailed.csv'")

# 3) MONTHLY FORECAST TOTALS (non-cumulative) --------------------------
monthly_forecast.to_csv("forecast_6months_monthly.csv", index=False)
print("✅ Monthly forecast exported to 'forecast_6months_monthly.csv'")

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
print("="*100)