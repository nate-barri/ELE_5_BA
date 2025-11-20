import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("ETL/dataset_ele_5.csv")

print("="*80)
print("CLOTHING STORE PREDICTIVE ANALYTICS - COMPREHENSIVE ANALYSIS")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Date Range: {df['purchase_date'].min()} to {df['purchase_date'].max()}")

# Data Quality Check
print("\n" + "="*80)
print("DATA QUALITY ASSESSMENT")
print("="*80)
print("\nMissing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
for col, count in missing.items():
    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")

print(f"\nReturn Rate: {df['is_returned'].mean()*100:.2f}%")
print(f"  - Returns: {df['is_returned'].sum()}")
print(f"  - No Returns: {(~df['is_returned']).sum()}")

# Data preprocessing
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['month'] = df['purchase_date'].dt.month
df['day_of_week'] = df['purchase_date'].dt.dayofweek
df['quarter'] = df['purchase_date'].dt.quarter
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Handle missing values strategically
# For size (categorical), create "Unknown" category
df['size'] = df['size'].fillna('Unknown')

# For customer_rating, use median (or could use category-specific median)
df['customer_rating'] = df['customer_rating'].fillna(df['customer_rating'].median())

# Encode categorical variables
le_category = LabelEncoder()
le_brand = LabelEncoder()
le_season = LabelEncoder()
le_size = LabelEncoder()
le_color = LabelEncoder()
le_country = LabelEncoder()

df['category_encoded'] = le_category.fit_transform(df['category'])
df['brand_encoded'] = le_brand.fit_transform(df['brand'])
df['season_encoded'] = le_season.fit_transform(df['season'])
df['size_encoded'] = le_size.fit_transform(df['size'])
df['color_encoded'] = le_color.fit_transform(df['color'])
df['country_encoded'] = le_country.fit_transform(df['country'])

# Advanced feature engineering
df['discount_amount'] = df['original_price'] - df['current_price']
df['discount_percentage'] = (df['discount_amount'] / df['original_price']) * 100
df['price_per_rating'] = df['current_price'] / (df['customer_rating'] + 0.1)
df['is_high_discount'] = (df['markdown_percentage'] > 30).astype(int)
df['is_premium'] = (df['original_price'] > df['original_price'].quantile(0.75)).astype(int)
df['is_low_stock'] = (df['stock_quantity'] < df['stock_quantity'].quantile(0.25)).astype(int)
df['sales_per_stock'] = df['total_sales'] / (df['stock_quantity'] + 1)
df['is_high_rating'] = (df['customer_rating'] >= 4.0).astype(int)

# ============================================================================
# PREDICTION 1: RETURN PREDICTION (CLASSIFICATION) - WITH CLASS BALANCING
# ============================================================================
print("\n" + "="*80)
print("PREDICTION 1: PRODUCT RETURN PREDICTION (CLASSIFICATION)")
print("="*80)
print("Business Impact: Returns cost 15-30% of revenue in fashion retail")
print("Goal: Identify high-risk products/transactions to reduce return rates\n")

# Prepare features for return prediction
feature_cols = ['category_encoded', 'brand_encoded', 'season_encoded', 'size_encoded',
                'color_encoded', 'original_price', 'markdown_percentage', 'current_price',
                'stock_quantity', 'customer_rating', 'month', 'day_of_week', 
                'quarter', 'discount_amount', 'country_encoded', 'is_high_discount',
                'is_premium', 'is_weekend']

X_return = df[feature_cols].copy()
y_return = df['is_returned'].copy()

# Handle any remaining missing values
X_return = X_return.fillna(X_return.median())

# Split data first
X_train_ret, X_test_ret, y_train_ret, y_test_ret = train_test_split(
    X_return, y_return, test_size=0.2, random_state=42, stratify=y_return
)

# Apply SMOTE to handle class imbalance (only on training data)
print("Applying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train_ret_balanced, y_train_ret_balanced = smote.fit_resample(X_train_ret, y_train_ret)

print(f"Original training set: {y_train_ret.value_counts().to_dict()}")
print(f"Balanced training set: {pd.Series(y_train_ret_balanced).value_counts().to_dict()}")

# Scale features
scaler_ret = StandardScaler()
X_train_ret_scaled = scaler_ret.fit_transform(X_train_ret_balanced)
X_test_ret_scaled = scaler_ret.transform(X_test_ret)

# Train Random Forest Classifier with adjusted parameters
rf_classifier = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced'
)
rf_classifier.fit(X_train_ret_scaled, y_train_ret_balanced)

# Predictions
y_pred_ret = rf_classifier.predict(X_test_ret_scaled)
y_pred_ret_proba = rf_classifier.predict_proba(X_test_ret_scaled)[:, 1]

# Evaluation Metrics
accuracy = accuracy_score(y_test_ret, y_pred_ret)
precision = precision_score(y_test_ret, y_pred_ret, zero_division=0)
recall = recall_score(y_test_ret, y_pred_ret, zero_division=0)
f1 = f1_score(y_test_ret, y_pred_ret, zero_division=0)
roc_auc = roc_auc_score(y_test_ret, y_pred_ret_proba)

print("\nCLASSIFICATION METRICS:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test_ret, y_pred_ret)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Business metrics
if cm[1,1] > 0:
    cost_per_return = 50  # Estimated cost per return
    returns_prevented = cm[1,1]
    potential_savings = returns_prevented * cost_per_return
    print(f"\nBUSINESS IMPACT:")
    print(f"  Returns correctly identified: {cm[1,1]}")
    print(f"  Potential cost savings: ${potential_savings:.2f}")
    print(f"  False alarms: {cm[0,1]} (acceptable for prevention strategy)")

# Feature importance
feature_importance_ret = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features Predicting Returns:")
print(feature_importance_ret.head(10).to_string(index=False))

# ============================================================================
# PREDICTION 2: SALES FORECASTING (REGRESSION) - IMPROVED
# ============================================================================
print("\n" + "="*80)
print("PREDICTION 2: SALES FORECASTING (REGRESSION)")
print("="*80)
print("Business Impact: Optimize inventory and prevent stockouts/overstock")
print("Goal: Predict total_sales for better purchasing decisions\n")

# Enhanced features for sales prediction
sales_features = ['category_encoded', 'brand_encoded', 'season_encoded', 'size_encoded',
                  'color_encoded', 'original_price', 'markdown_percentage', 'current_price',
                  'stock_quantity', 'customer_rating', 'month', 'day_of_week', 
                  'quarter', 'discount_amount', 'country_encoded', 'is_high_discount',
                  'is_premium', 'is_weekend', 'is_low_stock', 'is_high_rating']

X_sales = df[sales_features].copy()
y_sales = df['total_sales'].copy()

X_sales = X_sales.fillna(X_sales.median())

# Split data
X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
    X_sales, y_sales, test_size=0.2, random_state=42
)

# Scale features
scaler_sales = StandardScaler()
X_train_sales_scaled = scaler_sales.fit_transform(X_train_sales)
X_test_sales_scaled = scaler_sales.transform(X_test_sales)

# Train Gradient Boosting Regressor (often better for structured data)
gb_sales = GradientBoostingRegressor(
    n_estimators=200, 
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
gb_sales.fit(X_train_sales_scaled, y_train_sales)

# Predictions
y_pred_sales = gb_sales.predict(X_test_sales_scaled)

# Calculate comprehensive evaluation metrics
r2 = r2_score(y_test_sales, y_pred_sales)
mae = mean_absolute_error(y_test_sales, y_pred_sales)
mse = mean_squared_error(y_test_sales, y_pred_sales)
rmse = np.sqrt(mse)

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test_sales - y_pred_sales) / (y_test_sales + 1e-10))) * 100

# MASE (Mean Absolute Scaled Error)
naive_forecast_error = np.mean(np.abs(np.diff(y_train_sales)))
mase = mae / (naive_forecast_error + 1e-10)

print("REGRESSION METRICS:")
print(f"R² Score: {r2:.4f} {'✓' if r2 > 0.6 else '(Model explains ' + str(int(r2*100)) + '% of variance)'}")
print(f"MAE (Mean Absolute Error): ${mae:.2f}")
print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
print(f"MASE (Mean Absolute Scaled Error): {mase:.4f} {'✓ Better than naive' if mase < 1 else '(Needs improvement)'}")

print(f"\nMean Actual Sales: ${y_test_sales.mean():.2f}")
print(f"Mean Predicted Sales: ${y_pred_sales.mean():.2f}")
print(f"Prediction Error Range: ±${mae:.2f} (MAE)")
print(f"Prediction Accuracy: {(1 - mape/100)*100:.1f}%")

# Feature importance
feature_importance_sales = pd.DataFrame({
    'feature': sales_features,
    'importance': gb_sales.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features Predicting Sales:")
print(feature_importance_sales.head(10).to_string(index=False))

# ============================================================================
# PREDICTION 3: CUSTOMER RATING PREDICTION (REGRESSION)
# ============================================================================
print("\n" + "="*80)
print("PREDICTION 3: CUSTOMER RATING PREDICTION (REGRESSION)")
print("="*80)
print("Business Impact: Quality control and product selection")
print("Goal: Predict customer_rating to identify potential quality issues\n")

# Prepare features (excluding rating itself)
feature_cols_rating = ['category_encoded', 'brand_encoded', 'season_encoded', 
                       'size_encoded', 'color_encoded', 'original_price', 
                       'markdown_percentage', 'current_price', 'stock_quantity',
                       'month', 'country_encoded', 'discount_amount', 'is_premium']

X_rating = df[feature_cols_rating].copy()
y_rating = df['customer_rating'].copy()

# Remove any rows where rating is still NaN
valid_mask = ~y_rating.isna()
X_rating = X_rating[valid_mask]
y_rating = y_rating[valid_mask]

X_rating = X_rating.fillna(X_rating.median())

print(f"Valid samples for rating prediction: {len(X_rating)}")

# Split data
X_train_rat, X_test_rat, y_train_rat, y_test_rat = train_test_split(
    X_rating, y_rating, test_size=0.2, random_state=42
)

# Scale features
scaler_rat = StandardScaler()
X_train_rat_scaled = scaler_rat.fit_transform(X_train_rat)
X_test_rat_scaled = scaler_rat.transform(X_test_rat)

# Train Random Forest Regressor
rf_rating = RandomForestRegressor(
    n_estimators=200, 
    random_state=42, 
    max_depth=10,
    min_samples_split=10
)
rf_rating.fit(X_train_rat_scaled, y_train_rat)

# Predictions
y_pred_rat = rf_rating.predict(X_test_rat_scaled)

# Evaluation metrics
r2_rat = r2_score(y_test_rat, y_pred_rat)
mae_rat = mean_absolute_error(y_test_rat, y_pred_rat)
rmse_rat = np.sqrt(mean_squared_error(y_test_rat, y_pred_rat))
mape_rat = np.mean(np.abs((y_test_rat - y_pred_rat) / (y_test_rat + 1e-10))) * 100

# MASE for rating
naive_forecast_error_rat = np.mean(np.abs(np.diff(y_train_rat)))
mase_rat = mae_rat / (naive_forecast_error_rat + 1e-10)

print("REGRESSION METRICS:")
print(f"R² Score: {r2_rat:.4f}")
print(f"MAE (Mean Absolute Error): {mae_rat:.4f} stars")
print(f"RMSE (Root Mean Squared Error): {rmse_rat:.4f} stars")
print(f"MAPE (Mean Absolute Percentage Error): {mape_rat:.2f}%")
print(f"MASE (Mean Absolute Scaled Error): {mase_rat:.4f}")

print(f"\nMean Actual Rating: {y_test_rat.mean():.2f} stars")
print(f"Mean Predicted Rating: {y_pred_rat.mean():.2f} stars")
print(f"Rating Range: {y_test_rat.min():.1f} - {y_test_rat.max():.1f}")

# Identify low-rated products
low_rated_threshold = 3.5
predicted_low_rated = np.sum(y_pred_rat < low_rated_threshold)
actual_low_rated = np.sum(y_test_rat < low_rated_threshold)
print(f"\nQuality Alert Analysis:")
print(f"  Products with actual rating < {low_rated_threshold}: {actual_low_rated}")
print(f"  Products predicted rating < {low_rated_threshold}: {predicted_low_rated}")

# ============================================================================
# PREDICTION 4: OPTIMAL MARKDOWN STRATEGY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PREDICTION 4: MARKDOWN STRATEGY OPTIMIZATION")
print("="*80)
print("Business Impact: Maximize revenue through optimal pricing")
print("Goal: Understand markdown impact on sales velocity\n")

# Analyze markdown effectiveness by category
markdown_analysis = df.groupby(['category', 'is_high_discount']).agg({
    'total_sales': ['mean', 'sum', 'count'],
    'stock_quantity': 'mean',
    'customer_rating': 'mean'
}).round(2)

print("Markdown Impact by Category:")
print(markdown_analysis)

# Calculate revenue per markdown level
markdown_bins = [0, 10, 20, 30, 40, 100]
markdown_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40%+']
df['markdown_bracket'] = pd.cut(df['markdown_percentage'], bins=markdown_bins, labels=markdown_labels)

print("\n\nSales Performance by Markdown Level:")
markdown_perf = df.groupby('markdown_bracket').agg({
    'total_sales': ['mean', 'sum', 'count'],
    'customer_rating': 'mean',
    'is_returned': 'mean'
}).round(2)
print(markdown_perf)

# ============================================================================
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("BUSINESS INSIGHTS & ACTIONABLE RECOMMENDATIONS")
print("="*80)

# Return rate analysis
return_rate = df['is_returned'].mean() * 100
print(f"\n1. RETURN MANAGEMENT:")
print(f"   Current Return Rate: {return_rate:.2f}%")
print(f"   Model Performance:")
print(f"     - Accuracy: {accuracy*100:.1f}%")
print(f"     - Can identify {recall*100:.1f}% of actual returns (Recall)")
print(f"     - Precision: {precision*100:.1f}% of flagged items are actual returns")
print(f"   → Implement pre-shipment quality checks for high-risk orders")
print(f"   → Add detailed size guides for categories with high return rates")

# Sales insights
print(f"\n2. SALES FORECASTING:")
print(f"   Model Performance: R²={r2:.3f}, MAPE={mape:.1f}%")
if r2 > 0.5:
    print(f"   ✓ Strong predictive power - reliable for planning")
elif r2 > 0.3:
    print(f"   ~ Moderate predictive power - useful with caution")
else:
    print(f"   ⚠ Model needs more features or data volume")
print(f"   Average prediction error: ±${mae:.2f}")
print(f"   → Use for: inventory reordering, demand forecasting")
print(f"   → Optimize: stock levels by category and season")

# Rating insights
avg_rating = df['customer_rating'].mean()
low_rated_products = len(df[df['customer_rating'] < 3.5])
print(f"\n3. QUALITY CONTROL:")
print(f"   Average Rating: {avg_rating:.2f} stars")
print(f"   Products rated < 3.5 stars: {low_rated_products} ({low_rated_products/len(df)*100:.1f}%)")
print(f"   Rating Prediction MAE: ±{mae_rat:.2f} stars")
print(f"   → Flag products predicted < 3.5 stars for quality review")
print(f"   → Investigate: brand and category patterns in low ratings")

# Markdown insights
avg_markdown = df['markdown_percentage'].mean()
high_discount_mask = df['is_high_discount'] == 1
low_discount_mask = df['is_high_discount'] == 0
high_markdown_sales = df.loc[high_discount_mask, 'total_sales'].mean()
low_markdown_sales = df.loc[low_discount_mask, 'total_sales'].mean()
print(f"\n4. PRICING STRATEGY:")
print(f"   Average Markdown: {avg_markdown:.1f}%")
print(f"   High discount (>30%) avg sales: ${high_markdown_sales:.2f}")
print(f"   Low discount (≤30%) avg sales: ${low_markdown_sales:.2f}")
print(f"   → Optimal markdown appears to be category-dependent")
print(f"   → Test dynamic pricing based on stock age and velocity")

# Category performance
print(f"\n5. CATEGORY PERFORMANCE:")
category_metrics = df.groupby('category').agg({
    'total_sales': ['mean', 'sum'],
    'customer_rating': 'mean',
    'is_returned': lambda x: x.mean() * 100
}).round(2)
category_metrics.columns = ['Avg_Sales', 'Total_Sales', 'Avg_Rating', 'Return_Rate%']
category_metrics = category_metrics.sort_values('Total_Sales', ascending=False)
print(category_metrics)

# Seasonal insights
print(f"\n6. SEASONAL TRENDS:")
season_metrics = df.groupby('season').agg({
    'total_sales': ['mean', 'sum'],
    'customer_rating': 'mean'
}).round(2)
season_metrics.columns = ['Avg_Sales', 'Total_Sales', 'Avg_Rating']
print(season_metrics)

print("\n" + "="*80)
print("TOP 5 RECOMMENDED ACTIONS:")
print("="*80)
print("1. Deploy return prediction model to flag high-risk orders")
print("   → Implement additional quality checks before shipping")
print("")
print("2. Use sales forecasts for automated inventory management")
print("   → Set reorder points based on predicted demand")
print("")
print("3. Quality alert system for products with predicted rating < 3.5")
print("   → Review product descriptions and supplier quality")
print("")
print("4. Optimize markdown strategy by category and season")
print("   → Dynamic pricing: aggressive markdowns for slow-movers")
print("")
print("5. Focus on high-performing categories and seasonal timing")
print("   → Allocate marketing budget to winners, reduce losers")
print("="*80)

# Create comprehensive visualizations
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Actual vs Predicted Sales
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test_sales, y_pred_sales, alpha=0.5, s=30)
ax1.plot([y_test_sales.min(), y_test_sales.max()], 
         [y_test_sales.min(), y_test_sales.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Sales ($)', fontsize=10)
ax1.set_ylabel('Predicted Sales ($)', fontsize=10)
ax1.set_title(f'Sales Prediction\nR²={r2:.3f}, MAPE={mape:.1f}%', fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Feature Importance for Sales
ax2 = fig.add_subplot(gs[0, 1])
top_features = feature_importance_sales.head(10)
ax2.barh(range(len(top_features)), top_features['importance'])
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'], fontsize=8)
ax2.set_xlabel('Importance', fontsize=10)
ax2.set_title('Top 10 Features - Sales', fontsize=11)
ax2.invert_yaxis()

# Plot 3: Confusion Matrix for Returns
ax3 = fig.add_subplot(gs[0, 2])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_xlabel('Predicted', fontsize=10)
ax3.set_ylabel('Actual', fontsize=10)
ax3.set_title(f'Return Prediction\nAcc={accuracy:.2f}, F1={f1:.2f}', fontsize=11)

# Plot 4: Actual vs Predicted Ratings
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test_rat, y_pred_rat, alpha=0.5, s=30)
ax4.plot([y_test_rat.min(), y_test_rat.max()], 
         [y_test_rat.min(), y_test_rat.max()], 'r--', lw=2)
ax4.set_xlabel('Actual Rating', fontsize=10)
ax4.set_ylabel('Predicted Rating', fontsize=10)
ax4.set_title(f'Rating Prediction\nR²={r2_rat:.3f}, MAE={mae_rat:.2f}', fontsize=11)
ax4.grid(True, alpha=0.3)

# Plot 5: Sales by Markdown Level
ax5 = fig.add_subplot(gs[1, 1])
markdown_sales = df.groupby('markdown_bracket')['total_sales'].mean().sort_index()
ax5.bar(range(len(markdown_sales)), markdown_sales.values, color='steelblue')
ax5.set_xticks(range(len(markdown_sales)))
ax5.set_xticklabels(markdown_sales.index, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Avg Sales ($)', fontsize=10)
ax5.set_title('Sales by Markdown Level', fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Return Rate by Category
ax6 = fig.add_subplot(gs[1, 2])
return_by_cat = df.groupby('category')['is_returned'].mean().sort_values(ascending=False) * 100
ax6.barh(range(len(return_by_cat)), return_by_cat.values, color='coral')
ax6.set_yticks(range(len(return_by_cat)))
ax6.set_yticklabels(return_by_cat.index, fontsize=9)
ax6.set_xlabel('Return Rate (%)', fontsize=10)
ax6.set_title('Return Rate by Category', fontsize=11)
ax6.invert_yaxis()

# Plot 7: Sales Distribution
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(df['total_sales'], bins=50, color='green', alpha=0.7, edgecolor='black')
ax7.axvline(df['total_sales'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["total_sales"].mean():.0f}')
ax7.set_xlabel('Total Sales ($)', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Sales Distribution', fontsize=11)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')

# Plot 8: Rating Distribution
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(df['customer_rating'].dropna(), bins=20, color='purple', alpha=0.7, edgecolor='black')
ax8.axvline(df['customer_rating'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["customer_rating"].mean():.2f}')
ax8.set_xlabel('Customer Rating', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.set_title('Rating Distribution', fontsize=11)
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3, axis='y')

# Plot 9: Category Performance
ax9 = fig.add_subplot(gs[2, 2])
cat_sales = df.groupby('category')['total_sales'].sum().sort_values(ascending=True)
ax9.barh(range(len(cat_sales)), cat_sales.values, color='teal')
ax9.set_yticks(range(len(cat_sales)))
ax9.set_yticklabels(cat_sales.index, fontsize=9)
ax9.set_xlabel('Total Sales ($)', fontsize=10)
ax9.set_title('Total Sales by Category', fontsize=11)

plt.suptitle('Clothing Store Predictive Analytics Dashboard', fontsize=14, fontweight='bold', y=0.995)
plt.savefig('predictive_analytics_dashboard.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive dashboard saved as 'predictive_analytics_dashboard.png'")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All Models Trained with Full Evaluation Metrics")
print("="*80)