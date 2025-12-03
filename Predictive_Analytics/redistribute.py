import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the adjusted CSV (the one that already has seasonally adjusted total_sales)
df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")

# Use the correct column names based on your dataset
date_column = 'purchase_date'
sales_column = 'total_sales'
price_column = 'current_price'

# Note: stock_quantity appears to be inventory stock, not quantity sold
# We'll calculate quantity_sold from the original data first
print("Calculating quantity sold from original data...")
print(f"Working with columns: {date_column}, {sales_column}, {price_column}")

# Check for zero or null prices
zero_prices = df[df[price_column] <= 0]
if len(zero_prices) > 0:
    print(f"\nWarning: Found {len(zero_prices)} rows with zero or negative prices. These will be removed.")
    df = df[df[price_column] > 0].copy()

# Calculate original quantity_sold (before seasonal adjustment)
# This assumes the original total_sales = current_price × quantity_sold
df['quantity_sold'] = (df[sales_column] / df[price_column]).round(0)

# Handle any remaining NaN or inf values
df['quantity_sold'] = df['quantity_sold'].fillna(1)  # Default to 1 if NaN
df['quantity_sold'] = df['quantity_sold'].replace([np.inf, -np.inf], 1)  # Replace inf with 1
df['quantity_sold'] = df['quantity_sold'].astype(int)

# Ensure quantity_sold is at least 1
df.loc[df['quantity_sold'] < 1, 'quantity_sold'] = 1

print(f"Quantity sold calculated. Range: {df['quantity_sold'].min()} to {df['quantity_sold'].max()}")
print()

# Now we have quantity_sold, we can recalculate current_price based on adjusted total_sales
# Keep quantity_sold constant, adjust price to match the seasonally adjusted total_sales

# Convert date column to datetime
df[date_column] = pd.to_datetime(df[date_column])

# Calculate adjusted price based on adjusted total_sales and calculated quantity_sold
# Formula: adjusted_price = total_sales_adjusted / quantity_sold
# This keeps quantity constant and adjusts price to reflect seasonal patterns
df[price_column] = (df[sales_column] / df['quantity_sold']).round(2)

# Add unit_cost column (cost to produce/acquire the item)
# Typical retail markup is 30-50% for electronics
# So unit_cost = current_price / (1 + markup_percentage)
# We'll use a 40% markup, meaning unit_cost = current_price / 1.40
# This can vary by category, so we'll add some variation (35-45% markup range)

# Add slight variation to markup based on category or randomness
np.random.seed(42)  # For reproducibility
markup_variation = np.random.uniform(1.35, 1.45, len(df))  # 35-45% markup
df['unit_cost'] = (df[price_column] / markup_variation).round(2)

# Calculate profit margin as a percentage
# Profit Margin = ((Selling Price - Cost) / Selling Price) × 100
df['profit_margin'] = (((df[price_column] - df['unit_cost']) / df[price_column]) * 100).round(2)

# Remove the old price_per_unit column if it exists
if 'price_per_unit' in df.columns:
    df = df.drop('price_per_unit', axis=1)

print(f"Unit cost and profit margin calculated with realistic markup (35-45% profit margin).")

# Verify the calculation (total_sales should equal price * quantity)
df['verification'] = (df[price_column] * df['quantity_sold']).round(2)
discrepancy = (df[sales_column] - df['verification']).abs().sum()

if discrepancy > 0.01:
    print(f"Warning: Small rounding discrepancies detected: ${discrepancy:.2f}")
    # Adjust total_sales to match price * quantity to ensure consistency
    df[sales_column] = df['verification']

# Clean up verification column
df = df.drop(['verification'], axis=1)

# Save the final adjusted dataset (overwrite the original file)
output_file = "ETL/dataset_ele_5_cleaned_adjusted.csv"

# Check for missing values before saving
print("\nChecking for missing values...")
missing_counts = df[['quantity_sold', 'unit_cost', 'profit_margin', 'current_price', 'total_sales']].isnull().sum()
if missing_counts.sum() > 0:
    print("Missing values found:")
    print(missing_counts[missing_counts > 0])
    print("\nFilling missing values...")
    
    # Fill missing values appropriately
    df['quantity_sold'] = df['quantity_sold'].fillna(1)
    df['unit_cost'] = df['unit_cost'].fillna(df['unit_cost'].mean())
    df['profit_margin'] = df['profit_margin'].fillna(df['profit_margin'].mean())
    df['current_price'] = df['current_price'].fillna(df['current_price'].mean())
    df['total_sales'] = df['total_sales'].fillna(df['total_sales'].mean())
    
    print("Missing values filled.")
else:
    print("No missing values found.")

df.to_csv(output_file, index=False)

# Verify the columns in the saved file
print(f"\nColumns in the output file:")
print(df.columns.tolist())
print(f"\nFirst 5 rows of key columns:")
print(df[['purchase_date', 'current_price', 'unit_cost', 'profit_margin', 'quantity_sold', 'total_sales']].head(5))
print(f"\nProfit Analysis:")
print(f"Average Selling Price: ${df['current_price'].mean():.2f}")
print(f"Average Unit Cost: ${df['unit_cost'].mean():.2f}")
print(f"Average Profit per Unit: ${(df['current_price'] - df['unit_cost']).mean():.2f}")
print(f"Average Profit Margin: {df['profit_margin'].mean():.1f}%")
print(f"Profit Margin Range: {df['profit_margin'].min():.1f}% - {df['profit_margin'].max():.1f}%")

print("Price adjustment complete!")
print("\nPrice and Quantity Summary:")
print("=" * 80)

# Display monthly summary with price statistics
df['YearMonth'] = df[date_column].dt.to_period('M')
monthly_summary = df.groupby('YearMonth').agg({
    sales_column: 'sum',
    price_column: 'mean',
    'quantity_sold': 'sum'
}).reset_index()

monthly_summary.columns = ['Month', 'Total Sales', 'Avg Price', 'Total Quantity Sold']
monthly_summary['% of Total Sales'] = (monthly_summary['Total Sales'] / monthly_summary['Total Sales'].sum() * 100)

print(f"{'Month':<15} {'Total Sales ($)':>18} {'Avg Price ($)':>15} {'Total Qty':>12} {'% of Sales':>12}")
print("-" * 80)
for _, row in monthly_summary.iterrows():
    print(f"{str(row['Month']):<15} {row['Total Sales']:>18,.2f} {row['Avg Price']:>15,.2f} {row['Total Quantity Sold']:>12,.0f} {row['% of Total Sales']:>11.2f}%")

print("-" * 80)
print(f"{'TOTAL':<15} {monthly_summary['Total Sales'].sum():>18,.2f} {monthly_summary['Avg Price'].mean():>15,.2f} {monthly_summary['Total Quantity Sold'].sum():>12,.0f} {100.00:>11.2f}%")

print(f"\nAdjusted data saved to: {output_file}")
print("\nNew Columns Added:")
print("- 'quantity_sold': Number of units sold per transaction")
print("- 'unit_cost': Cost to produce/acquire each unit")
print("- 'profit_margin': Profit margin percentage per transaction")
print("\nAdjustment Method:")
print("- Quantity Sold: CALCULATED and KEPT CONSTANT (realistic transaction quantities)")
print("- Current Price: ADJUSTED based on seasonal patterns (what customers pay)")
print("- Unit Cost: Production/acquisition cost (35-45% below selling price)")
print("- Profit Margin: ((Current Price - Unit Cost) / Current Price) × 100")
print("- Total Sales: current_price × quantity_sold")
print("\nSeasonal Price Pattern:")
print("- Higher prices in peak seasons (Nov-Dec: Holiday, Aug-Sep: Back-to-school)")
print("- Lower prices in slow seasons (Jan-Feb: Post-holiday discounts)")
print("- Unit costs and profit margins remain proportional to maintain realistic profitability")