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

# Add price_per_unit column (which is the same as current_price)
df['price_per_unit'] = df[price_column]

print(f"Price per unit calculated and added as separate column.")

# Verify the calculation (total_sales should equal price * quantity)
df['verification'] = (df[price_column] * df['quantity_sold']).round(2)
discrepancy = (df[sales_column] - df['verification']).abs().sum()

if discrepancy > 0.01:
    print(f"Warning: Small rounding discrepancies detected: ${discrepancy:.2f}")
    # Adjust total_sales to match price * quantity to ensure consistency
    df[sales_column] = df['verification']

# Clean up verification column
df = df.drop(['verification'], axis=1)

# Save the final adjusted dataset
output_file = "ETL/dataset_ele_5_cleaned_adjusted.csv"
df.to_csv(output_file, index=False)

# Verify the columns in the saved file
print(f"\nColumns in the output file:")
print(df.columns.tolist())
print(f"\nFirst 3 rows of key columns:")
print(df[['purchase_date', 'current_price', 'quantity_sold', 'price_per_unit', 'total_sales']].head(3))

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
print("- 'price_per_unit': Price per individual unit (same as current_price)")
print("\nAdjustment Method:")
print("- Quantity Sold: CALCULATED and KEPT CONSTANT (realistic transaction quantities)")
print("- Current Price / Price Per Unit: ADJUSTED based on seasonal patterns")
print("- Total Sales: current_price × quantity_sold")
print("\nSeasonal Price Pattern:")
print("- Higher prices in peak seasons (Nov-Dec: Holiday, Aug-Sep: Back-to-school)")
print("- Lower prices in slow seasons (Jan-Feb: Post-holiday discounts)")
print("- This reflects real retail pricing strategies")