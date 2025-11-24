import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read the original CSV
df = pd.read_csv("ETL/dataset_ele_5_cleaned.csv")

# Use the correct column names
date_column = 'purchase_date'
sales_column = 'total_sales'

# Convert date column to datetime
df[date_column] = pd.to_datetime(df[date_column])

# Define realistic seasonal patterns for electronics/retail
# Peak seasons: November-December (holiday shopping), Back-to-school (August-September)
# Low seasons: January-February (post-holiday), June-July (summer slowdown)
seasonal_weights = {
    1: 0.75,   # January - Post-holiday slump
    2: 0.70,   # February - Lowest (shortest month, post-holiday)
    3: 0.85,   # March - Recovery begins
    4: 0.90,   # April - Spring steady
    5: 0.95,   # May - Pre-summer
    6: 0.85,   # June - Summer slowdown starts
    7: 0.80,   # July - Summer low
    8: 1.10,   # August - Back-to-school boost
    9: 1.05,   # September - Back-to-school continues
    10: 1.00,  # October - Steady, pre-holiday
    11: 1.35,  # November - Black Friday/Holiday season
    12: 1.40   # December - Peak holiday shopping
}

# Calculate total sales
total_sales = df[sales_column].sum()

# Get the date range
min_date = df[date_column].min()
max_date = df[date_column].max()

# Generate monthly targets based on seasonal weights
months_in_range = []
current_date = min_date.replace(day=1)
end_date = max_date.replace(day=1)

while current_date <= end_date:
    months_in_range.append({
        'year': current_date.year,
        'month': current_date.month,
        'weight': seasonal_weights[current_date.month]
    })
    # Move to next month
    if current_date.month == 12:
        current_date = current_date.replace(year=current_date.year + 1, month=1)
    else:
        current_date = current_date.replace(month=current_date.month + 1)

# Calculate total weight and target sales per month
total_weight = sum(m['weight'] for m in months_in_range)
for month_info in months_in_range:
    month_info['target_sales'] = (month_info['weight'] / total_weight) * total_sales

# Create a function to assign sales to appropriate months
def assign_month_target(row):
    year = row[date_column].year
    month = row[date_column].month
    
    for m in months_in_range:
        if m['year'] == year and m['month'] == month:
            return m['target_sales']
    return 0

# Group by year-month and get current sales
df['YearMonth'] = df[date_column].dt.to_period('M')
current_monthly = df.groupby('YearMonth')[sales_column].sum()

# Calculate target for each month
monthly_targets = {}
for month_info in months_in_range:
    period = pd.Period(year=month_info['year'], month=month_info['month'], freq='M')
    monthly_targets[period] = month_info['target_sales']

# Redistribute sales proportionally within each month
df['MonthTarget'] = df['YearMonth'].map(monthly_targets)
df['MonthCurrentTotal'] = df.groupby('YearMonth')[sales_column].transform('sum')
df['ScaleFactor'] = df['MonthTarget'] / df['MonthCurrentTotal']

# Apply the scale factor to adjust individual transaction amounts
df['total_sales_adjusted'] = df[sales_column] * df['ScaleFactor']

# Round to 2 decimal places
df['total_sales_adjusted'] = df['total_sales_adjusted'].round(2)

# Replace the original total_sales column with adjusted values
df[sales_column] = df['total_sales_adjusted']

# Clean up temporary columns
df = df.drop(['YearMonth', 'MonthTarget', 'MonthCurrentTotal', 'ScaleFactor', 'total_sales_adjusted'], axis=1)

# Save the adjusted dataset
output_file = "ETL/dataset_ele_5_cleaned_adjusted.csv"
df.to_csv(output_file, index=False)

print("Sales redistribution complete!")
print("\nMonthly Sales Summary:")
print("=" * 60)

# Display monthly summary
df['YearMonth'] = df[date_column].dt.to_period('M')
monthly_summary = df.groupby('YearMonth')[sales_column].sum().reset_index()
monthly_summary.columns = ['Month', 'Total Sales']
monthly_summary['% of Total'] = (monthly_summary['Total Sales'] / monthly_summary['Total Sales'].sum() * 100)

total_sum = monthly_summary['Total Sales'].sum()

print(f"{'Month':<15} {'Total Sales ($)':>20} {'% of Total':>15}")
print("-" * 60)
for _, row in monthly_summary.iterrows():
    print(f"{str(row['Month']):<15} {row['Total Sales']:>20,.2f} {row['% of Total']:>14.2f}%")
print("-" * 60)
print(f"{'TOTAL':<15} {total_sum:>20,.2f} {100.00:>14.2f}%")

print(f"\nAdjusted data saved to: {output_file}")
print("\nSeasonal Pattern Applied:")
print("- Peak Season: November-December (Holiday shopping)")
print("- Strong Season: August-September (Back-to-school)")
print("- Low Season: January-February (Post-holiday slump)")
print("- Moderate: March-May, October (Steady periods)")