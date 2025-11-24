import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Read the data
df = pd.read_csv("ETL/dataset_ele_5_cleaned_adjusted.csv")
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Sort by date
df = df.sort_values('purchase_date')

# Create daily aggregation
daily_sales = df.groupby('purchase_date')['total_sales'].sum().reset_index()

# Create monthly aggregation
df['year_month'] = df['purchase_date'].dt.to_period('M')
monthly_sales = df.groupby('year_month')['total_sales'].sum().reset_index()
monthly_sales['year_month_str'] = monthly_sales['year_month'].astype(str)

# Calculate cumulative sales
daily_sales['cumulative_sales'] = daily_sales['total_sales'].cumsum()

# ============================================================================
# DISCOUNT ANALYSIS PREPARATION
# ============================================================================
# Create discount flag using markdown_percentage
df['has_discount'] = df['markdown_percentage'] > 0

# Calculate revenue metrics using actual columns
df['revenue_without_discount'] = df['original_price']  # Original price before discount
df['discount_amount'] = df['original_price'] - df['current_price']  # Money lost to discount
df['discount_revenue'] = df['total_sales'].where(df['has_discount'], 0)  # Revenue from discounted items

print(f"✓ Using columns: markdown_percentage, original_price, current_price")
print(f"✓ Found {df['has_discount'].sum():,} transactions with discounts ({df['has_discount'].sum()/len(df)*100:.1f}%)")

# ============================================================================
# CHART 1: DAILY SALES TREND (Improved Readability)
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 6))

# Plot bars with reduced alpha for better visibility
ax.bar(daily_sales['purchase_date'], daily_sales['total_sales'], 
       color='steelblue', alpha=0.6, width=1, label='Daily Sales')

# Calculate and plot moving average for trend
window = 7  # 7-day moving average
daily_sales['ma_7'] = daily_sales['total_sales'].rolling(window=window, center=True).mean()
ax.plot(daily_sales['purchase_date'], daily_sales['ma_7'], 
        color='red', linewidth=2.5, label=f'{window}-Day Moving Average', linestyle='--')

# Formatting
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Daily Sales Trend with Moving Average', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
print("✓ Chart 1: Daily Sales Trend displayed")

# ============================================================================
# CHART 2: MONTHLY SALES TREND (Stacked Bar Chart by Brand)
# ============================================================================
# Create pivot table for monthly sales by brand
df['year_month'] = df['purchase_date'].dt.to_period('M')
monthly_brand_sales = df.pivot_table(
    index='year_month',
    columns='brand',
    values='total_sales',
    aggfunc='sum',
    fill_value=0
)

# Get top 8 brands by total sales, group rest as "Others"
top_brands = df.groupby('brand')['total_sales'].sum().nlargest(8).index.tolist()
monthly_brand_sales_top = monthly_brand_sales[top_brands].copy()
monthly_brand_sales_top['Others'] = monthly_brand_sales.drop(columns=top_brands).sum(axis=1)

fig, ax = plt.subplots(figsize=(16, 8))

# Define color palette
colors = plt.cm.Set3(range(len(monthly_brand_sales_top.columns)))

# Create stacked bar chart
monthly_brand_sales_top.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    color=colors,
    edgecolor='white',
    linewidth=0.5,
    width=0.85
)

# Add total value labels on top of stacked bars
for i, (idx, row) in enumerate(monthly_brand_sales_top.iterrows()):
    total = row.sum()
    ax.text(i, total, f'${total:,.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Formatting
ax.set_xlabel('Month', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Monthly Sales Trend by Brand (Stacked)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels([str(x) for x in monthly_brand_sales_top.index], rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Improve legend
ax.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left', 
          fontsize=10, title_fontsize=11, frameon=True, shadow=True)

plt.tight_layout()
plt.show()
print("✓ Chart 2: Monthly Sales Stacked by Brand displayed")

# ============================================================================
# CHART 3: MONTHLY SALES GROWTH (Line Chart with Markers)
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

# Plot line with markers
line = ax.plot(range(len(monthly_sales)), monthly_sales['total_sales'], 
               color='green', linewidth=3, marker='o', markersize=10, 
               markerfacecolor='lightgreen', markeredgecolor='darkgreen', 
               markeredgewidth=2, label='Monthly Sales')

# Highlight max and min
max_idx = monthly_sales['total_sales'].idxmax()
min_idx = monthly_sales['total_sales'].idxmin()

ax.scatter(max_idx, monthly_sales.loc[max_idx, 'total_sales'], 
           color='red', s=300, marker='o', zorder=5, edgecolor='darkred', linewidth=2)
ax.scatter(min_idx, monthly_sales.loc[min_idx, 'total_sales'], 
           color='blue', s=300, marker='o', zorder=5, edgecolor='darkblue', linewidth=2)

# Add annotations for max and min
ax.annotate(f'Peak: ${monthly_sales.loc[max_idx, "total_sales"]:,.0f}', 
            xy=(max_idx, monthly_sales.loc[max_idx, 'total_sales']),
            xytext=(max_idx, monthly_sales.loc[max_idx, 'total_sales'] * 1.08),
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.annotate(f'Low: ${monthly_sales.loc[min_idx, "total_sales"]:,.0f}', 
            xy=(min_idx, monthly_sales.loc[min_idx, 'total_sales']),
            xytext=(min_idx, monthly_sales.loc[min_idx, 'total_sales'] * 0.85),
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

# Formatting
ax.set_xlabel('Month', fontsize=13, fontweight='bold')
ax.set_ylabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Monthly Sales Growth Pattern', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(monthly_sales)))
ax.set_xticklabels(monthly_sales['year_month_str'], rotation=45, ha='right')
ax.grid(True, alpha=0.3)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.show()
print("✓ Chart 3: Monthly Growth displayed")

# ============================================================================
# CHART 4: CUMULATIVE SALES OVER TIME
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(daily_sales['purchase_date'], daily_sales['cumulative_sales'], 
                color='mediumpurple', alpha=0.4)
ax.plot(daily_sales['purchase_date'], daily_sales['cumulative_sales'], 
        color='purple', linewidth=2.5, label='Cumulative Sales')

# Add total annotation
total_sales = daily_sales['cumulative_sales'].iloc[-1]
ax.text(0.98, 0.95, f'Total: ${total_sales:,.0f}', 
        transform=ax.transAxes, fontsize=14, fontweight='bold',
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7))

# Formatting
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Cumulative Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Cumulative Sales Over Time', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Format y-axis
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

plt.tight_layout()
plt.show()
print("✓ Chart 4: Cumulative Sales displayed")

# ============================================================================
# CHART 5: SEASONAL SALES COMPARISON
# ============================================================================
# Map months to seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['purchase_date'].dt.month.apply(get_season)
seasonal_sales = df.groupby('season')['total_sales'].sum().reindex(['Spring', 'Summer', 'Fall', 'Winter'])

fig, ax = plt.subplots(figsize=(12, 7))

colors = ['#90EE90', '#FFD700', '#FF8C00', '#4682B4']
bars = ax.bar(seasonal_sales.index, seasonal_sales.values, color=colors, 
              edgecolor='black', linewidth=2, alpha=0.8)

# Add value labels
for bar, value in zip(bars, seasonal_sales.values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${value:,.0f}\n({value/seasonal_sales.sum()*100:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Sales by Season', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
print("✓ Chart 5: Seasonal Sales displayed")

# ============================================================================
# CHART 6: TOP CATEGORIES BY SALES
# ============================================================================
category_sales = df.groupby('category')['total_sales'].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(range(len(category_sales)), category_sales.values, 
               color='teal', edgecolor='darkslategray', linewidth=1.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, category_sales.values)):
    ax.text(value, i, f'  ${value:,.0f}',
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(category_sales)))
ax.set_yticklabels(category_sales.index, fontsize=11)
ax.set_xlabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Sales by Product Category', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
print("✓ Chart 6: Category Sales displayed")

# ============================================================================
# CHART 7: TOP BRANDS BY SALES
# ============================================================================
brand_sales = df.groupby('brand')['total_sales'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(range(len(brand_sales)), brand_sales.values, 
               color='royalblue', edgecolor='navy', linewidth=1.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, brand_sales.values)):
    ax.text(value, i, f'  ${value:,.0f}',
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(brand_sales)))
ax.set_yticklabels(brand_sales.index, fontsize=11)
ax.set_xlabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Brands by Sales', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.show()
print("✓ Chart 7: Brand Sales displayed")

# ============================================================================
# CHART 8: RETURN RATE ANALYSIS
# ============================================================================
return_analysis = df.groupby('is_returned').agg({
    'total_sales': 'sum',
    'product_id': 'count'
}).rename(columns={'product_id': 'count'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart for return count
labels = ['Not Returned', 'Returned']
colors = ['#2ecc71', '#e74c3c']
explode = (0, 0.1)

ax1.pie(return_analysis['count'], labels=labels, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Return Rate by Transaction Count', fontsize=14, fontweight='bold', pad=20)

# Bar chart for sales impact
ax2.bar(labels, return_analysis['total_sales'], color=colors, 
        edgecolor='black', linewidth=2)
for i, value in enumerate(return_analysis['total_sales']):
    ax2.text(i, value, f'${value:,.0f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_ylabel('Total Sales ($)', fontsize=12, fontweight='bold')
ax2.set_title('Sales by Return Status', fontsize=14, fontweight='bold', pad=20)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
print("✓ Chart 8: Return Analysis displayed")

# ============================================================================
# CHART 9: SALES BY COUNTRY (TOP 10)
# ============================================================================
country_sales = df.groupby('country')['total_sales'].sum().sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(range(len(country_sales)), country_sales.values, 
               color='mediumseagreen', edgecolor='darkgreen', linewidth=1.5)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, country_sales.values)):
    ax.text(value, i, f'  ${value:,.0f}',
            va='center', fontsize=10, fontweight='bold')

ax.set_yticks(range(len(country_sales)))
ax.set_yticklabels(country_sales.index, fontsize=11)
ax.set_xlabel('Total Sales ($)', fontsize=13, fontweight='bold')
ax.set_title('Top 10 Countries by Sales', fontsize=16, fontweight='bold', pad=20)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.show()
print("✓ Chart 9: Country Sales displayed")

# ============================================================================
# NEW CHART 10: DISCOUNT vs NO DISCOUNT - UNITS SOLD COMPARISON
# ============================================================================
discount_units = df.groupby('has_discount')['product_id'].count()

fig, ax = plt.subplots(figsize=(10, 7))

labels = ['No Discount', 'With Discount']
colors = ['#3498db', '#e67e22']
explode = (0.05, 0.05)

wedges, texts, autotexts = ax.pie(discount_units.values, labels=labels, autopct='%1.1f%%',
                                    colors=colors, explode=explode, startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})

# Add unit counts in the center
total_units = discount_units.sum()
ax.text(0, 0, f'Total Units\n{total_units:,}', 
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

ax.set_title('Units Sold: With vs Without Discount', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()
print("✓ Chart 10: Discount Units Comparison displayed")

# ============================================================================
# NEW CHART 11: REVENUE BREAKDOWN - DISCOUNTED vs NON-DISCOUNTED
# ============================================================================
revenue_comparison = df.groupby('has_discount').agg({
    'total_sales': 'sum',
    'product_id': 'count'
}).rename(columns={'product_id': 'units_sold'})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart comparing revenue
labels = ['No Discount', 'With Discount']
colors = ['#27ae60', '#e74c3c']

bars1 = ax1.bar(labels, revenue_comparison['total_sales'].values, color=colors, 
                edgecolor='black', linewidth=2, alpha=0.85)

for bar, value in zip(bars1, revenue_comparison['total_sales'].values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'${value:,.0f}\n({value/revenue_comparison["total_sales"].sum()*100:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('Total Revenue ($)', fontsize=13, fontweight='bold')
ax1.set_title('Revenue: With vs Without Discount', fontsize=14, fontweight='bold', pad=15)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.grid(True, alpha=0.3, axis='y')

# Bar chart comparing units sold
bars2 = ax2.bar(labels, revenue_comparison['units_sold'].values, color=colors, 
                edgecolor='black', linewidth=2, alpha=0.85)

for bar, value in zip(bars2, revenue_comparison['units_sold'].values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:,}\n({value/revenue_comparison["units_sold"].sum()*100:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_ylabel('Units Sold', fontsize=13, fontweight='bold')
ax2.set_title('Units Sold: With vs Without Discount', fontsize=14, fontweight='bold', pad=15)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
print("✓ Chart 11: Revenue & Units Breakdown displayed")

# ============================================================================
# NEW CHART 12: DISCOUNT IMPACT ANALYSIS - WATERFALL CHART
# ============================================================================
total_revenue_before_discount = df['revenue_without_discount'].sum()
total_discount_amount = df['discount_amount'].sum()
actual_revenue = df['total_sales'].sum()

fig, ax = plt.subplots(figsize=(14, 8))

categories = ['Potential\nRevenue', 'Discount\nGiven', 'Actual\nRevenue']
values = [total_revenue_before_discount, -total_discount_amount, actual_revenue]
colors_waterfall = ['#2ecc71', '#e74c3c', '#3498db']

# Create waterfall effect
x_pos = [0, 1, 2]
y_start = [0, total_revenue_before_discount, 0]
y_height = [total_revenue_before_discount, -total_discount_amount, actual_revenue]

for i in range(len(categories)):
    ax.bar(x_pos[i], y_height[i], bottom=y_start[i], color=colors_waterfall[i], 
           edgecolor='black', linewidth=2, width=0.6, alpha=0.85)
    
    # Add value labels
    if i == 1:  # Discount (negative)
        label_y = y_start[i] + y_height[i]/2
        ax.text(x_pos[i], label_y, f'-${abs(y_height[i]):,.0f}',
                ha='center', va='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    else:
        label_y = y_start[i] + y_height[i] + total_revenue_before_discount * 0.02
        ax.text(x_pos[i], label_y, f'${y_height[i]:,.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

# Draw connecting lines
ax.plot([0.3, 0.7], [total_revenue_before_discount, total_revenue_before_discount], 
        'k--', linewidth=1.5, alpha=0.5)
ax.plot([1.3, 1.7], [actual_revenue, actual_revenue], 
        'k--', linewidth=1.5, alpha=0.5)

ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylabel('Revenue ($)', fontsize=13, fontweight='bold')
ax.set_title('Discount Impact on Revenue (Waterfall Analysis)', fontsize=16, fontweight='bold', pad=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax.grid(True, alpha=0.3, axis='y')

# Add summary box
summary_text = f'Discount Rate: {(total_discount_amount/total_revenue_before_discount)*100:.1f}%\nRevenue Retained: {(actual_revenue/total_revenue_before_discount)*100:.1f}%'
ax.text(0.98, 0.97, summary_text, transform=ax.transAxes,
        fontsize=11, fontweight='bold', ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()
print("✓ Chart 12: Discount Impact Waterfall displayed")

# ============================================================================
# NEW CHART 13: MONTHLY DISCOUNT TREND
# ============================================================================
monthly_discount_analysis = df.groupby('year_month').agg({
    'discount_amount': 'sum',
    'total_sales': 'sum',
    'revenue_without_discount': 'sum'
}).reset_index()

monthly_discount_analysis['discount_rate'] = (monthly_discount_analysis['discount_amount'] / 
                                               monthly_discount_analysis['revenue_without_discount'] * 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Top chart: Stacked area for revenue composition
x_range = range(len(monthly_discount_analysis))
ax1.fill_between(x_range, monthly_discount_analysis['total_sales'], 
                 color='#27ae60', alpha=0.7, label='Actual Revenue')
ax1.fill_between(x_range, monthly_discount_analysis['total_sales'],
                 monthly_discount_analysis['revenue_without_discount'],
                 color='#e74c3c', alpha=0.5, label='Discount Amount')

ax1.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
ax1.set_title('Monthly Revenue: Actual vs Potential (with Discount Loss)', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11, loc='upper left')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x_range)
ax1.set_xticklabels([str(x) for x in monthly_discount_analysis['year_month']], rotation=45, ha='right')

# Bottom chart: Discount rate trend
ax2.plot(x_range, monthly_discount_analysis['discount_rate'], 
         color='#e67e22', linewidth=3, marker='o', markersize=8, 
         markerfacecolor='#f39c12', markeredgecolor='#d35400', markeredgewidth=2)
ax2.fill_between(x_range, monthly_discount_analysis['discount_rate'], 
                 alpha=0.3, color='#e67e22')

# Add average line
avg_discount_rate = monthly_discount_analysis['discount_rate'].mean()
ax2.axhline(y=avg_discount_rate, color='red', linestyle='--', linewidth=2, 
            label=f'Average: {avg_discount_rate:.1f}%')

ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
ax2.set_ylabel('Discount Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Monthly Discount Rate Trend', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(x_range)
ax2.set_xticklabels([str(x) for x in monthly_discount_analysis['year_month']], rotation=45, ha='right')

plt.tight_layout()
plt.show()
print("✓ Chart 13: Monthly Discount Trend displayed")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("COMPREHENSIVE SALES ANALYSIS SUMMARY")
print("="*70)
print(f"\n📊 OVERALL METRICS")
print(f"  Total Sales: ${total_sales:,.2f}")
print(f"  Total Transactions: {len(df):,}")
print(f"  Average Transaction Value: ${df['total_sales'].mean():,.2f}")

print(f"\n📅 DAILY METRICS")
print(f"  Average Daily Sales: ${daily_sales['total_sales'].mean():,.2f}")
print(f"  Highest Daily Sales: ${daily_sales['total_sales'].max():,.2f}")
print(f"  Lowest Daily Sales: ${daily_sales['total_sales'].min():,.2f}")

print(f"\n📅 MONTHLY METRICS")
print(f"  Average Monthly Sales: ${monthly_sales['total_sales'].mean():,.2f}")
print(f"  Highest Month: {monthly_sales.loc[max_idx, 'year_month_str']} (${monthly_sales['total_sales'].max():,.2f})")
print(f"  Lowest Month: {monthly_sales.loc[min_idx, 'year_month_str']} (${monthly_sales['total_sales'].min():,.2f})")

print(f"\n🌍 GEOGRAPHIC")
print(f"  Countries: {df['country'].nunique()}")
print(f"  Top Country: {country_sales.index[0]} (${country_sales.values[0]:,.2f})")

print(f"\n📦 PRODUCTS")
print(f"  Categories: {df['category'].nunique()}")
print(f"  Brands: {df['brand'].nunique()}")
print(f"  Top Category: {category_sales.index[0]} (${category_sales.values[0]:,.2f})")

print(f"\n↩️ RETURNS")
print(f"  Return Rate: {(df['is_returned'].sum() / len(df) * 100):.2f}%")
print(f"  Returned Transactions: {df['is_returned'].sum():,}")

print("\n" + "="*70)
print("💰 DISCOUNT ANALYSIS")
print("="*70)
print(f"\n📊 UNITS SOLD")
print(f"  Total Units Sold: {len(df):,}")
print(f"  Units WITHOUT Discount: {(~df['has_discount']).sum():,} ({(~df['has_discount']).sum()/len(df)*100:.1f}%)")
print(f"  Units WITH Discount: {df['has_discount'].sum():,} ({df['has_discount'].sum()/len(df)*100:.1f}%)")

print(f"\n💵 REVENUE BREAKDOWN")
print(f"  Potential Revenue (before discounts): ${total_revenue_before_discount:,.2f}")
print(f"  Actual Revenue (after discounts): ${actual_revenue:,.2f}")
print(f"  Revenue from NON-Discounted Sales: ${df[~df['has_discount']]['total_sales'].sum():,.2f}")
print(f"  Revenue from Discounted Sales: ${df[df['has_discount']]['total_sales'].sum():,.2f}")

print(f"\n📉 DISCOUNT IMPACT")
print(f"  Total Discount Amount Given: ${total_discount_amount:,.2f}")
print(f"  Average Discount per Transaction: ${df['discount_amount'].mean():,.2f}")
print(f"  Overall Discount Rate: {(total_discount_amount/total_revenue_before_discount)*100:.2f}%")
print(f"  Revenue Retention Rate: {(actual_revenue/total_revenue_before_discount)*100:.2f}%")

print(f"\n💡 DISCOUNT EFFECTIVENESS")
discounted_avg = df[df['has_discount']]['total_sales'].mean()
non_discounted_avg = df[~df['has_discount']]['total_sales'].mean()
print(f"  Avg Transaction Value (WITH discount): ${discounted_avg:,.2f}")
print(f"  Avg Transaction Value (WITHOUT discount): ${non_discounted_avg:,.2f}")
print(f"  Difference: ${discounted_avg - non_discounted_avg:,.2f}")

# Calculate ROI of discounts (revenue gained vs discount given)
discount_revenue_gained = df[df['has_discount']]['total_sales'].sum()
discount_cost = df[df['has_discount']]['discount_amount'].sum()
if discount_cost > 0:
    discount_roi = (discount_revenue_gained / discount_cost) * 100
    print(f"\n🎯 DISCOUNT ROI")
    print(f"  Revenue from Discounted Items: ${discount_revenue_gained:,.2f}")
    print(f"  Cost of Discounts: ${discount_cost:,.2f}")
    print(f"  ROI: {discount_roi:.1f}% (${discount_roi/100:.2f} revenue per $1 discount)")

print("\n" + "="*70)
print("✅ All 13 charts displayed successfully!")
print("="*70)