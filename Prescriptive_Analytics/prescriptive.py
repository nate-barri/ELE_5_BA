import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.optimize import minimize, linprog, minimize_scalar
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

from scipy.optimize import milp, LinearConstraint, Bounds

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("=" * 100)
print(" " * 20 + "PRESCRIPTIVE ANALYTICS - OPTIMIZATION MODELS")
print("=" * 100)

PRESCRIPTIVE_MODELS = {
    "Classical Inventory Models": [],
    "Optimization Models": [],
    "Stochastic Models": [],
    "Pricing Models": []
}

def register_model(category, model_name, description):
    """Register a prescriptive model used"""
    PRESCRIPTIVE_MODELS[category].append({"name": model_name, "description": description})


# =============================================================================
# DATA LOADING - From Predictive Analytics folder
# =============================================================================

# OLD VERSION - Loading from ETL folder (DISABLED)
# def load_data():
#     """Load and preprocess data from ETL folder"""
#     csv_file = "ETL/dataset_ele_5_cleaned_adjusted.csv"
#     df = pd.read_csv(csv_file)
#     ...

# NEW VERSION - Loading from Predictive_Analytics folder
def load_data():
    """Load data from Predictive_Analytics folder"""
    csv_file = "../ETL/dataset_ele_5_cleaned_adjusted.csv"
    
    df = pd.read_csv(csv_file)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df = df.sort_values('purchase_date').reset_index(drop=True)
    
    # Create features
    df['has_discount'] = df['markdown_percentage'] > 0
    df['discount_amount'] = df['original_price'] - df['current_price']
    
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
    df['year_month'] = df['purchase_date'].dt.to_period('M')
    df['month'] = df['purchase_date'].dt.month
    df['day_of_week'] = df['purchase_date'].dt.dayofweek
    df['quarter'] = df['purchase_date'].dt.quarter
    df['week_of_year'] = df['purchase_date'].dt.isocalendar().week.astype(int)
    
    print(f"\nDataset loaded: {len(df):,} transactions")
    print(f"Date Range: {df['purchase_date'].min().strftime('%Y-%m-%d')} to {df['purchase_date'].max().strftime('%Y-%m-%d')}")
    
    return df


# =============================================================================
# HELPER: Compute demand statistics for prescriptive models
# =============================================================================

def compute_demand_statistics(df):
    """Compute demand statistics needed for prescriptive models"""
    print("\n" + "=" * 100)
    print("COMPUTING DEMAND STATISTICS FOR PRESCRIPTIVE MODELS")
    print("=" * 100)
    
    # Product-level demand statistics
    product_demand = df.groupby('product_id').agg({
        'total_sales': ['sum', 'mean', 'std', 'count'],
        'current_price': 'mean',
        'original_price': 'mean',
        'markdown_percentage': 'mean',
        'unit_cost': 'mean',
        'profit_margin': 'mean',
        'stock_quantity': 'mean',
        'category': 'first',
        'brand': 'first'
    }).reset_index()

    product_demand.columns = ['product_id', 'total_revenue', 'mean_daily_revenue', 'std_daily_revenue',
                               'num_transactions', 'avg_price', 'original_price', 'avg_markdown',
                               'avg_unit_cost', 'avg_profit_margin', 'avg_stock_quantity',
                               'category', 'brand']
    
    # Calculate units sold (approximation)
    product_demand['units_sold'] = product_demand['total_revenue'] / product_demand['avg_price']
    product_demand['mean_daily_demand'] = product_demand['units_sold'] / product_demand['num_transactions']
    product_demand['std_daily_demand'] = product_demand['std_daily_revenue'] / product_demand['avg_price']
    product_demand['std_daily_demand'] = product_demand['std_daily_demand'].fillna(
        product_demand['mean_daily_demand'] * 0.3)  # Default CV of 0.3
    
    # Category-level statistics
    category_demand = df.groupby('category').agg({
        'total_sales': ['sum', 'mean', 'std'],
        'current_price': 'mean',
        'markdown_percentage': 'mean'
    }).reset_index()
    category_demand.columns = ['category', 'total_revenue', 'mean_revenue', 'std_revenue', 
                                'avg_price', 'avg_markdown']
    
    print(f"\nProducts analyzed: {len(product_demand)}")
    print(f"Categories analyzed: {len(category_demand)}")
    
    return product_demand, category_demand


# =============================================================================
# MODEL 1: ECONOMIC ORDER QUANTITY (EOQ)
# Classical inventory optimization model
# =============================================================================

def model_eoq(df, product_demand):
    """
    Economic Order Quantity (EOQ) Model
    
    Formula: EOQ = sqrt(2 * D * S / H)
    Where:
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
    
    This model determines the optimal order quantity that minimizes
    total inventory costs (ordering + holding costs).
    """
    print("\n" + "=" * 100)
    print("MODEL 1: ECONOMIC ORDER QUANTITY (EOQ)")
    print("Objective: Minimize total inventory costs (ordering + holding)")
    print("=" * 100)
    
    register_model("Classical Inventory Models", "Economic Order Quantity (EOQ)",
                   "Determines optimal order quantity minimizing total inventory costs")
    
    # Parameters (using dataset columns)
    # Ordering cost estimated as 10% of average unit cost per order
    ordering_cost_base = product_demand['avg_unit_cost'].mean()
    ordering_cost = ordering_cost_base * 0.1  # 10% of unit cost as ordering cost per order
    # Holding cost rate based on profit margin (conservative estimate)
    holding_cost_rate = product_demand['avg_profit_margin'].mean() * 0.5  # 50% of profit margin
    lead_time_days = 7
    service_level = 0.95
    z_score = norm.ppf(service_level)
    
    # Calculate EOQ for each product
    eoq_results = product_demand.copy()
    
    # Annualize demand (assuming data represents a sample period)
    days_in_data = (df['purchase_date'].max() - df['purchase_date'].min()).days
    eoq_results['annual_demand'] = eoq_results['units_sold'] * (365 / max(days_in_data, 1))
    
    # Holding cost per unit (product-specific using profit margin)
    eoq_results['holding_cost'] = eoq_results['avg_price'] * (eoq_results['avg_profit_margin'] * 0.5)
    
    # EOQ Formula
    eoq_results['eoq'] = np.sqrt(
        (2 * eoq_results['annual_demand'] * ordering_cost) / 
        eoq_results['holding_cost'].replace(0, 0.01)
    )
    
    # Safety Stock (for demand variability)
    eoq_results['safety_stock'] = z_score * eoq_results['std_daily_demand'] * np.sqrt(lead_time_days)
    
    # Reorder Point
    eoq_results['reorder_point'] = (eoq_results['mean_daily_demand'] * lead_time_days) + eoq_results['safety_stock']
    
    # Total Inventory Costs
    eoq_results['annual_ordering_cost'] = (eoq_results['annual_demand'] / eoq_results['eoq'].replace(0, 1)) * ordering_cost
    eoq_results['annual_holding_cost'] = (eoq_results['eoq'] / 2 + eoq_results['safety_stock']) * eoq_results['holding_cost']
    eoq_results['total_inventory_cost'] = eoq_results['annual_ordering_cost'] + eoq_results['annual_holding_cost']
    
    # Display top results
    print("\n[EOQ PRESCRIPTIONS] Top 10 Products:")
    print("-" * 100)
    
    top_products = eoq_results.nlargest(10, 'annual_demand')
    for _, row in top_products.iterrows():
        print(f"\nProduct: {row['product_id']} ({row['category']})")
        print(f"  Annual Demand: {row['annual_demand']:.0f} units")
        print(f"  Optimal Order Quantity (EOQ): {row['eoq']:.0f} units")
        print(f"  Safety Stock: {row['safety_stock']:.0f} units")
        print(f"  Reorder Point: {row['reorder_point']:.0f} units")
        print(f"  Total Annual Cost: ${row['total_inventory_cost']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    top_20 = eoq_results.nlargest(20, 'annual_demand')
    
    axes[0, 0].barh(range(len(top_20)), top_20['eoq'].values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels([str(p)[:15] for p in top_20['product_id']])
    axes[0, 0].set_xlabel('EOQ (units)')
    axes[0, 0].set_title('MODEL: Economic Order Quantity by Product', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    axes[0, 1].barh(range(len(top_20)), top_20['safety_stock'].values, color='#f39c12')
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels([str(p)[:15] for p in top_20['product_id']])
    axes[0, 1].set_xlabel('Safety Stock (units)')
    axes[0, 1].set_title('MODEL: Safety Stock for 95% Service Level', fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    x_pos = np.arange(len(top_20))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, top_20['annual_ordering_cost'], width, label='Ordering Cost', color='#e74c3c')
    axes[1, 0].bar(x_pos + width/2, top_20['annual_holding_cost'], width, label='Holding Cost', color='#3498db')
    axes[1, 0].set_ylabel('Annual Cost ($)')
    axes[1, 0].set_title('MODEL: EOQ Cost Components', fontweight='bold')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([str(p)[:8] for p in top_20['product_id']], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Cost curve for sample product
    sample = top_20.iloc[0]
    Q_range = np.linspace(1, sample['eoq'] * 3, 100)
    ordering = (sample['annual_demand'] / Q_range) * ordering_cost
    holding = (Q_range / 2) * sample['holding_cost']
    total = ordering + holding
    
    axes[1, 1].plot(Q_range, ordering, 'r--', label='Ordering Cost', linewidth=2)
    axes[1, 1].plot(Q_range, holding, 'b--', label='Holding Cost', linewidth=2)
    axes[1, 1].plot(Q_range, total, 'g-', label='Total Cost', linewidth=3)
    axes[1, 1].axvline(x=sample['eoq'], color='black', linestyle=':', linewidth=2, label=f"EOQ = {sample['eoq']:.0f}")
    axes[1, 1].set_xlabel('Order Quantity')
    axes[1, 1].set_ylabel('Annual Cost ($)')
    axes[1, 1].set_title(f"MODEL: EOQ Cost Trade-off (Product: {str(sample['product_id'])[:15]})", fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_eoq.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return eoq_results


# =============================================================================
# MODEL 2: NEWSVENDOR MODEL
# Single-period stochastic inventory model
# =============================================================================

def model_newsvendor(df, product_demand):
    """
    Newsvendor Model (Single-Period Inventory)
    
    Formula: Q* = F^(-1)(Cu / (Cu + Co))
    Where:
        Cu = Cost of underage (lost profit from stockout)
        Co = Cost of overage (loss from excess inventory)
        F^(-1) = Inverse CDF of demand distribution
    
    This model determines optimal order quantity for perishable goods
    or single-period decisions under demand uncertainty.
    """
    print("\n" + "=" * 100)
    print("MODEL 2: NEWSVENDOR MODEL")
    print("Objective: Optimize single-period inventory under demand uncertainty")
    print("=" * 100)
    
    register_model("Stochastic Models", "Newsvendor Model",
                   "Optimal order quantity for single-period inventory under demand uncertainty")
    
    # Parameters (using dataset columns)
    gross_margin = product_demand['avg_profit_margin'].mean()  # Use average profit margin from data
    salvage_rate = min(0.5, product_demand['avg_markdown'].mean() / 100)  # Salvage rate based on markdown percentage
    
    newsvendor_results = product_demand.copy()
    
    # Cost calculations (product-specific)
    newsvendor_results['unit_cost'] = newsvendor_results['avg_unit_cost']  # Use actual unit cost from data
    newsvendor_results['underage_cost'] = newsvendor_results['avg_price'] - newsvendor_results['unit_cost']  # Cu = profit margin
    newsvendor_results['overage_cost'] = newsvendor_results['unit_cost'] * (1 - salvage_rate)  # Co = cost - salvage
    
    # Critical ratio
    newsvendor_results['critical_ratio'] = (
        newsvendor_results['underage_cost'] / 
        (newsvendor_results['underage_cost'] + newsvendor_results['overage_cost'])
    )
    
    # Optimal order quantity (assuming normal distribution)
    newsvendor_results['optimal_quantity'] = (
        newsvendor_results['mean_daily_demand'] + 
        norm.ppf(newsvendor_results['critical_ratio']) * newsvendor_results['std_daily_demand']
    )
    newsvendor_results['optimal_quantity'] = newsvendor_results['optimal_quantity'].clip(lower=0)
    
    # Expected profit calculation
    def expected_profit(row):
        Q = row['optimal_quantity']
        mu = row['mean_daily_demand']
        sigma = row['std_daily_demand']
        Cu = row['underage_cost']
        Co = row['overage_cost']
        
        if sigma <= 0:
            return Q * Cu - max(0, Q - mu) * (Cu + Co)
        
        # Expected sales
        z = (Q - mu) / sigma
        expected_sales = mu - sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
        expected_leftover = Q - expected_sales
        
        profit = Cu * expected_sales - Co * expected_leftover
        return profit
    
    newsvendor_results['expected_profit'] = newsvendor_results.apply(expected_profit, axis=1)
    
    # Service level achieved
    newsvendor_results['achieved_service_level'] = newsvendor_results['critical_ratio']
    
    # Display results
    print("\n[NEWSVENDOR PRESCRIPTIONS] Top 10 Products:")
    print("-" * 100)
    
    top_products = newsvendor_results.nlargest(10, 'expected_profit')
    for _, row in top_products.iterrows():
        print(f"\nProduct: {row['product_id']} ({row['category']})")
        print(f"  Mean Demand: {row['mean_daily_demand']:.1f} units")
        print(f"  Demand Std Dev: {row['std_daily_demand']:.1f} units")
        print(f"  Critical Ratio: {row['critical_ratio']:.3f}")
        print(f"  Optimal Order Quantity: {row['optimal_quantity']:.0f} units")
        print(f"  Expected Daily Profit: ${row['expected_profit']:.2f}")
        print(f"  Service Level: {row['achieved_service_level']*100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    top_20 = newsvendor_results.nlargest(20, 'expected_profit')
    
    axes[0, 0].barh(range(len(top_20)), top_20['optimal_quantity'].values, color='#2ecc71')
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels([str(p)[:15] for p in top_20['product_id']])
    axes[0, 0].set_xlabel('Optimal Quantity (units)')
    axes[0, 0].set_title('MODEL: Newsvendor Optimal Order Quantity', fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    axes[0, 1].barh(range(len(top_20)), top_20['critical_ratio'].values, color='#9b59b6')
    axes[0, 1].set_yticks(range(len(top_20)))
    axes[0, 1].set_yticklabels([str(p)[:15] for p in top_20['product_id']])
    axes[0, 1].set_xlabel('Critical Ratio')
    axes[0, 1].set_title('MODEL: Critical Ratio (Cu / (Cu + Co))', fontweight='bold')
    axes[0, 1].axvline(x=0.5, color='red', linestyle='--', label='Break-even')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Profit vs Quantity curve for sample product
    sample = top_20.iloc[0]
    Q_range = np.linspace(0, sample['mean_daily_demand'] * 3, 100)
    profits = []
    for Q in Q_range:
        mu, sigma = sample['mean_daily_demand'], sample['std_daily_demand']
        Cu, Co = sample['underage_cost'], sample['overage_cost']
        if sigma > 0:
            z = (Q - mu) / sigma
            exp_sales = mu - sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
            profit = Cu * exp_sales - Co * (Q - exp_sales)
        else:
            profit = Cu * min(Q, mu) - Co * max(0, Q - mu)
        profits.append(profit)
    
    axes[1, 0].plot(Q_range, profits, 'g-', linewidth=2)
    axes[1, 0].axvline(x=sample['optimal_quantity'], color='red', linestyle='--', 
                       label=f"Q* = {sample['optimal_quantity']:.0f}")
    axes[1, 0].axvline(x=sample['mean_daily_demand'], color='blue', linestyle=':', 
                       label=f"Mean = {sample['mean_daily_demand']:.0f}")
    axes[1, 0].set_xlabel('Order Quantity')
    axes[1, 0].set_ylabel('Expected Profit ($)')
    axes[1, 0].set_title(f"MODEL: Newsvendor Profit Curve", fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Service level distribution (robust version)
    # Drop NaN and infinite values to avoid histogram errors
    service_levels = newsvendor_results['achieved_service_level'].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if service_levels.empty:
        # No valid data — just show a message
        axes[1, 1].text(
            0.5, 0.5,
            "No valid service level data",
            ha='center', va='center',
            transform=axes[1, 1].transAxes
        )
    else:
        data_min = service_levels.min()
        data_max = service_levels.max()
        n_unique = service_levels.nunique()

        # If all values are (almost) the same, histogram will break.
        # Use a single bar instead.
        if np.isclose(data_min, data_max):
            axes[1, 1].bar(
                [service_levels.iloc[0]],
                [len(service_levels)],
                width=0.02,
                color='steelblue',
                edgecolor='white'
            )
        else:
            # Choose a safe number of bins based on unique values
            n_bins = min(30, max(1, int(n_unique)))
            axes[1, 1].hist(
                service_levels,
                bins=n_bins,
                color='steelblue',
                edgecolor='white'
            )

        # Mean line
        mean_sl = service_levels.mean()
        axes[1, 1].axvline(
            x=mean_sl,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f"Mean: {mean_sl:.2f}"
        )

        axes[1, 1].set_xlabel('Critical Ratio / Service Level')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('MODEL: Distribution of Optimal Service Levels', fontweight='bold')
        axes[1, 1].legend()

    
    return newsvendor_results


# =============================================================================
# MODEL 3: MIXED-INTEGER PROGRAMMING (MIP)
# Optimal assortment and inventory allocation
# =============================================================================

def model_mip_assortment(df, product_demand, category_demand):
    """
    Mixed-Integer Programming (MIP) for Assortment Optimization
    
    Maximize: Sum(revenue_i * x_i)
    Subject to:
        - Sum(space_i * x_i) <= total_space (shelf space constraint)
        - Sum(cost_i * x_i) <= budget (budget constraint)
        - x_i in {0, 1} (binary: include or exclude product)
    
    This model determines which products to include in assortment
    to maximize revenue subject to space and budget constraints.
    """
    print("\n" + "=" * 100)
    print("MODEL 3: MIXED-INTEGER PROGRAMMING (MIP)")
    print("Objective: Optimize product assortment subject to constraints")
    print("=" * 100)
    
    register_model("Optimization Models", "Mixed-Integer Programming (MIP)",
                   "Binary optimization for product assortment decisions")
    
    # Prepare data for MIP
    mip_data = product_demand.copy()
    
    # Use dataset columns for space and cost requirements
    mip_data['space_required'] = mip_data['avg_stock_quantity'] * 0.1  # Space proportional to stock quantity
    mip_data['inventory_cost'] = mip_data['avg_unit_cost'] * mip_data['avg_stock_quantity']  # Actual inventory cost
    mip_data['contribution_margin'] = mip_data['total_revenue'] * mip_data['avg_profit_margin']  # Use actual profit margin
    
    # Constraints
    total_space = mip_data['space_required'].sum() * 0.6  # Can only use 60% of total space
    total_budget = mip_data['inventory_cost'].sum() * 0.5  # 50% of total inventory budget
    
    # For demonstration, we'll use a greedy heuristic approximation
    # (Full MIP would require pulp or scipy.optimize.milp)
    
    # Calculate efficiency score (avoid division by zero)
    denominator = mip_data['space_required'] + mip_data['inventory_cost'] / 1000
    mip_data['efficiency_score'] = mip_data['contribution_margin'] / denominator.replace(0, 0.01)
    
    # Sort by efficiency and select products
    mip_data_sorted = mip_data.sort_values('efficiency_score', ascending=False)
    
    selected_products = []
    remaining_space = total_space
    remaining_budget = total_budget
    
    for _, row in mip_data_sorted.iterrows():
        if row['space_required'] <= remaining_space and row['inventory_cost'] <= remaining_budget:
            selected_products.append(row['product_id'])
            remaining_space -= row['space_required']
            remaining_budget -= row['inventory_cost']
    
    mip_data['selected'] = mip_data['product_id'].isin(selected_products).astype(int)
    
    # Results
    selected_df = mip_data[mip_data['selected'] == 1]
    excluded_df = mip_data[mip_data['selected'] == 0]
    
    print(f"\n[MIP RESULTS]")
    print(f"  Total Products: {len(mip_data)}")
    print(f"  Selected Products: {len(selected_df)}")
    print(f"  Excluded Products: {len(excluded_df)}")
    print(f"  Space Utilization: {(total_space - remaining_space) / total_space * 100:.1f}%")
    print(f"  Budget Utilization: {(total_budget - remaining_budget) / total_budget * 100:.1f}%")
    print(f"  Total Contribution Margin: ${selected_df['contribution_margin'].sum():,.2f}")
    
    print("\n[MIP PRESCRIPTIONS] Top 10 Selected Products:")
    print("-" * 100)
    for _, row in selected_df.nlargest(10, 'contribution_margin').iterrows():
        print(f"  {row['product_id']}: Margin=${row['contribution_margin']:.0f}, Space={row['space_required']:.1f}")
    
    print("\n[MIP PRESCRIPTIONS] Top 10 Products to Exclude:")
    print("-" * 100)
    for _, row in excluded_df.nsmallest(10, 'efficiency_score').iterrows():
        print(f"  {row['product_id']}: Efficiency Score={row['efficiency_score']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Selected vs Excluded Distribution
    selection_counts = mip_data['selected'].value_counts()
    axes[0, 0].pie([selection_counts.get(1, 0), selection_counts.get(0, 0)], 
                   labels=['Selected', 'Excluded'], autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title('MODEL: MIP Assortment Selection', fontweight='bold')
    
    # Plot 2: Efficiency Score Distribution
    axes[0, 1].hist(mip_data[mip_data['selected']==1]['efficiency_score'], bins=20, 
                    alpha=0.7, label='Selected', color='#2ecc71')
    axes[0, 1].hist(mip_data[mip_data['selected']==0]['efficiency_score'], bins=20, 
                    alpha=0.7, label='Excluded', color='#e74c3c')
    axes[0, 1].set_xlabel('Efficiency Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('MODEL: Efficiency Score by Selection', fontweight='bold')
    axes[0, 1].legend()
    
    # Plot 3: Space vs Margin (bubble chart)
    colors = ['#2ecc71' if s == 1 else '#e74c3c' for s in mip_data['selected']]
    axes[1, 0].scatter(mip_data['space_required'], mip_data['contribution_margin']/1000,
                       c=colors, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Space Required (units)')
    axes[1, 0].set_ylabel('Contribution Margin ($K)')
    axes[1, 0].set_title('MODEL: Space vs Margin Trade-off', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Category breakdown of selection
    cat_selection = mip_data.groupby('category').agg({
        'selected': 'sum',
        'product_id': 'count'
    })
    cat_selection['selection_rate'] = cat_selection['selected'] / cat_selection['product_id'] * 100
    cat_selection_sorted = cat_selection.sort_values('selection_rate')
    
    axes[1, 1].barh(range(len(cat_selection_sorted)), cat_selection_sorted['selection_rate'].values, 
                    color='steelblue')
    axes[1, 1].set_yticks(range(len(cat_selection_sorted)))
    axes[1, 1].set_yticklabels(cat_selection_sorted.index)
    axes[1, 1].set_xlabel('Selection Rate (%)')
    axes[1, 1].set_title('MODEL: MIP Selection Rate by Category', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_mip.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mip_data


# =============================================================================
# MODEL 4: STOCHASTIC OPTIMIZATION
# Multi-scenario inventory optimization under uncertainty
# =============================================================================

def model_stochastic_optimization(df, product_demand):
    """
    Stochastic Optimization (Two-Stage)
    
    Stage 1: Determine order quantity before demand is known
    Stage 2: Observe demand, make recourse decisions
    
    Minimize: E[ordering_cost + holding_cost + shortage_cost]
    
    This model optimizes decisions under multiple demand scenarios,
    accounting for uncertainty in demand forecasts.
    """
    print("\n" + "=" * 100)
    print("MODEL 4: STOCHASTIC OPTIMIZATION")
    print("Objective: Optimize inventory under demand uncertainty (multi-scenario)")
    print("=" * 100)
    
    register_model("Stochastic Models", "Stochastic Optimization (Two-Stage)",
                   "Multi-scenario optimization under demand uncertainty")
    
    # Parameters (using dataset columns)
    n_scenarios = 1000
    ordering_cost = product_demand['avg_unit_cost'].mean() * 0.1  # 10% of average unit cost
    holding_cost_rate = product_demand['avg_profit_margin'].mean() * 0.5  # Based on profit margin
    shortage_cost_rate = holding_cost_rate * 2  # Higher than holding cost
    
    stochastic_results = []
    
    # Analyze top products
    top_products = product_demand.nlargest(30, 'total_revenue')
    
    for _, product in top_products.iterrows():
        mu = product['mean_daily_demand']
        sigma = product['std_daily_demand']
        price = product['avg_price']
        
        if sigma <= 0:
            sigma = mu * 0.3
        
        # Generate demand scenarios
        np.random.seed(42)
        demand_scenarios = np.maximum(0, np.random.normal(mu, sigma, n_scenarios))
        
        # Evaluate different order quantities
        Q_candidates = np.linspace(max(1, mu - 2*sigma), mu + 3*sigma, 50)
        
        best_Q = mu
        best_cost = float('inf')
        
        for Q in Q_candidates:
            scenario_costs = []
            for d in demand_scenarios:
                # Stage 2 recourse
                if d <= Q:
                    # Excess inventory
                    holding = (Q - d) * price * holding_cost_rate
                    shortage = 0
                else:
                    # Shortage
                    holding = 0
                    shortage = (d - Q) * price * shortage_cost_rate
                
                total = ordering_cost + holding + shortage
                scenario_costs.append(total)
            
            avg_cost = np.mean(scenario_costs)
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_Q = Q
        
        # Calculate risk metrics
        final_costs = []
        for d in demand_scenarios:
            if d <= best_Q:
                cost = ordering_cost + (best_Q - d) * price * holding_cost_rate
            else:
                cost = ordering_cost + (d - best_Q) * price * shortage_cost_rate
            final_costs.append(cost)
        
        stochastic_results.append({
            'product_id': product['product_id'],
            'category': product['category'],
            'mean_demand': mu,
            'std_demand': sigma,
            'optimal_order': best_Q,
            'expected_cost': best_cost,
            'cost_std': np.std(final_costs),
            'var_95': np.percentile(final_costs, 95),  # Value at Risk
            'cvar_95': np.mean([c for c in final_costs if c >= np.percentile(final_costs, 95)])  # CVaR
        })
    
    stochastic_df = pd.DataFrame(stochastic_results)
    
    print("\n[STOCHASTIC OPTIMIZATION PRESCRIPTIONS] Top 10 Products:")
    print("-" * 100)
    
    for _, row in stochastic_df.head(10).iterrows():
        print(f"\nProduct: {row['product_id']} ({row['category']})")
        print(f"  Mean Demand: {row['mean_demand']:.1f} units")
        print(f"  Optimal Order (Stochastic): {row['optimal_order']:.0f} units")
        print(f"  Expected Daily Cost: ${row['expected_cost']:.2f}")
        print(f"  Cost Std Dev: ${row['cost_std']:.2f}")
        print(f"  95% VaR: ${row['var_95']:.2f}")
        print(f"  95% CVaR: ${row['cvar_95']:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Optimal Order vs Mean Demand
    axes[0, 0].scatter(stochastic_df['mean_demand'], stochastic_df['optimal_order'], 
                       c='steelblue', s=100, alpha=0.7)
    max_val = max(stochastic_df['mean_demand'].max(), stochastic_df['optimal_order'].max())
    axes[0, 0].plot([0, max_val], [0, max_val], 'r--', label='Order = Demand')
    axes[0, 0].set_xlabel('Mean Demand')
    axes[0, 0].set_ylabel('Optimal Order Quantity')
    axes[0, 0].set_title('MODEL: Stochastic Optimal Order vs Mean Demand', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Risk Metrics (VaR vs CVaR)
    axes[0, 1].scatter(stochastic_df['var_95'], stochastic_df['cvar_95'], 
                       c='#e74c3c', s=100, alpha=0.7)
    axes[0, 1].plot([stochastic_df['var_95'].min(), stochastic_df['var_95'].max()],
                    [stochastic_df['var_95'].min(), stochastic_df['var_95'].max()], 'k--')
    axes[0, 1].set_xlabel('95% Value at Risk ($)')
    axes[0, 1].set_ylabel('95% Conditional VaR ($)')
    axes[0, 1].set_title('MODEL: Risk Analysis (VaR vs CVaR)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cost Distribution for sample product
    sample = top_products.iloc[0]
    mu, sigma = sample['mean_daily_demand'], sample['std_daily_demand']
    if sigma <= 0:
        sigma = mu * 0.3
    demand_scenarios = np.maximum(0, np.random.normal(mu, sigma, n_scenarios))
    optimal_Q = stochastic_df.iloc[0]['optimal_order']
    
    costs = []
    for d in demand_scenarios:
        if d <= optimal_Q:
            cost = ordering_cost + (optimal_Q - d) * sample['avg_price'] * holding_cost_rate
        else:
            cost = ordering_cost + (d - optimal_Q) * sample['avg_price'] * shortage_cost_rate
        costs.append(cost)
    
    axes[1, 0].hist(costs, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    axes[1, 0].axvline(x=np.mean(costs), color='red', linestyle='--', linewidth=2, 
                       label=f'Expected: ${np.mean(costs):.2f}')
    axes[1, 0].axvline(x=np.percentile(costs, 95), color='orange', linestyle='--', linewidth=2,
                       label=f'95% VaR: ${np.percentile(costs, 95):.2f}')
    axes[1, 0].set_xlabel('Total Cost ($)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MODEL: Cost Distribution Under Uncertainty', fontweight='bold')
    axes[1, 0].legend()
    
    # Plot 4: Expected Cost by Category
    cat_costs = stochastic_df.groupby('category')['expected_cost'].mean().sort_values()
    axes[1, 1].barh(range(len(cat_costs)), cat_costs.values, color='#2ecc71')
    axes[1, 1].set_yticks(range(len(cat_costs)))
    axes[1, 1].set_yticklabels(cat_costs.index)
    axes[1, 1].set_xlabel('Expected Daily Cost ($)')
    axes[1, 1].set_title('MODEL: Expected Cost by Category', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_stochastic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stochastic_df


# =============================================================================
# MODEL 5: DYNAMIC PRICING OPTIMIZATION
# Real-time price adjustment based on demand
# =============================================================================

def model_dynamic_pricing(df, product_demand):
    """
    Dynamic Pricing Optimization
    
    Maximize: Revenue = P(p) * D(p)
    Where:
        P(p) = price
        D(p) = demand as function of price (price elasticity)
    
    Optimal Price: p* = c * e / (e - 1)
    Where:
        c = marginal cost
        e = price elasticity of demand
    
    This model determines optimal prices based on demand elasticity
    and can adjust prices dynamically based on inventory levels.
    """
    print("\n" + "=" * 100)
    print("MODEL 5: DYNAMIC PRICING OPTIMIZATION")
    print("Objective: Maximize revenue through optimal pricing")
    print("=" * 100)
    
    register_model("Pricing Models", "Dynamic Pricing Optimization",
                   "Optimize prices based on demand elasticity and inventory levels")
    
    pricing_results = []
    
    # Estimate price elasticity by category
    for category in df['category'].unique():
        cat_data = df[df['category'] == category].copy()
        
        if len(cat_data) < 50:
            continue
        
        # Calculate elasticity from discount response
        # Using markdown as a proxy for price change
        discount_data = cat_data[cat_data['markdown_percentage'] > 0]
        no_discount_data = cat_data[cat_data['markdown_percentage'] == 0]
        
        if len(discount_data) > 10 and len(no_discount_data) > 10:
            # Average quantity at different price points
            q_discount = len(discount_data) / discount_data['current_price'].mean()
            q_full = len(no_discount_data) / no_discount_data['current_price'].mean()
            
            p_discount = discount_data['current_price'].mean()
            p_full = no_discount_data['current_price'].mean()
            
            # Price elasticity of demand
            if p_full != p_discount and q_full > 0:
                elasticity = ((q_discount - q_full) / q_full) / ((p_discount - p_full) / p_full)
                elasticity = abs(elasticity)  # Elasticity is typically positive
            else:
                elasticity = 1.5  # Default assumption
        else:
            elasticity = 1.5
        
        # Ensure reasonable elasticity range
        elasticity = max(0.5, min(elasticity, 5.0))
        
        # Current pricing
        avg_price = cat_data['current_price'].mean()
        avg_original = cat_data['original_price'].mean()
        avg_unit_cost = cat_data['unit_cost'].mean()
        margin = cat_data['profit_margin'].mean()  # Use actual profit margin from data
        marginal_cost = avg_unit_cost  # Use actual unit cost from data
        
        # Optimal price based on elasticity
        if elasticity > 1:
            optimal_price = marginal_cost * elasticity / (elasticity - 1)
        else:
            optimal_price = avg_price * 1.1  # If inelastic, can increase price
        
        # Price change recommendation
        price_change_pct = (optimal_price - avg_price) / avg_price * 100
        
        # Expected revenue impact
        if elasticity != 0:
            quantity_change = -elasticity * price_change_pct / 100
            revenue_change = (1 + price_change_pct/100) * (1 + quantity_change) - 1
        else:
            revenue_change = price_change_pct / 100
        
        pricing_results.append({
            'category': category,
            'current_avg_price': avg_price,
            'estimated_elasticity': elasticity,
            'marginal_cost': marginal_cost,
            'optimal_price': optimal_price,
            'price_change_pct': price_change_pct,
            'expected_revenue_change_pct': revenue_change * 100,
            'current_margin': margin * 100,
            'num_products': len(cat_data['product_id'].unique())
        })
    
    pricing_df = pd.DataFrame(pricing_results)
    
    print("\n[DYNAMIC PRICING PRESCRIPTIONS]:")
    print("-" * 100)
    
    for _, row in pricing_df.iterrows():
        if row['price_change_pct'] > 5:
            action = "INCREASE PRICE"
        elif row['price_change_pct'] < -5:
            action = "DECREASE PRICE"
        else:
            action = "MAINTAIN PRICE"
        
        print(f"\nCategory: {row['category']}")
        print(f"  Current Avg Price: ${row['current_avg_price']:.2f}")
        print(f"  Price Elasticity: {row['estimated_elasticity']:.2f}")
        print(f"  Optimal Price: ${row['optimal_price']:.2f}")
        print(f"  Recommended Change: {row['price_change_pct']:+.1f}%")
        print(f"  Expected Revenue Impact: {row['expected_revenue_change_pct']:+.1f}%")
        print(f"  PRESCRIPTION: {action}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Price Elasticity by Category
    elasticity_sorted = pricing_df.sort_values('estimated_elasticity')
    colors = ['#e74c3c' if e < 1 else '#f39c12' if e < 2 else '#2ecc71' 
              for e in elasticity_sorted['estimated_elasticity']]
    axes[0, 0].barh(range(len(elasticity_sorted)), elasticity_sorted['estimated_elasticity'].values, 
                    color=colors)
    axes[0, 0].axvline(x=1, color='black', linestyle='--', linewidth=2, label='Unit Elasticity')
    axes[0, 0].set_yticks(range(len(elasticity_sorted)))
    axes[0, 0].set_yticklabels(elasticity_sorted['category'])
    axes[0, 0].set_xlabel('Price Elasticity of Demand')
    axes[0, 0].set_title('MODEL: Estimated Price Elasticity by Category', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Current vs Optimal Price
    x_pos = np.arange(len(pricing_df))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, pricing_df['current_avg_price'], width, 
                   label='Current', color='#3498db')
    axes[0, 1].bar(x_pos + width/2, pricing_df['optimal_price'], width, 
                   label='Optimal', color='#2ecc71')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('MODEL: Current vs Optimal Price', fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(pricing_df['category'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Price Change Recommendations
    change_sorted = pricing_df.sort_values('price_change_pct')
    colors = ['#e74c3c' if c < 0 else '#2ecc71' for c in change_sorted['price_change_pct']]
    axes[1, 0].barh(range(len(change_sorted)), change_sorted['price_change_pct'].values, color=colors)
    axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1, 0].set_yticks(range(len(change_sorted)))
    axes[1, 0].set_yticklabels(change_sorted['category'])
    axes[1, 0].set_xlabel('Price Change (%)')
    axes[1, 0].set_title('PRESCRIPTION: Recommended Price Changes', fontweight='bold')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Expected Revenue Impact
    revenue_sorted = pricing_df.sort_values('expected_revenue_change_pct')
    colors = ['#e74c3c' if r < 0 else '#2ecc71' for r in revenue_sorted['expected_revenue_change_pct']]
    axes[1, 1].barh(range(len(revenue_sorted)), revenue_sorted['expected_revenue_change_pct'].values, 
                    color=colors)
    axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=1)
    axes[1, 1].set_yticks(range(len(revenue_sorted)))
    axes[1, 1].set_yticklabels(revenue_sorted['category'])
    axes[1, 1].set_xlabel('Expected Revenue Change (%)')
    axes[1, 1].set_title('MODEL: Expected Revenue Impact', fontweight='bold')
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_dynamic_pricing.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pricing_df


# =============================================================================
# MODEL 6: MARKDOWN OPTIMIZATION
# End-of-season clearance pricing
# =============================================================================

def model_markdown_optimization(df, product_demand):
    """
    Markdown Optimization Model
    
    Objective: Maximize total revenue over markdown periods
    
    Revenue = Sum_t [P_t * D_t(P_t)]
    Subject to:
        - P_t >= P_{t+1} (prices can only decrease)
        - Inventory constraints
        - Minimum price constraints
    
    This model determines optimal markdown schedule to clear
    inventory while maximizing total revenue.
    """
    print("\n" + "=" * 100)
    print("MODEL 6: MARKDOWN OPTIMIZATION")
    print("Objective: Optimize clearance pricing schedule")
    print("=" * 100)
    
    register_model("Pricing Models", "Markdown Optimization",
                   "Optimize end-of-season clearance pricing to maximize revenue")
    
    markdown_results = []
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        
        if len(cat_data) < 30:
            continue
        
        # Current pricing and markdown analysis
        avg_price = cat_data['current_price'].mean()
        avg_original = cat_data['original_price'].mean()
        current_markdown = cat_data['markdown_percentage'].mean()
        
        # Analyze markdown effectiveness
        discount_brackets = [0, 10, 20, 30, 40, 100]
        bracket_performance = []
        
        for i in range(len(discount_brackets) - 1):
            lower, upper = discount_brackets[i], discount_brackets[i+1]
            bracket_data = cat_data[(cat_data['markdown_percentage'] >= lower) & 
                                    (cat_data['markdown_percentage'] < upper)]
            if len(bracket_data) > 0:
                bracket_performance.append({
                    'bracket': f"{lower}-{upper}%",
                    'avg_markdown': bracket_data['markdown_percentage'].mean(),
                    'units_sold': len(bracket_data),
                    'revenue': bracket_data['total_sales'].sum(),
                    'avg_transaction': bracket_data['total_sales'].mean()
                })
        
        bracket_df = pd.DataFrame(bracket_performance)
        
        if len(bracket_df) > 0:
            # Find optimal markdown level
            bracket_df['revenue_per_markdown'] = bracket_df['revenue'] / (bracket_df['avg_markdown'] + 1)
            optimal_bracket = bracket_df.loc[bracket_df['revenue_per_markdown'].idxmax()]
            
            # Markdown schedule recommendation (3-period clearance)
            initial_markdown = max(5, current_markdown * 0.5)
            mid_markdown = optimal_bracket['avg_markdown']
            final_markdown = min(50, mid_markdown * 1.5)
            
            markdown_results.append({
                'category': category,
                'current_avg_markdown': current_markdown,
                'current_avg_price': avg_price,
                'original_price': avg_original,
                'optimal_markdown': optimal_bracket['avg_markdown'],
                'period_1_markdown': initial_markdown,
                'period_2_markdown': mid_markdown,
                'period_3_markdown': final_markdown,
                'expected_clearance_rate': min(95, 70 + final_markdown),
                'num_products': cat_data['product_id'].nunique()
            })
    
    markdown_df = pd.DataFrame(markdown_results)
    
    print("\n[MARKDOWN OPTIMIZATION PRESCRIPTIONS]:")
    print("-" * 100)
    
    for _, row in markdown_df.iterrows():
        print(f"\nCategory: {row['category']}")
        print(f"  Current Avg Markdown: {row['current_avg_markdown']:.1f}%")
        print(f"  Optimal Markdown Level: {row['optimal_markdown']:.1f}%")
        print(f"  PRESCRIPTION - Markdown Schedule:")
        print(f"    Period 1 (Week 1-2): {row['period_1_markdown']:.0f}% off")
        print(f"    Period 2 (Week 3-4): {row['period_2_markdown']:.0f}% off")
        print(f"    Period 3 (Week 5+):  {row['period_3_markdown']:.0f}% off")
        print(f"  Expected Clearance Rate: {row['expected_clearance_rate']:.0f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Current vs Optimal Markdown
    x_pos = np.arange(len(markdown_df))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, markdown_df['current_avg_markdown'], width, 
                   label='Current', color='#3498db')
    axes[0, 0].bar(x_pos + width/2, markdown_df['optimal_markdown'], width, 
                   label='Optimal', color='#2ecc71')
    axes[0, 0].set_ylabel('Markdown (%)')
    axes[0, 0].set_title('MODEL: Current vs Optimal Markdown', fontweight='bold')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(markdown_df['category'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Markdown Schedule
    schedule_data = markdown_df[['category', 'period_1_markdown', 'period_2_markdown', 'period_3_markdown']].melt(
        id_vars='category', var_name='period', value_name='markdown'
    )
    period_labels = {'period_1_markdown': 'Period 1', 'period_2_markdown': 'Period 2', 'period_3_markdown': 'Period 3'}
    schedule_data['period'] = schedule_data['period'].map(period_labels)
    
    for i, cat in enumerate(markdown_df['category'].unique()[:5]):  # Show top 5
        cat_schedule = schedule_data[schedule_data['category'] == cat]
        axes[0, 1].plot(['Period 1', 'Period 2', 'Period 3'], 
                        cat_schedule.sort_values('period')['markdown'].values,
                        marker='o', linewidth=2, label=cat[:10])
    axes[0, 1].set_xlabel('Clearance Period')
    axes[0, 1].set_ylabel('Markdown (%)')
    axes[0, 1].set_title('MODEL: Markdown Schedule by Category', fontweight='bold')
    axes[0, 1].legend(loc='upper left', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Expected Clearance Rate
    clearance_sorted = markdown_df.sort_values('expected_clearance_rate')
    axes[1, 0].barh(range(len(clearance_sorted)), clearance_sorted['expected_clearance_rate'].values, 
                    color='#9b59b6')
    axes[1, 0].set_yticks(range(len(clearance_sorted)))
    axes[1, 0].set_yticklabels(clearance_sorted['category'])
    axes[1, 0].set_xlabel('Expected Clearance Rate (%)')
    axes[1, 0].set_title('MODEL: Expected Clearance by Category', fontweight='bold')
    axes[1, 0].axvline(x=80, color='green', linestyle='--', label='Target: 80%')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Price Decay Curve
    periods = np.array([0, 1, 2, 3, 4, 5, 6])
    sample_cat = markdown_df.iloc[0]
    prices = [100]  # Start at 100%
    for p in [sample_cat['period_1_markdown'], sample_cat['period_1_markdown'],
              sample_cat['period_2_markdown'], sample_cat['period_2_markdown'],
              sample_cat['period_3_markdown'], sample_cat['period_3_markdown']]:
        prices.append(100 - p)
    
    axes[1, 1].step(periods, prices, where='post', linewidth=3, color='#e74c3c')
    axes[1, 1].fill_between(periods, prices, 0, step='post', alpha=0.3, color='#e74c3c')
    axes[1, 1].set_xlabel('Week')
    axes[1, 1].set_ylabel('Price (% of Original)')
    axes[1, 1].set_title(f"MODEL: Markdown Decay Curve ({sample_cat['category']})", fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('model_markdown.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return markdown_df


# =============================================================================
# SUMMARY: LIST ALL PRESCRIPTIVE MODELS
# =============================================================================

def list_prescriptive_models():
    """Print comprehensive list of all prescriptive models used"""
    print("\n" + "=" * 100)
    print(" " * 25 + "PRESCRIPTIVE ANALYTICS - MODELS SUMMARY")
    print("=" * 100)
    
    all_models = []
    
    for category, models in PRESCRIPTIVE_MODELS.items():
        if models:
            print(f"\n{'-' * 100}")
            print(f"{category.upper()}:")
            print("-" * 100)
            for model in models:
                print(f"  * {model['name']}")
                print(f"    Description: {model['description']}")
                all_models.append(model['name'])
    
    print(f"\n{'=' * 100}")
    print(f"TOTAL PRESCRIPTIVE MODELS: {len(all_models)}")
    print("=" * 100)
    
    # Models list
    print("\n[MODELS LIST]")
    print("1. Economic Order Quantity (EOQ) - Classical inventory optimization")
    print("2. Newsvendor Model - Single-period stochastic inventory")
    print("3. Mixed-Integer Programming (MIP) - Binary assortment optimization")
    print("4. Stochastic Optimization - Multi-scenario uncertainty optimization")
    print("5. Dynamic Pricing Optimization - Price elasticity-based pricing")
    print("6. Markdown Optimization - End-of-season clearance pricing")
    
    return PRESCRIPTIVE_MODELS


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print(" " * 30 + "PRESCRIPTIVE ANALYTICS PIPELINE")
    print(" " * 25 + "Optimization Models for Business Decisions")
    print("=" * 100)
    
    # Load data
    df = load_data()
    
    # Compute demand statistics
    product_demand, category_demand = compute_demand_statistics(df)
    
    # Run all prescriptive models
    print("\n" + "=" * 100)
    print("RUNNING PRESCRIPTIVE OPTIMIZATION MODELS")
    print("=" * 100)
    
    # Model 1: EOQ
    eoq_results = model_eoq(df, product_demand)
    
    # Model 2: Newsvendor
    newsvendor_results = model_newsvendor(df, product_demand)
    
    # Model 3: MIP Assortment
    mip_results = model_mip_assortment(df, product_demand, category_demand)
    
    # Model 4: Stochastic Optimization
    stochastic_results = model_stochastic_optimization(df, product_demand)
    
    # Model 5: Dynamic Pricing
    pricing_results = model_dynamic_pricing(df, product_demand)
    
    # Model 6: Markdown Optimization
    markdown_results = model_markdown_optimization(df, product_demand)
    
    # Summary
    models_summary = list_prescriptive_models()
    
    # Export results
    print("\n" + "=" * 100)
    print("EXPORTING RESULTS")
    print("=" * 100)
    
    output_dir = "Prescriptive_Analytics/Model_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    eoq_results.to_csv(f"{output_dir}/model_1_eoq_results.csv", index=False)
    newsvendor_results.to_csv(f"{output_dir}/model_2_newsvendor_results.csv", index=False)
    mip_results.to_csv(f"{output_dir}/model_3_mip_results.csv", index=False)
    stochastic_results.to_csv(f"{output_dir}/model_4_stochastic_results.csv", index=False)
    pricing_results.to_csv(f"{output_dir}/model_5_dynamic_pricing_results.csv", index=False)
    markdown_results.to_csv(f"{output_dir}/model_6_markdown_results.csv", index=False)
    
    print(f"\n[OK] All model results exported to {output_dir}/")
    
    print("\n" + "=" * 100)
    print(" " * 30 + "PRESCRIPTIVE ANALYTICS COMPLETE")
    print("=" * 100)
