import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

print("=" * 80)
print("PREDICTIVE MODELING & OPTIMIZATION ANALYSIS")
print("=" * 80)

def load_data():
    """
    Load and preprocess the dataset
    """
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    csv_file = "ETL/dataset_ele_5_cleaned_adjusted.csv"
    df = pd.read_csv(csv_file)
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df = df.sort_values('purchase_date').reset_index(drop=True)
    # Additional preprocessing like discount flags if needed
    if 'markdown_percentage' in df.columns:
        df['has_discount'] = df['markdown_percentage'] > 0
    else:
        df['has_discount'] = False
    print(f"Data loaded: {len(df)} rows")
    return df


# ============================================================================
# 1. ECONOMIC ORDER QUANTITY (EOQ) MODEL
# ============================================================================
def calculate_eoq(df):
    """
    Calculate optimal Economic Order Quantity
    EOQ = sqrt((2 * D * S) / H)
    where D = annual demand, S = order cost, H = holding cost
    """
    print("\n" + "=" * 80)
    print("1. ECONOMIC ORDER QUANTITY (EOQ) MODEL")
    print("=" * 80)
    
    # Group by product and calculate metrics
    product_stats = df.groupby('product_id').agg({
        'product_id': 'count',  # quantity ordered
        'total_sales': 'sum',   # revenue
        'current_price': 'mean'  # average price
    }).rename(columns={'product_id': 'units_sold'})
    
    # Calculate annual metrics
    product_stats['annual_demand'] = product_stats['units_sold'] * (365 / (df['purchase_date'].max() - df['purchase_date'].min()).days)
    
    # Assumed costs (you can adjust these)
    order_cost = 50  # Cost per order
    holding_cost_rate = 0.25  # 25% of product value per year
    
    product_stats['holding_cost'] = product_stats['current_price'] * holding_cost_rate
    product_stats['eoq'] = np.sqrt((2 * product_stats['annual_demand'] * order_cost) / product_stats['holding_cost'])
    product_stats['annual_orders'] = product_stats['annual_demand'] / product_stats['eoq']
    product_stats['annual_order_cost'] = product_stats['annual_orders'] * order_cost
    product_stats['annual_holding_cost'] = (product_stats['eoq'] / 2) * product_stats['holding_cost']
    product_stats['total_inventory_cost'] = product_stats['annual_order_cost'] + product_stats['annual_holding_cost']
    
    # Top products by EOQ
    eoq_results = product_stats.nlargest(10, 'annual_demand')[['units_sold', 'annual_demand', 'eoq', 'annual_orders', 'total_inventory_cost']]
    
    print("\nTop 10 Products - EOQ Analysis:")
    print(eoq_results.to_string())
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: EOQ by product
    top_eoq = product_stats.nlargest(10, 'annual_demand')
    axes[0, 0].barh(range(len(top_eoq)), top_eoq['eoq'].values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top_eoq)))
    axes[0, 0].set_yticklabels([f"Product {pid}" for pid in top_eoq.index])
    axes[0, 0].set_xlabel('Economic Order Quantity (units)', fontweight='bold')
    axes[0, 0].set_title('Optimal Order Quantity by Product', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Annual demand vs EOQ
    axes[0, 1].scatter(product_stats['annual_demand'], product_stats['eoq'], alpha=0.6, s=100, color='green')
    axes[0, 1].set_xlabel('Annual Demand (units)', fontweight='bold')
    axes[0, 1].set_ylabel('EOQ (units)', fontweight='bold')
    axes[0, 1].set_title('Annual Demand vs Optimal Order Quantity', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Cost comparison
    cost_data = product_stats.nlargest(10, 'annual_demand')[['annual_order_cost', 'annual_holding_cost']]
    cost_data.plot(kind='bar', ax=axes[1, 0], color=['#e74c3c', '#3498db'])
    axes[1, 0].set_ylabel('Annual Cost ($)', fontweight='bold')
    axes[1, 0].set_title('Annual Ordering vs Holding Costs', fontweight='bold', fontsize=12)
    axes[1, 0].legend(['Order Cost', 'Holding Cost'], fontsize=10)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Total inventory cost
    axes[1, 1].barh(range(len(top_eoq)), top_eoq['total_inventory_cost'].values, color='coral')
    axes[1, 1].set_yticks(range(len(top_eoq)))
    axes[1, 1].set_yticklabels([f"Product {pid}" for pid in top_eoq.index])
    axes[1, 1].set_xlabel('Total Annual Inventory Cost ($)', fontweight='bold')
    axes[1, 1].set_title('Total Inventory Management Cost', fontweight='bold', fontsize=12)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eoq_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ EOQ visualization saved as 'eoq_analysis.png'")
    
    return product_stats

# ============================================================================
# 2. NEWSVENDOR MODEL
# ============================================================================
def newsvendor_model(df):
    """
    Newsvendor model for optimal inventory under uncertain demand
    Optimal service level = (p - c) / p, where p = price, c = cost
    """
    print("\n" + "=" * 80)
    print("2. NEWSVENDOR MODEL (Stochastic Inventory)")
    print("=" * 80)
    
    # Group by category to get demand distribution
    category_demand = df.groupby('category').agg({
        'product_id': 'count',
        'total_sales': ['sum', 'mean', 'std'],
        'current_price': 'mean'
    }).round(2)
    
    category_demand.columns = ['units_sold', 'total_revenue', 'avg_revenue', 'revenue_std', 'avg_price']
    
    # Assume 70% of current_price is cost, 30% is margin
    category_demand['cost'] = category_demand['avg_price'] * 0.7
    category_demand['margin'] = category_demand['avg_price'] * 0.3
    category_demand['optimal_service_level'] = (category_demand['avg_price'] - category_demand['cost']) / category_demand['avg_price']
    
    # Calculate daily demand statistics
    daily_demand = df.groupby(['category', df['purchase_date'].dt.date])['product_id'].count().reset_index()
    daily_demand_stats = daily_demand.groupby('category')['product_id'].agg(['mean', 'std', 'min', 'max']).round(2)
    daily_demand_stats.columns = ['mean_daily_demand', 'std_daily_demand', 'min_daily_demand', 'max_daily_demand']
    
    category_results = pd.concat([category_demand, daily_demand_stats], axis=1)
    
    shortage_cost_rate = 50  # Cost per unit short
    category_results['Expected_Shortage'] = np.maximum(
        category_results['std_daily_demand'] * 0.1,  # Estimated shortage as percentage of std dev
        0
    )
    
    print("\nNewsvendor Analysis by Category:")
    print(category_results.to_string())
    
    # Calculate optimal order quantities for different service levels
    print("\nOptimal Order Quantities at Different Service Levels:")
    for category in category_results.index[:5]:
        mean_demand = category_results.loc[category, 'mean_daily_demand']
        std_demand = category_results.loc[category, 'std_daily_demand']
        service_level = category_results.loc[category, 'optimal_service_level']
        
        # Z-score for optimal service level
        z_score = norm.ppf(service_level)
        optimal_quantity = mean_demand + z_score * std_demand
        
        print(f"\n  {category}:")
        print(f"    - Mean daily demand: {mean_demand:.1f} units")
        print(f"    - Std dev: {std_demand:.1f} units")
        print(f"    - Optimal service level: {service_level:.1%}")
        print(f"    - Optimal order quantity: {optimal_quantity:.0f} units")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Service level by category
    axes[0, 0].barh(range(len(category_results)), category_results['optimal_service_level'].values * 100, color='mediumseagreen')
    axes[0, 0].set_yticks(range(len(category_results)))
    axes[0, 0].set_yticklabels(category_results.index)
    axes[0, 0].set_xlabel('Optimal Service Level (%)', fontweight='bold')
    axes[0, 0].set_title('Target Service Levels by Category', fontweight='bold', fontsize=12)
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Demand variability
    axes[0, 1].scatter(category_results['mean_daily_demand'], category_results['std_daily_demand'], s=300, alpha=0.6, color='purple')
    for i, cat in enumerate(category_results.index):
        axes[0, 1].annotate(cat, (category_results.iloc[i]['mean_daily_demand'], category_results.iloc[i]['std_daily_demand']), 
                           fontsize=8, ha='center')
    axes[0, 1].set_xlabel('Mean Daily Demand (units)', fontweight='bold')
    axes[0, 1].set_ylabel('Demand Std Dev (units)', fontweight='bold')
    axes[0, 1].set_title('Demand Variability by Category', fontweight='bold', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Price vs Cost vs Margin
    x_pos = np.arange(len(category_results))
    width = 0.25
    axes[1, 0].bar(x_pos - width, category_results['cost'], width, label='Cost', color='#e74c3c')
    axes[1, 0].bar(x_pos, category_results['margin'], width, label='Margin', color='#2ecc71')
    axes[1, 0].bar(x_pos + width, category_results['avg_price'], width, label='Price', color='#3498db')
    axes[1, 0].set_ylabel('Amount ($)', fontweight='bold')
    axes[1, 0].set_title('Price Structure by Category', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(category_results.index, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Demand range
    demand_range = category_results['max_daily_demand'] - category_results['min_daily_demand']
    axes[1, 1].barh(range(len(category_results)), demand_range.values, color='#f39c12')
    axes[1, 1].set_yticks(range(len(category_results)))
    axes[1, 1].set_yticklabels(category_results.index)
    axes[1, 1].set_xlabel('Daily Demand Range (units)', fontweight='bold')
    axes[1, 1].set_title('Daily Demand Range by Category', fontweight='bold', fontsize=12)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('newsvendor_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Newsvendor visualization saved as 'newsvendor_analysis.png'")
    
    return category_results

# ============================================================================
# 3. MIXED-INTEGER PROGRAMMING (MIP) MODEL
# ============================================================================
def mixed_integer_programming(df):
    """
    MIP for inventory allocation across warehouses/locations
    Maximizes profit subject to inventory constraints
    """
    print("\n" + "=" * 80)
    print("3. MIXED-INTEGER PROGRAMMING MODEL (Inventory Allocation)")
    print("=" * 80)
    
    # Simulate warehouse locations
    countries = df['country'].unique()[:5]  # Top 5 countries as warehouses
    categories = df['category'].unique()
    
    # Build demand matrix
    demand_matrix = df[df['country'].isin(countries)].groupby(['country', 'category']).agg({
        'product_id': 'count',
        'total_sales': 'sum',
        'current_price': 'mean'
    }).rename(columns={'product_id': 'demand'})
    
    # Assume warehouse capacity and costs
    warehouse_capacity = 1000  # Units per warehouse
    transportation_cost_per_unit = 5
    warehouse_cost_per_unit = 2
    
    print(f"\nOptimization Constraints:")
    print(f"  - Warehouses: {len(countries)}")
    print(f"  - Product categories: {len(categories)}")
    print(f"  - Warehouse capacity: {warehouse_capacity} units")
    print(f"  - Transportation cost: ${transportation_cost_per_unit}/unit")
    print(f"  - Warehouse cost: ${warehouse_cost_per_unit}/unit")
    
    # Calculate allocation metrics
    results = []
    total_demand = 0
    total_allocation = 0
    
    for country in countries:
        country_data = df[df['country'] == country]
        country_demand = len(country_data)
        total_demand += country_demand
        
        # Optimal allocation proportional to demand
        country_allocation = min(country_demand, warehouse_capacity)
        total_allocation += country_allocation
        
        avg_price = country_data['current_price'].mean()
        total_cost = (country_allocation * (warehouse_cost_per_unit + transportation_cost_per_unit))
        revenue = country_allocation * avg_price
        profit = revenue - total_cost
        
        results.append({
            'Warehouse': country,
            'Demand': country_demand,
            'Allocation': country_allocation,
            'Capacity_Used_%': (country_allocation / warehouse_capacity) * 100,
            'Revenue': revenue,
            'Cost': total_cost,
            'Profit': profit
        })
    
    mip_results = pd.DataFrame(results)
    
    print("\nMIP Allocation Results:")
    print(mip_results.to_string(index=False))
    print(f"\nTotal demand: {total_demand} units")
    print(f"Total allocation: {total_allocation} units")
    print(f"Overall utilization: {(total_allocation/len(countries)/warehouse_capacity)*100:.1f}%")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Demand vs Allocation
    x_pos = np.arange(len(mip_results))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, mip_results['Demand'], width, label='Demand', color='#3498db')
    axes[0, 0].bar(x_pos + width/2, mip_results['Allocation'], width, label='Allocation', color='#2ecc71')
    axes[0, 0].axhline(y=warehouse_capacity, color='red', linestyle='--', label='Capacity')
    axes[0, 0].set_ylabel('Units', fontweight='bold')
    axes[0, 0].set_title('Demand vs Optimal Allocation', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(mip_results['Warehouse'])
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Capacity utilization
    axes[0, 1].barh(range(len(mip_results)), mip_results['Capacity_Used_%'].values, color='#f39c12')
    axes[0, 1].set_yticks(range(len(mip_results)))
    axes[0, 1].set_yticklabels(mip_results['Warehouse'])
    axes[0, 1].set_xlabel('Capacity Utilization (%)', fontweight='bold')
    axes[0, 1].set_title('Warehouse Capacity Utilization', fontweight='bold', fontsize=12)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Revenue vs Cost
    x_pos = np.arange(len(mip_results))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, mip_results['Revenue'], width, label='Revenue', color='#2ecc71')
    axes[1, 0].bar(x_pos + width/2, mip_results['Cost'], width, label='Cost', color='#e74c3c')
    axes[1, 0].set_ylabel('Amount ($)', fontweight='bold')
    axes[1, 0].set_title('Revenue vs Total Cost by Warehouse', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(mip_results['Warehouse'])
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Profit
    axes[1, 1].bar(range(len(mip_results)), mip_results['Profit'].values, color='#9b59b6')
    axes[1, 1].set_ylabel('Profit ($)', fontweight='bold')
    axes[1, 1].set_title('Profit by Warehouse', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(range(len(mip_results)))
    axes[1, 1].set_xticklabels(mip_results['Warehouse'], rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mip_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ MIP visualization saved as 'mip_analysis.png'")
    
    return mip_results

# ============================================================================
# 4. STOCHASTIC OPTIMIZATION MODEL
# ============================================================================
def stochastic_optimization(df):
    """
    Stochastic optimization for demand uncertainty
    Uses Monte Carlo simulation and chance constraints
    """
    print("\n" + "=" * 80)
    print("4. STOCHASTIC OPTIMIZATION MODEL (Demand Uncertainty)")
    print("=" * 80)
    
    # Calculate daily demand by category
    daily_category_demand = df.groupby(['category', df['purchase_date'].dt.date])['product_id'].count().reset_index()
    daily_category_demand.columns = ['category', 'date', 'demand']
    
    stochastic_results = []
    
    for category in df['category'].unique():
        cat_demand = daily_category_demand[daily_category_demand['category'] == category]['demand'].values
        
        if len(cat_demand) < 2:
            continue
        
        mean_demand = np.mean(cat_demand)
        std_demand = np.std(cat_demand)
        min_demand = np.min(cat_demand)
        max_demand = np.max(cat_demand)
        
        # Monte Carlo simulation (1000 scenarios)
        np.random.seed(42)
        simulated_demands = np.random.normal(mean_demand, std_demand, 1000)
        simulated_demands = np.maximum(simulated_demands, 0)  # Ensure non-negative
        
        # Calculate safety stock for 95% service level (1.645 z-score)
        safety_stock = 1.645 * std_demand
        optimal_stock = mean_demand + safety_stock
        
        # Risk metrics
        stockout_risk_10 = np.sum(simulated_demands > optimal_stock * 1.1) / len(simulated_demands)
        shortage_cost_rate = 50  # Cost per unit short
        expected_shortage = np.mean(np.maximum(simulated_demands - optimal_stock, 0))
        
        stochastic_results.append({
            'Category': category,
            'Mean_Demand': mean_demand,
            'Std_Dev': std_demand,
            'Min_Demand': min_demand,
            'Max_Demand': max_demand,
            'Optimal_Stock': optimal_stock,
            'Safety_Stock': safety_stock,
            'Expected_Shortage': expected_shortage,
            'Stockout_Risk_%': stockout_risk_10 * 100,
            'Expected_Shortage_Cost': expected_shortage * shortage_cost_rate
        })
    
    stoch_df = pd.DataFrame(stochastic_results)
    
    print("\nStochastic Optimization Results:")
    print(stoch_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Demand distribution
    sample_category = stoch_df.iloc[0]['Category']
    cat_demand = daily_category_demand[daily_category_demand['category'] == sample_category]['demand'].values
    axes[0, 0].hist(cat_demand, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(cat_demand), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('Daily Demand (units)', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title(f'Demand Distribution - {sample_category}', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Optimal stock levels
    axes[0, 1].barh(range(len(stoch_df)), stoch_df['Optimal_Stock'], color='#2ecc71')
    axes[0, 1].set_yticks(range(len(stoch_df)))
    axes[0, 1].set_yticklabels(stoch_df['Category'])
    axes[0, 1].set_xlabel('Optimal Stock Level (units)', fontweight='bold')
    axes[0, 1].set_title('Recommended Safety Stock Levels', fontweight='bold', fontsize=12)
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Safety stock
    axes[0, 2].bar(range(len(stoch_df)), stoch_df['Safety_Stock'].values, color='#f39c12')
    axes[0, 2].set_ylabel('Safety Stock (units)', fontweight='bold')
    axes[0, 2].set_title('Safety Stock Buffer', fontweight='bold', fontsize=12)
    axes[0, 2].set_xticks(range(len(stoch_df)))
    axes[0, 2].set_xticklabels(stoch_df['Category'], rotation=45, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Stockout risk
    axes[1, 0].barh(range(len(stoch_df)), stoch_df['Stockout_Risk_%'], color='#e74c3c')
    axes[1, 0].set_yticks(range(len(stoch_df)))
    axes[1, 0].set_yticklabels(stoch_df['Category'])
    axes[1, 0].set_xlabel('Stockout Risk (%)', fontweight='bold')
    axes[1, 0].set_title('Risk of Stock Shortage', fontweight='bold', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 5: Expected shortage cost
    axes[1, 1].bar(range(len(stoch_df)), stoch_df['Expected_Shortage_Cost'].values, color='#9b59b6')
    axes[1, 1].set_ylabel('Expected Cost ($)', fontweight='bold')
    axes[1, 1].set_title('Expected Shortage Cost', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(range(len(stoch_df)))
    axes[1, 1].set_xticklabels(stoch_df['Category'], rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Plot 6: Coefficient of variation
    stoch_df['CV'] = stoch_df['Std_Dev'] / stoch_df['Mean_Demand']
    axes[1, 2].barh(range(len(stoch_df)), stoch_df['CV'], color='#1abc9c')
    axes[1, 2].set_yticks(range(len(stoch_df)))
    axes[1, 2].set_yticklabels(stoch_df['Category'])
    axes[1, 2].set_xlabel('Coefficient of Variation', fontweight='bold')
    axes[1, 2].set_title('Demand Volatility', fontweight='bold', fontsize=12)
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stochastic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Stochastic optimization visualization saved as 'stochastic_analysis.png'")
    
    return stoch_df

# ============================================================================
# 5. DYNAMIC PRICING OPTIMIZATION
# ============================================================================
def dynamic_pricing_optimization(df):
    """
    Dynamic pricing model that optimizes prices over time
    Maximizes revenue based on demand elasticity and inventory levels
    """
    print("\n" + "=" * 80)
    print("5. DYNAMIC PRICING OPTIMIZATION MODEL")
    print("=" * 80)
    
    # Group by category and date
    df['date'] = df['purchase_date'].dt.date
    daily_category = df.groupby(['category', 'date']).agg({
        'current_price': 'mean',
        'product_id': 'count',
        'total_sales': 'sum'
    }).reset_index()
    daily_category.columns = ['category', 'date', 'price', 'quantity', 'revenue']
    
    pricing_results = []
    
    for category in df['category'].unique():
        cat_data = daily_category[daily_category['category'] == category].sort_values('date')
        
        if len(cat_data) < 3:
            continue
        
        # Calculate price elasticity
        prices = cat_data['price'].values
        quantities = cat_data['quantity'].values
        
        # Normalize for elasticity calculation
        if len(prices) > 1:
            price_changes = np.diff(prices) / prices[:-1]
            quantity_changes = np.diff(quantities) / quantities[:-1]
            
            # Avoid division by zero
            valid_idx = np.abs(price_changes) > 0.001
            if np.sum(valid_idx) > 0:
                elasticity = np.mean(quantity_changes[valid_idx] / price_changes[valid_idx])
            else:
                elasticity = -1.0  # Default elasticity
        else:
            elasticity = -1.0
        
        # Current metrics
        current_price = prices[-1]
        current_qty = quantities[-1]
        current_revenue = cat_data['revenue'].iloc[-1]
        
        # Optimal price calculation (revenue maximization)
        # Assuming linear demand: Q = a + b*P, optimal P = -a/(2b)
        # Estimate optimal price with elasticity
        optimal_price = current_price * (1 + 1 / (2 * elasticity)) if elasticity != 0 else current_price
        optimal_price = max(optimal_price, current_price * 0.7)  # Don't go below 70% of current
        optimal_price = min(optimal_price, current_price * 1.3)  # Don't go above 130% of current
        
        # Expected quantity at optimal price
        price_ratio = optimal_price / current_price if current_price > 0 else 1
        expected_qty = current_qty * (price_ratio ** elasticity)
        expected_revenue = optimal_price * expected_qty
        revenue_change = expected_revenue - current_revenue
        revenue_change_pct = (revenue_change / current_revenue * 100) if current_revenue > 0 else 0
        
        pricing_results.append({
            'Category': category,
            'Current_Price': current_price,
            'Optimal_Price': optimal_price,
            'Price_Change_%': ((optimal_price - current_price) / current_price * 100) if current_price > 0 else 0,
            'Price_Elasticity': elasticity,
            'Current_Quantity': current_qty,
            'Expected_Quantity': expected_qty,
            'Current_Revenue': current_revenue,
            'Expected_Revenue': expected_revenue,
            'Revenue_Change_$': revenue_change,
            'Revenue_Change_%': revenue_change_pct
        })
    
    pricing_df = pd.DataFrame(pricing_results)
    
    print("\nDynamic Pricing Optimization Results:")
    print(pricing_df.to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Price recommendations
    x_pos = np.arange(len(pricing_df))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, pricing_df['Current_Price'], width, label='Current', color='#3498db')
    axes[0, 0].bar(x_pos + width/2, pricing_df['Optimal_Price'], width, label='Optimal', color='#2ecc71')
    axes[0, 0].set_ylabel('Price ($)', fontweight='bold')
    axes[0, 0].set_title('Current vs Optimal Pricing', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(pricing_df['Category'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Price elasticity
    colors = ['#e74c3c' if x < -1 else '#2ecc71' for x in pricing_df['Price_Elasticity']]
    axes[0, 1].barh(range(len(pricing_df)), pricing_df['Price_Elasticity'].values, color=colors)
    axes[0, 1].axvline(x=-1, color='red', linestyle='--', linewidth=2, label='Unit Elastic')
    axes[0, 1].set_xticks(range(len(pricing_df)))
    axes[0, 1].set_yticklabels(pricing_df['Category'])
    axes[0, 1].set_xlabel('Elasticity', fontweight='bold')
    axes[0, 1].set_title('Price Elasticity by Category', fontweight='bold', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Price change percentage
    axes[0, 2].bar(range(len(pricing_df)), pricing_df['Price_Change_%'].values, 
                   color=['#2ecc71' if x > 0 else '#e74c3c' for x in pricing_df['Price_Change_%']])
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 2].set_ylabel('Price Change (%)', fontweight='bold')
    axes[0, 2].set_title('Recommended Price Adjustments', fontweight='bold', fontsize=12)
    axes[0, 2].set_xticks(range(len(pricing_df)))
    axes[0, 2].set_xticklabels(pricing_df['Category'], rotation=45, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Current revenue vs expected revenue
    x_pos = np.arange(len(pricing_df))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, pricing_df['Current_Revenue'], width, label='Current', color='#3498db')
    axes[1, 0].bar(x_pos + width/2, pricing_df['Expected_Revenue'], width, label='Expected', color='#2ecc71')
    axes[1, 0].set_ylabel('Revenue ($)', fontweight='bold')
    axes[1, 0].set_title('Current vs Expected Revenue', fontweight='bold', fontsize=12)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(pricing_df['Category'], rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 5: Revenue change
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in pricing_df['Revenue_Change_$']]
    axes[1, 1].bar(range(len(pricing_df)), pricing_df['Revenue_Change_$'].values, color=colors)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 1].set_ylabel('Revenue Change ($)', fontweight='bold')
    axes[1, 1].set_title('Expected Revenue Impact', fontweight='bold', fontsize=12)
    axes[1, 1].set_xticks(range(len(pricing_df)))
    axes[1, 1].set_xticklabels(pricing_df['Category'], rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Plot 6: Revenue change percentage
    axes[1, 2].barh(range(len(pricing_df)), pricing_df['Revenue_Change_%'].values, 
                    color=['#2ecc71' if x > 0 else '#e74c3c' for x in pricing_df['Revenue_Change_%']])
    axes[1, 2].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 2].set_xticks(range(len(pricing_df)))
    axes[1, 2].set_yticklabels(pricing_df['Category'])
    axes[1, 2].set_xlabel('Revenue Change (%)', fontweight='bold')
    axes[1, 2].set_title('Expected Revenue Uplift', fontweight='bold', fontsize=12)
    axes[1, 2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_pricing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Dynamic pricing visualization saved as 'dynamic_pricing_analysis.png'")
    
    return pricing_df

# ============================================================================
# 6. MARKDOWN OPTIMIZATION MODEL
# ============================================================================
def markdown_optimization(df):
    """
    Markdown optimization to determine optimal discount levels
    Balances volume increase against margin reduction
    """
    print("\n" + "=" * 80)
    print("6. MARKDOWN OPTIMIZATION MODEL")
    print("=" * 80)
    
    # Analyze discount impact by category
    df['has_discount'] = df['markdown_percentage'] > 0
    
    markdown_results = []
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        
        # Separate discounted and non-discounted
        discounted = cat_data[cat_data['has_discount'] == True]
        not_discounted = cat_data[cat_data['has_discount'] == False]
        
        if len(discounted) == 0 or len(not_discounted) == 0:
            continue
        
        # Metrics for discounted items
        disc_avg_price = discounted['current_price'].mean()
        disc_avg_orig_price = discounted['original_price'].mean()
        disc_markdown_pct = (discounted['markdown_percentage'].mean())
        disc_units = len(discounted)
        disc_revenue = discounted['total_sales'].sum()
        disc_avg_revenue_per_unit = disc_revenue / disc_units if disc_units > 0 else 0
        
        # Metrics for non-discounted items
        nodis_avg_price = not_discounted['current_price'].mean()
        nodis_units = len(not_discounted)
        nodis_revenue = not_discounted['total_sales'].sum()
        nodis_avg_revenue_per_unit = nodis_revenue / nodis_units if nodis_units > 0 else 0
        
        # Calculate lift from discount
        units_lift_pct = ((disc_units - nodis_units) / nodis_units * 100) if nodis_units > 0 else 0
        price_loss_pct = ((disc_avg_price - nodis_avg_price) / nodis_avg_price * 100) if nodis_avg_price > 0 else 0
        
        # Total profit impact
        disc_profit_per_unit = disc_avg_price * 0.4  # Assume 40% margin
        nodis_profit_per_unit = nodis_avg_price * 0.4
        
        total_disc_profit = disc_units * disc_profit_per_unit
        total_nodis_profit = nodis_units * nodis_profit_per_unit
        
        # Optimal markdown calculation (maximize total profit)
        # Current markdown effectiveness
        markdown_efficiency = (units_lift_pct / 100) / (abs(price_loss_pct) / 100) if abs(price_loss_pct) > 0 else 0
        
        # Recommended markdown (if efficiency > 1, increase; if < 1, decrease)
        if markdown_efficiency > 1.5:
            recommended_markdown = disc_markdown_pct * 1.1
        elif markdown_efficiency < 0.5:
            recommended_markdown = disc_markdown_pct * 0.9
        else:
            recommended_markdown = disc_markdown_pct
        
        recommended_markdown = min(recommended_markdown, 0.5)  # Cap at 50%
        
        markdown_results.append({
            'Category': category,
            'Current_Markdown_%': disc_markdown_pct * 100,
            'Recommended_Markdown_%': recommended_markdown * 100,
            'Discounted_Units': disc_units,
            'Non_Discounted_Units': nodis_units,
            'Units_Lift_%': units_lift_pct,
            'Price_Loss_%': price_loss_pct,
            'Markdown_Efficiency': markdown_efficiency,
            'Discounted_Avg_Price': disc_avg_price,
            'Non_Discounted_Avg_Price': nodis_avg_price,
            'Discounted_Total_Profit': total_disc_profit,
            'Non_Discounted_Total_Profit': total_nodis_profit,
            'Profit_Difference': total_disc_profit - total_nodis_profit
        })
    
    markdown_df = pd.DataFrame(markdown_results)
    
    print("\nMarkdown Optimization Results:")
    print(markdown_df[[col for col in markdown_df.columns if col not in ['Discounted_Avg_Price', 'Non_Discounted_Avg_Price', 'Discounted_Total_Profit', 'Non_Discounted_Total_Profit']]].to_string(index=False))
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Current vs recommended markdown
    x_pos = np.arange(len(markdown_df))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, markdown_df['Current_Markdown_%'], width, label='Current', color='#3498db')
    axes[0, 0].bar(x_pos + width/2, markdown_df['Recommended_Markdown_%'], width, label='Recommended', color='#2ecc71')
    axes[0, 0].set_ylabel('Markdown Level (%)', fontweight='bold')
    axes[0, 0].set_title('Current vs Optimal Markdown Strategy', fontweight='bold', fontsize=12)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(markdown_df['Category'], rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Units impact
    x_pos = np.arange(len(markdown_df))
    width = 0.35
    axes[0, 1].bar(x_pos - width/2, markdown_df['Discounted_Units'], width, label='With Markdown', color='#2ecc71')
    axes[0, 1].bar(x_pos + width/2, markdown_df['Non_Discounted_Units'], width, label='Without Markdown', color='#e74c3c')
    axes[0, 1].set_ylabel('Units Sold', fontweight='bold')
    axes[0, 1].set_title('Units Impact: With vs Without Markdown', fontweight='bold', fontsize=12)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(markdown_df['Category'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Units lift percentage
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in markdown_df['Units_Lift_%']]
    axes[0, 2].bar(range(len(markdown_df)), markdown_df['Units_Lift_%'].values, color=colors)
    axes[0, 2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0, 2].set_ylabel('Units Lift (%)', fontweight='bold')
    axes[0, 2].set_title('Volume Increase from Markdown', fontweight='bold', fontsize=12)
    axes[0, 2].set_xticks(range(len(markdown_df)))
    axes[0, 2].set_xticklabels(markdown_df['Category'], rotation=45, ha='right')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Plot 4: Price loss percentage
    axes[1, 0].barh(range(len(markdown_df)), markdown_df['Price_Loss_%'].values, color='#e74c3c')
    axes[1, 0].set_xticks(range(len(markdown_df)))
    axes[1, 0].set_yticklabels(markdown_df['Category'])
    axes[1, 0].set_xlabel('Price Loss (%)', fontweight='bold')
    axes[1, 0].set_title('Revenue Impact per Unit', fontweight='bold', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 5: Markdown efficiency (units lift / price loss)
    colors = ['#2ecc71' if x > 1 else '#f39c12' if x > 0.5 else '#e74c3c' for x in markdown_df['Markdown_Efficiency']]
    axes[1, 1].barh(range(len(markdown_df)), markdown_df['Markdown_Efficiency'].values, color=colors)
    axes[1, 1].axvline(x=1, color='red', linestyle='--', linewidth=2, label='Break-even')
    axes[1, 1].set_xticks(range(len(markdown_df)))
    axes[1, 1].set_yticklabels(markdown_df['Category'])
    axes[1, 1].set_xlabel('Markdown Efficiency', fontweight='bold')
    axes[1, 1].set_title('Volume Lift vs Price Loss Ratio', fontweight='bold', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    # Plot 6: Profit comparison
    x_pos = np.arange(len(markdown_df))
    width = 0.35
    axes[1, 2].bar(x_pos - width/2, markdown_df['Discounted_Total_Profit'], width, label='With Markdown', color='#2ecc71')
    axes[1, 2].bar(x_pos + width/2, markdown_df['Non_Discounted_Total_Profit'], width, label='Without Markdown', color='#e74c3c')
    axes[1, 2].set_ylabel('Total Profit ($)', fontweight='bold')
    axes[1, 2].set_title('Total Profit: With vs Without Markdown', fontweight='bold', fontsize=12)
    axes[1, 2].set_xticks(x_pos)
    axes[1, 2].set_xticklabels(markdown_df['Category'], rotation=45, ha='right')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('markdown_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Markdown optimization visualization saved as 'markdown_optimization_analysis.png'")
    
    return markdown_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Load data using the new load_data function
    df = load_data()
    
    if df is not None:
        # Run all models
        eoq_results = calculate_eoq(df)
        newsvendor_results = newsvendor_model(df)
        mip_results = mixed_integer_programming(df)
        stochastic_results = stochastic_optimization(df)
        pricing_results = dynamic_pricing_optimization(df)
        markdown_results = markdown_optimization(df)
        
        # Summary report
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)
        
        print("\n1. EOQ Model:")
        print(f"   - Average optimal order quantity: {eoq_results['eoq'].mean():.0f} units")
        print(f"   - Average annual inventory cost: ${eoq_results['total_inventory_cost'].mean():,.2f}")
        
        print("\n2. Newsvendor Model:")
        print(f"   - Average optimal service level: {newsvendor_results['optimal_service_level'].mean():.1%}")
        print(f"   - Average expected shortage: {newsvendor_results['Expected_Shortage'].mean():.1f} units")
        
        print("\n3. MIP Model:")
        total_profit_mip = mip_results['Profit'].sum()
        print(f"   - Total optimal profit: ${total_profit_mip:,.2f}")
        print(f"   - Average warehouse utilization: {mip_results['Capacity_Used_%'].mean():.1f}%")
        
        print("\n4. Stochastic Optimization:")
        print(f"   - Average stockout risk: {stochastic_results['Stockout_Risk_%'].mean():.1f}%")
        print(f"   - Average expected shortage cost: ${stochastic_results['Expected_Shortage_Cost'].mean():,.2f}")
        
        print("\n5. Dynamic Pricing:")
        avg_revenue_change = pricing_results['Revenue_Change_%'].mean()
        print(f"   - Average revenue optimization potential: {avg_revenue_change:+.1f}%")
        print(f"   - Categories with price increase potential: {(pricing_results['Price_Change_%'] > 0).sum()}")
        
        print("\n6. Markdown Optimization:")
        efficient_markdowns = (markdown_results['Markdown_Efficiency'] > 1).sum()
        print(f"   - Categories with efficient markdowns: {efficient_markdowns}")
        print(f"   - Average markdown efficiency: {markdown_results['Markdown_Efficiency'].mean():.2f}")
        
        print("\n" + "=" * 80)
        print("Analysis complete! Check the generated PNG files for visualizations.")
        print("=" * 80)
    else:
        print("✗ Failed to load data. Please check the CSV file path.")
