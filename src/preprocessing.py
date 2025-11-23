from traceback import print_tb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
sns.set_style("whitegrid")

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUT_VISUALIZATIONS_DIR

#load the data
features=pd.read_csv(os.path.join(DATA_RAW_DIR, "walmart-features.csv"))
store=pd.read_csv(os.path.join(DATA_RAW_DIR, "walmart-stores.csv"))
train=pd.read_csv(os.path.join(DATA_RAW_DIR, "walmart-train.csv"))
test=pd.read_csv(os.path.join(DATA_RAW_DIR, "walmart-test.csv"))

#print the data
print("Features:")
print(features.head())
print("Store:") 
print(store.head())
print("Train:")
print(train.head())
print("Test:")
print(test.describe())

feature_store=pd.merge(features,store,on="Store", how="inner")
print("Feature types:")
print(feature_store.dtypes)

#convert the date to datetime
feature_store.Date=pd.to_datetime(feature_store.Date)
train.Date=pd.to_datetime(train.Date)
test.Date=pd.to_datetime(test.Date)

#add the week, month, year, and day to the feature_store
feature_store['Week']=feature_store.Date.dt.isocalendar().week
feature_store['Month']=feature_store.Date.dt.month
feature_store['Year']=feature_store.Date.dt.year

#merge the train data and test data with the feature_store
train_detail=train.merge(feature_store, how="inner", on=["Store","Date", "IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)
print("Train detail:")
print(train_detail.head())

test_detail=test.merge(feature_store, how="inner", on=["Store","Date", "IsHoliday"]).sort_values(by=["Store","Dept","Date"]).reset_index(drop=True)
print("Test detail:")
print(test_detail.head())
#summarize null values and data types
null_summary = pd.DataFrame({
    'Column': train_detail.columns,
    'Null_Count': train_detail.isnull().sum().values,
    'Null_Percentage': (train_detail.isnull().sum() / len(train_detail) * 100).values,
    'Data_Type': train_detail.dtypes.values
})
print("Null values and data types of train_detail:")
print(null_summary.to_string(index=False))


#holiday effect check
holiday_sales=train_detail.groupby('IsHoliday')['Weekly_Sales'].mean()
print("Average sales on holidays:")
print(holiday_sales)

holiday_sales_plot=holiday_sales.plot(kind='bar', color=['skyblue', 'salmon'])
plt.xlabel('IsHoliday')
plt.ylabel('Average Sales')
plt.title('Holiday Effect')
plt.tight_layout()
output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'holiday_effect.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {output_path}")

# Calculate mean and median of Weekly_Sales by Date
sales_by_date = train_detail.groupby('Date')['Weekly_Sales'].agg(['mean', 'median']).reset_index()
sales_by_date = sales_by_date.sort_values('Date')

print("Mean and Median of Weekly_Sales by Date:")
print(sales_by_date.head(10))

# Calculate mean and median of Weekly_Sales by Date
sales_by_date = train_detail.groupby('Date')['Weekly_Sales'].agg(['mean', 'median']).reset_index()
sales_by_date = sales_by_date.sort_values('Date')

print("Mean and Median of Weekly_Sales by Date:")
print(sales_by_date.head(10))

# Create correlation plot between mean and median by Date
fig, (ax1) = plt.subplots(1, figsize=(15, 5))

# Line plot: Mean and Median over time
ax1.plot(sales_by_date['Date'], sales_by_date['mean'], label='Mean', color='blue', linewidth=2)
ax1.plot(sales_by_date['Date'], sales_by_date['median'], label='Median', color='red', linewidth=2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Weekly Sales')
ax1.set_title('Mean vs Median of Weekly_Sales over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)


# Calculate and display correlation coefficient
correlation = sales_by_date['mean'].corr(sales_by_date['median'])

plt.tight_layout()
output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'mean_median_sales_over_time.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {output_path}")

# Calculate average sales by Store
sales_by_store = train_detail.groupby('Store')['Weekly_Sales'].mean().sort_index(ascending=True)
print("Average Sales by Store:")
print(sales_by_store.head(10))

# Calculate average sales by Dept
sales_by_dept = train_detail.groupby('Dept')['Weekly_Sales'].mean().sort_index(ascending=True)
print("Average Sales by Dept:")
print(sales_by_dept.head(10))

# Create plots for average sales by Store and Dept
fig, (ax1) = plt.subplots(1, figsize=(18, 6))

# Bar plot: Average Sales by Store
sales_by_store.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.7)
ax1.set_xlabel('Store', fontsize=12)
ax1.set_ylabel('Average Weekly Sales', fontsize=12)
ax1.set_title('Average Weekly Sales by Store', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', rotation=90, labelsize=8)
ax1.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'average_sales_by_store.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {output_path}")

fig, (ax2) = plt.subplots(1, figsize=(18, 6))
# Bar plot: Average Sales by Dept
sales_by_dept.plot(kind='bar', ax=ax2, color='coral', alpha=0.7)
ax2.set_xlabel('Dept', fontsize=12)
ax2.set_ylabel('Average Weekly Sales', fontsize=12)
ax2.set_title('Average Weekly Sales by Dept', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'average_sales_by_dept.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {output_path}")

#correlation matrix
numeric_cols = train_detail.select_dtypes(include=[np.number]).columns.tolist()
correlation_matrix = train_detail[numeric_cols].corr()
plt.figure(figsize=(14, 12))

sns.heatmap(correlation_matrix,  annot=True, cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Fields in train_detail', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'correlation_matrix.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Đã lưu: {output_path}")

# Function to plot boxplot between two fields
def plot_boxplot(x_column, data=train_detail, figsize=(10, 6), title=None, rotation=45):
   
    # Check if columns exist
    if x_column not in data.columns:
        print(f"Error: Column '{x_column}' not found in data")
        print(f"Available columns: {data.columns.tolist()}")
        return
    
    if 'Weekly_Sales' not in data.columns:
        print("Error: Column 'Weekly_Sales' not found in data")
        return
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot boxplot
    try:
        sns.boxplot(data=data, x=x_column, y='Weekly_Sales', palette='Set2')
    except:
        # Use matplotlib if seaborn is not available
        unique_values = data[x_column].unique()
        box_data = [data[data[x_column] == val]['Weekly_Sales'].dropna() for val in unique_values]
        plt.boxplot(box_data, labels=unique_values)
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel('Weekly Sales', fontsize=12)
    
    # Set labels and title
    if title is None:
        title = f'Weekly Sales Distribution by {x_column}'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_column, fontsize=12)
    plt.ylabel('Weekly Sales', fontsize=12)
    plt.xticks(rotation=rotation, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, f'boxplot_{x_column.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")
    
    # Print statistics
    print(f"\nStatistics by {x_column}:")
    stats = data.groupby(x_column)['Weekly_Sales'].agg(['mean', 'median', 'std', 'count'])
    print(stats.round(2))

plot_boxplot('IsHoliday')
plot_boxplot('Type')
train_detail.Type = train_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
test_detail.Type = test_detail.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))

# Function to plot correlation between Weekly_Sales and a continuous variable
def plot_correlation(x_column, data=train_detail, figsize=(10, 6), title=None, alpha=0.5, s=20):
    try:
        from scipy.stats import pearsonr
    except ImportError:
        print("Error: scipy is required for correlation analysis. Install with: pip install scipy")
        return
    
    # Check if columns exist
    if x_column not in data.columns:
        print(f"Error: Column '{x_column}' not found in data")
        print(f"Available columns: {data.columns.tolist()}")
        return
    
    if 'Weekly_Sales' not in data.columns:
        print("Error: Column 'Weekly_Sales' not found in data")
        return
    
    # Check if x_column is numeric
    if not pd.api.types.is_numeric_dtype(data[x_column]):
        print(f"Error: Column '{x_column}' is not numeric. Use plot_boxplot() for categorical variables.")
        return
    
    # Remove missing values
    plot_data = data[[x_column, 'Weekly_Sales']].dropna()
    
    if len(plot_data) == 0:
        print(f"Error: No valid data points after removing missing values")
        return
    
    # Calculate correlation
    correlation, p_value = pearsonr(plot_data[x_column], plot_data['Weekly_Sales'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot with regression line
    ax1.scatter(plot_data[x_column], plot_data['Weekly_Sales'], alpha=alpha, s=s, color='steelblue')
    
    # Add regression line
    z = np.polyfit(plot_data[x_column], plot_data['Weekly_Sales'], 1)
    p = np.poly1d(z)
    ax1.plot(plot_data[x_column], p(plot_data[x_column]), "r--", linewidth=2, label=f'Linear fit (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Set labels and title
    if title is None:
        title = f'Correlation: Weekly Sales vs {x_column}'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(x_column, fontsize=12)
    ax1.set_ylabel('Weekly Sales', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation info
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nP-value: {p_value:.4f}',
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Density plot (hexbin or 2D histogram) for better visualization of density
    try:
        hb = ax2.hexbin(plot_data[x_column], plot_data['Weekly_Sales'], gridsize=30, cmap='Blues', alpha=0.7)
        ax2.set_title(f'Density Plot: Weekly Sales vs {x_column}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_column, fontsize=12)
        ax2.set_ylabel('Weekly Sales', fontsize=12)
        plt.colorbar(hb, ax=ax2, label='Count')
    except:
        # Fallback to 2D histogram if hexbin fails
        ax2.hist2d(plot_data[x_column], plot_data['Weekly_Sales'], bins=30, cmap='Blues', alpha=0.7)
        ax2.set_title(f'Density Plot: Weekly Sales vs {x_column}', fontsize=14, fontweight='bold')
        ax2.set_xlabel(x_column, fontsize=12)
        ax2.set_ylabel('Weekly Sales', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, f'correlation_{x_column.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {output_path}")
    
    # Print statistics
    print(f"\nCorrelation Analysis: {x_column} vs Weekly_Sales")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant correlation (p < 0.05)")
    else:
        print("Not statistically significant (p >= 0.05)")
    
    print(f"\nBasic Statistics:")
    print(f"{x_column}: mean={plot_data[x_column].mean():.2f}, std={plot_data[x_column].std():.2f}")
    print(f"Weekly_Sales: mean={plot_data['Weekly_Sales'].mean():.2f}, std={plot_data['Weekly_Sales'].std():.2f}")

plot_correlation('Temperature')
plot_correlation('Fuel_Price')
plot_correlation('CPI')
plot_correlation('Unemployment')
plot_correlation('Size')

# =============================================================================
# CHỌN VÀ LƯU CÁC FEATURES ĐÃ CHỈ ĐỊNH
# =============================================================================
print("\n CHỌN VÀ LƯU CÁC FEATURES")

# Danh sách features được chỉ định
selected_features = ['Store', 'Dept', 'IsHoliday', 'Size', 'Type', 'Year', 'Week', 'Month']

# Kiểm tra các features có trong train_detail không
available_features = [f for f in selected_features if f in train_detail.columns]
missing_features = [f for f in selected_features if f not in train_detail.columns]

print(f"\n Sử dụng {len(available_features)}/{len(selected_features)} features:")
for i, feature in enumerate(available_features, 1):
    print(f"  {i:2d}. {feature}")

# Lưu danh sách features đã chọn vào feature_chosen.csv
feature_chosen_df = pd.DataFrame({
    'Feature': available_features,
    'Selected': ['Yes'] * len(available_features)
})
feature_chosen_path = os.path.join(DATA_PROCESSED_DIR, 'feature_chosen.csv')
feature_chosen_df.to_csv(feature_chosen_path, index=False)

# Xác định các cột cần giữ lại cho train_detail
# train_detail cần: Date, Weekly_Sales + các features đã chọn
train_keep_cols = ['Date', 'Weekly_Sales'] + available_features
train_keep_cols = [col for col in train_keep_cols if col in train_detail.columns]

# Xác định các cột cần giữ lại cho test_detail
# test_detail cần: Date + các features đã chọn (không có Weekly_Sales)
test_keep_cols = ['Date'] + available_features
test_keep_cols = [col for col in test_keep_cols if col in test_detail.columns]

# Tạo train_detail và test_detail chỉ với features đã chọn
train_detail_selected = train_detail[train_keep_cols].copy()
test_detail_selected = test_detail[test_keep_cols].copy()

# Xử lý missing values cho các features đã chọn
# Chỉ fillna cho các cột numeric, giữ nguyên boolean/categorical
for col in available_features:
    if col in train_detail_selected.columns:
        if train_detail_selected[col].dtype in [np.float64, np.int64]:
            train_detail_selected[col] = train_detail_selected[col].fillna(0)
    if col in test_detail_selected.columns:
        if test_detail_selected[col].dtype in [np.float64, np.int64]:
            test_detail_selected[col] = test_detail_selected[col].fillna(0)

# Lưu train_detail và test_detail ra file CSV
train_detail_selected.to_csv(os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv'), index=False)
test_detail_selected.to_csv(os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv'), index=False)

print(f"Đã lưu train_detail.csv: {train_detail_selected.shape}")
print(f"Đã lưu test_detail.csv: {test_detail_selected.shape}")

# Lưu thông tin summary
print(f"\nTrain Detail Summary (sau khi chọn features):")
print(f"  - Shape: {train_detail_selected.shape}")
print(f"  - Columns: {list(train_detail_selected.columns)}")
print(f"  - Features đã chọn: {len(available_features)}")
print(f"  - Date range: {train_detail_selected['Date'].min()} to {train_detail_selected['Date'].max()}")
if 'Store' in train_detail_selected.columns:
    print(f"  - Stores: {train_detail_selected['Store'].nunique()}")
if 'Dept' in train_detail_selected.columns:
    print(f"  - Departments: {train_detail_selected['Dept'].nunique()}")

print(f"\nTest Detail Summary (sau khi chọn features):")
print(f"  - Shape: {test_detail_selected.shape}")
print(f"  - Columns: {list(test_detail_selected.columns)}")
print(f"  - Features đã chọn: {len(available_features)}")
print(f"  - Date range: {test_detail_selected['Date'].min()} to {test_detail_selected['Date'].max()}")
if 'Store' in test_detail_selected.columns:
    print(f"  - Stores: {test_detail_selected['Store'].nunique()}")
if 'Dept' in test_detail_selected.columns:
    print(f"  - Departments: {test_detail_selected['Dept'].nunique()}")

print("\n HOÀN THÀNH PREPROCESSING!")
