# BƯỚC 2: BASELINE MODELS
# Người 2 - ML Engineer

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import functions từ ml_data_preparation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_data_preparation import create_evaluation_metrics, evaluate_model
from config import DATA_PROCESSED_DIR

def load_prepared_data():
    """Load dữ liệu đã được chuẩn bị từ BƯỚC 1"""
    print("=== LOAD DỮ LIỆU ĐÃ CHUẨN BỊ ===")
    
    try:
        X_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'))
        y_train = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_train.csv')).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv')).iloc[:, 0]
        weights = np.load(os.path.join(DATA_PROCESSED_DIR, 'weights.npy'))
        
        print(f"✓ Đã load dữ liệu:")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - weights: {weights.shape}")
        
        return X_train, X_test, y_train, y_test, weights
    
    except FileNotFoundError as e:
        print(f"Không tìm thấy file: {e}")
        print("Vui lòng chạy ml_data_preparation.py trước")
        return None, None, None, None, None

def train_linear_regression(X_train, X_test, y_train, y_test, metrics=None, weights=None):
    """
    Huấn luyện Linear Regression
    
    Args:
        X_train, X_test: Training và test features
        y_train, y_test: Training và test targets
        metrics (dict): Dictionary chứa các hàm đánh giá
        weights (array): Weights cho WMAE
        
    Returns:
        tuple: (model, results, training_time)
    """
    print("\n=== LINEAR REGRESSION ===")
    
    start_time = time.time()
    
    # Tạo và huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train, y_train) # Huấn luyện mô hình bằng cách tìm hệ số beta cho phương trình y = beta0 + beta1*x1 + beta2*x2 + ... + betan*xn (y là target, x1, x2, ..., xn là features)
    
    # Dự đoán
    y_pred = model.predict(X_test) # Dự đoán target cho test set
    
    training_time = time.time() - start_time
    
    # Đánh giá
    if metrics is None:
        metrics = create_evaluation_metrics()
    
    results = {}
    """
        Logic tính điểm: Đoạn vòng lặp for này duyệt qua từng loại thước đo (MAE, RMSE...) để tính toán điểm cho mô hình:
    -   Nếu là wmae: Nó gọi hàm tính WMAE và truyền thêm tham số weights (trọng số ngày lễ) vào.
    -   Nếu là thước đo thường (MAE, RMSE): Nó chỉ cần so sánh đáp án thực (y_test) và dự đoán (y_pred).
    """
    for metric_name, metric_func in metrics.items():
        if metric_name == 'wmae' and weights is not None:
            results[metric_name] = metric_func(y_test, y_pred, weights)
        else:
            results[metric_name] = metric_func(y_test, y_pred)
    
    results['training_time'] = training_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"MAE: {results['mae']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"WMAE: {results['wmae']:.2f}")
    
    return model, results, training_time

def train_random_forest(X_train, X_test, y_train, y_test, metrics=None, weights=None,
                       n_estimators=100, max_depth=None, random_state=42):
    """
    Huấn luyện Random Forest
    
    Args:
        X_train, X_test: Training và test features
        y_train, y_test: Training và test targets
        metrics (dict): Dictionary chứa các hàm đánh giá
        weights (array): Weights cho WMAE
        n_estimators (int): Số cây
        max_depth (int): Độ sâu tối đa
        random_state (int): Random seed
        
    Returns:
        tuple: (model, results, training_time)
    """
    print("\n=== RANDOM FOREST ===")
    
    start_time = time.time()
    
    # Tạo và huấn luyện mô hình
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1, #Sử dụng số lượng CPU (-1 = maximum)
        verbose=0
    )
    model.fit(X_train, y_train) # Huấn luyện mô hình bằng cách tìm các cây quyết định cho Random Forest
    
    # Dự đoán
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # Đánh giá
    if metrics is None:
        metrics = create_evaluation_metrics()
    
    results = {}
    for metric_name, metric_func in metrics.items():
        if metric_name == 'wmae' and weights is not None:
            results[metric_name] = metric_func(y_test, y_pred, weights)
        else:
            results[metric_name] = metric_func(y_test, y_pred)
    
    results['training_time'] = training_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"MAE: {results['mae']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"WMAE: {results['wmae']:.2f}")
    
    return model, results, training_time

def train_xgboost(X_train, X_test, y_train, y_test, metrics=None, weights=None,
                 n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
    """
    Huấn luyện XGBoost
    
    Args:
        X_train, X_test: Training và test features
        y_train, y_test: Training và test targets
        metrics (dict): Dictionary chứa các hàm đánh giá
        weights (array): Weights cho WMAE
        n_estimators (int): Số cây
        learning_rate (float): Learning rate (tốc độ học, số càng nhỏ thì mô hình học càng chậm nhưng chính xác hơn)
        max_depth (int): Độ sâu tối đa (số lượng cây con tối đa mà mỗi cây quyết định có thể có, mặc định là 6 vì XGBoost là mô hình có độ sâu tối đa là 6)
        random_state (int): Random seed (seed là một số nguyên để khởi tạo một giá trị ngẫu nhiên, để đảm bảo rằng kết quả của mô hình là reproducible)
        
    Returns:
        tuple: (model, results, training_time)
    """
    print("\n=== XGBOOST ===")
    
    start_time = time.time()
    
    # Tạo và huấn luyện mô hình
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1, #Sử dụng số lượng CPU (-1 = maximum)
        verbosity=0
    )
    model.fit(X_train, y_train) # Huấn luyện mô hình bằng XGBoost
    
    # Dự đoán
    y_pred = model.predict(X_test) # Dự đoán target cho test set
    
    training_time = time.time() - start_time
    
    # Đánh giá
    if metrics is None:
        metrics = create_evaluation_metrics()
    
    results = {}
    for metric_name, metric_func in metrics.items():
        if metric_name == 'wmae' and weights is not None:
            results[metric_name] = metric_func(y_test, y_pred, weights)
        else:
            results[metric_name] = metric_func(y_test, y_pred)
    
    results['training_time'] = training_time
    
    print(f"Training time: {training_time:.2f} seconds")
    print(f"MAE: {results['mae']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"WMAE: {results['wmae']:.2f}")
    
    return model, results, training_time

def compare_baseline_models(models_results):
    """
    So sánh các mô hình cơ sở
    
    Args:
        models_results (dict): Dictionary chứa kết quả các mô hình
        
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    print("\n=== SO SÁNH CÁC MÔ HÌNH CƠ SỞ ===")
    
    # Tạo bảng so sánh
    comparison_data = []
    for model_name, results in models_results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': results['mae'],
            'MSE': results['mse'],
            'RMSE': results['rmse'],
            'R²': results['r2'],
            'WMAE': results['wmae'],
            'Training Time (s)': results['training_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    
    # Sắp xếp theo WMAE (metric chính)
    comparison_df = comparison_df.sort_values('WMAE')
    
    print("\n" + "="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Tìm mô hình tốt nhất
    best_model_wmae = comparison_df.iloc[0]['Model']
    best_model_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
    best_model_r2 = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
    
    print(f"\n🏆 Mô hình tốt nhất theo WMAE: {best_model_wmae}")
    print(f"🏆 Mô hình tốt nhất theo MAE: {best_model_mae}")
    print(f"🏆 Mô hình tốt nhất theo R²: {best_model_r2}")
    
    return comparison_df

def visualize_baseline_results(models_results):
    """
    Trực quan hóa kết quả các mô hình cơ sở
    
    Args:
        models_results (dict): Dictionary chứa kết quả các mô hình
    """
    print("\n=== TRỰC QUAN HÓA KẾT QUẢ ===")
    
    n_models = len(models_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. So sánh MAE
    axes[0, 0].bar(models_results.keys(), [results['mae'] for results in models_results.values()])
    axes[0, 0].set_title('MAE Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. So sánh RMSE
    axes[0, 1].bar(models_results.keys(), [results['rmse'] for results in models_results.values()])
    axes[0, 1].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. So sánh R²
    axes[1, 0].bar(models_results.keys(), [results['r2'] for results in models_results.values()])
    axes[1, 0].set_title('R² Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. So sánh WMAE (metric chính)
    axes[1, 1].bar(models_results.keys(), [results['wmae'] for results in models_results.values()], color='coral')
    axes[1, 1].set_title('WMAE Comparison (Primary Metric)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('WMAE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    from config import OUTPUT_VISUALIZATIONS_DIR
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'baseline_models_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ: {output_path}")
    plt.show()

def save_baseline_models(models_dict):
    """
    Lưu các mô hình cơ sở
    
    Args:
        models_dict (dict): Dictionary chứa các mô hình
    """
    print("\n=== LƯU CÁC MÔ HÌNH CƠ SỞ ===")
    
    from config import MODELS_DIR
    for model_name, model in models_dict.items():
        filename = f"baseline_{model_name.lower().replace(' ', '_')}_model.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Đã lưu: {filepath}")

def main():
    """Hàm chính để chạy BƯỚC 2"""
    print("="*80)
    print("BƯỚC 2: BASELINE MODELS")
    print("="*80)
    
    # Load dữ liệu đã chuẩn bị
    X_train, X_test, y_train, y_test, weights = load_prepared_data()
    
    if X_train is None:
        print("\n❌ Không thể tiếp tục do thiếu dữ liệu")
        return None
    
    # Tạo evaluation metrics
    metrics = create_evaluation_metrics()
    
    # Huấn luyện các mô hình cơ sở
    models = {}
    models_results = {}
    
    # 1. Linear Regression
    lr_model, lr_results, lr_time = train_linear_regression(
        X_train, X_test, y_train, y_test, metrics, weights
    )
    models['Linear Regression'] = lr_model
    models_results['Linear Regression'] = lr_results
    
    # 2. Random Forest
    rf_model, rf_results, rf_time = train_random_forest(
        X_train, X_test, y_train, y_test, metrics, weights
    )
    models['Random Forest'] = rf_model
    models_results['Random Forest'] = rf_results
    
    # 3. XGBoost
    xgb_model, xgb_results, xgb_time = train_xgboost(
        X_train, X_test, y_train, y_test, metrics, weights
    )
    models['XGBoost'] = xgb_model
    models_results['XGBoost'] = xgb_results
    
    # So sánh các mô hình
    comparison_df = compare_baseline_models(models_results)
    
    # Trực quan hóa
    visualize_baseline_results(models_results)
    
    # Lưu mô hình
    save_baseline_models(models)
    
    # Lưu kết quả
    from config import OUTPUT_REPORTS_DIR
    output_path = os.path.join(OUTPUT_REPORTS_DIR, 'baseline_models_comparison.csv')
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Đã lưu kết quả: {output_path}")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH BƯỚC 2: BASELINE MODELS")
    print("="*80)
    print("\nCác file đã được tạo:")
    print("  - baseline_linear_regression_model.pkl")
    print("  - baseline_random_forest_model.pkl")
    print("  - baseline_xgboost_model.pkl")
    print("  - baseline_models_comparison.csv")
    print("  - baseline_models_comparison.png")
    print("\nBây giờ có thể tiếp tục BƯỚC 3 (Advanced Models & Hyperparameter Tuning)")
    
    return {
        'models': models,
        'results': models_results,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    results = main()
