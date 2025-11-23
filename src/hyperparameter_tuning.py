# BƯỚC 3: HYPERPARAMETER TUNING
# Người 2 - ML Engineer

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
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
from ml_data_preparation import create_evaluation_metrics
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
        
        return X_train, X_test, y_train, y_test, weights
    
    except FileNotFoundError as e:
        print(f"Không tìm thấy file: {e}")
        print("Vui lòng chạy ml_data_preparation.py trước")
        return None, None, None, None, None

def create_parameter_grids():
    """
    Tạo parameter grids cho các mô hình
    
    Returns:
        dict: Dictionary chứa parameter grids
    """
    print("\n=== TẠO PARAMETER GRIDS ===")
    
    param_grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': [0.8, 0.9, 1.0]
        },
        
        'xgboost': {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 6, 10, 15],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0],
            'min_child_weight': [1, 3, 5, 10]
        }
    }
    
    print("Đã tạo parameter grids cho Random Forest và XGBoost")
    return param_grids

def tune_random_forest(X_train, y_train, param_grid, n_iter=30, cv=3):
    """
    Tuning Random Forest với RandomizedSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        param_grid (dict): Parameter grid
        n_iter (int): Số lần thử nghiệm
        cv (int): Số folds cho cross-validation
        
    Returns:
        tuple: (best_model, best_params, tuning_results)
    """
    print("\n=== TUNING RANDOM FOREST ===")
    print(f"Số lần thử nghiệm: {n_iter}")
    print(f"Số folds CV: {cv}")
    print("Đang tuning Random Forest...")
    
    # Base model
    rf = RandomForestRegressor(random_state=42, n_jobs=-1, verbose=0)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv) # Time series cross-validation (chia dữ liệu thành các folds, mỗi fold là một thời gian, và dữ liệu trong mỗi fold là liên tục theo thời gian)
    
    # Randomized search
    search = RandomizedSearchCV(
        rf, param_grid, n_iter=n_iter, cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"\nĐã tuning Random Forest trong {tuning_time/60:.2f} phút ({tuning_time:.2f} giây)")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score (MAE): {-search.best_score_:.4f}")
    
    # Lưu kết quả tuning
    tuning_results = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'tuning_time': tuning_time,
        'cv_results': search.cv_results_
    }
    
    return search.best_estimator_, search.best_params_, tuning_results

def tune_xgboost(X_train, y_train, param_grid, n_iter=30, cv=3):
    """
    Tuning XGBoost với RandomizedSearchCV
    
    Args:
        X_train: Training features
        y_train: Training target
        param_grid (dict): Parameter grid
        n_iter (int): Số lần thử nghiệm
        cv (int): Số folds cho cross-validation
        
    Returns:
        tuple: (best_model, best_params, tuning_results)
    """
    print("\n=== TUNING XGBOOST ===")
    print(f"Số lần thử nghiệm: {n_iter}")
    print(f"Số folds CV: {cv}")
    print("Đang tuning XGBoost...")
    
    # Base model
    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Randomized search
    search = RandomizedSearchCV(
        xgb_model, param_grid, n_iter=n_iter, cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42, n_jobs=-1, verbose=1
    )
    
    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time
    
    print(f"\nĐã tuning XGBoost trong {tuning_time/60:.2f} phút ({tuning_time:.2f} giây)")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best score (MAE): {-search.best_score_:.4f}")
    
    # Lưu kết quả tuning
    tuning_results = {
        'best_score': search.best_score_,
        'best_params': search.best_params_,
        'tuning_time': tuning_time,
        'cv_results': search.cv_results_
    }
    
    return search.best_estimator_, search.best_params_, tuning_results

def evaluate_tuned_models(models, X_test, y_test, metrics=None, weights=None):
    """
    Đánh giá các mô hình đã được tuning
    
    Args:
        models (dict): Dictionary chứa các mô hình đã tuning
        X_test: Test features
        y_test: Test target
        metrics (dict): Dictionary chứa các hàm đánh giá
        weights (array): Weights cho WMAE
        
    Returns:
        dict: Kết quả đánh giá
    """
    print("\n=== ĐÁNH GIÁ CÁC MÔ HÌNH ĐÃ TUNING ===")
    
    if metrics is None:
        metrics = create_evaluation_metrics()
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nĐánh giá {model_name}...")
        
        # Dự đoán
        y_pred = model.predict(X_test)
        
        # Tính metrics
        model_results = {}
        for metric_name, metric_func in metrics.items():
            if metric_name == 'wmae' and weights is not None:
                model_results[metric_name] = metric_func(y_test, y_pred, weights)
            else:
                model_results[metric_name] = metric_func(y_test, y_pred)
        
        results[model_name] = model_results
        
        print(f"MAE: {model_results['mae']:.2f}")
        print(f"RMSE: {model_results['rmse']:.2f}")
        print(f"R²: {model_results['r2']:.4f}")
        print(f"WMAE: {model_results['wmae']:.2f}")
    
    return results

def compare_with_baseline(tuned_results, baseline_path=None):
    if baseline_path is None:
        from config import OUTPUT_REPORTS_DIR
        baseline_path = os.path.join(OUTPUT_REPORTS_DIR, 'baseline_models_comparison.csv')
    """
    So sánh kết quả tuned với baseline
    
    Args:
        tuned_results (dict): Kết quả các mô hình đã tuning
        baseline_path (str): Đường dẫn đến file baseline results
        
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    print("\n=== SO SÁNH VỚI BASELINE ===")
    
    # Load baseline results
    try:
        baseline_df = pd.read_csv(baseline_path)
        print("✓ Đã load kết quả baseline")
    except FileNotFoundError:
        print("⚠️ Không tìm thấy file baseline, chỉ hiển thị kết quả tuned")
        baseline_df = None
    
    # Tạo comparison data
    comparison_data = []
    
    # Thêm tuned models
    for model_name, results in tuned_results.items():
        comparison_data.append({
            'Model': f"{model_name} (Tuned)",
            'MAE': results['mae'],
            'RMSE': results['rmse'],
            'R²': results['r2'],
            'WMAE': results['wmae']
        })
    
    # Thêm baseline models nếu có
    if baseline_df is not None:
        for _, row in baseline_df.iterrows():
            comparison_data.append({
                'Model': row['Model'] + ' (Baseline)',
                'MAE': row['MAE'],
                'RMSE': row['RMSE'],
                'R²': row['R²'],
                'WMAE': row['WMAE']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values('WMAE')
    
    print("\n" + "="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    
    # Tính improvement
    if baseline_df is not None:
        print("\n=== CẢI THIỆN SO VỚI BASELINE ===")
        for model_name, results in tuned_results.items():
            # Tìm baseline tương ứng
            baseline_name = model_name.replace(' (Tuned)', '')
            baseline_row = baseline_df[baseline_df['Model'] == baseline_name]
            
            if not baseline_row.empty:
                baseline_wmae = baseline_row['WMAE'].iloc[0]
                tuned_wmae = results['wmae']
                improvement = ((baseline_wmae - tuned_wmae) / baseline_wmae) * 100
                
                print(f"{model_name}:")
                print(f"  Baseline WMAE: {baseline_wmae:.2f}")
                print(f"  Tuned WMAE: {tuned_wmae:.2f}")
                print(f"  Improvement: {improvement:.2f}%")
    
    return comparison_df

def visualize_tuning_results(tuning_results_dict):
    """
    Trực quan hóa kết quả tuning
    
    Args:
        tuning_results_dict (dict): Dictionary chứa kết quả tuning
    """
    print("\n=== TRỰC QUAN HÓA KẾT QUẢ TUNING ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Tuning time comparison
    axes[0].bar(tuning_results_dict.keys(), 
               [results['tuning_time']/60 for results in tuning_results_dict.values()])
    axes[0].set_title('Tuning Time Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time (minutes)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Best score comparison
    axes[1].bar(tuning_results_dict.keys(), 
               [-results['best_score'] for results in tuning_results_dict.values()])
    axes[1].set_title('Best Score Comparison (MAE)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    from config import OUTPUT_VISUALIZATIONS_DIR
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'hyperparameter_tuning_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ: {output_path}")
    plt.show()

def save_tuned_models(models_dict, tuning_results_dict):
    """
    Lưu các mô hình đã tuning
    
    Args:
        models_dict (dict): Dictionary chứa các mô hình
        tuning_results_dict (dict): Dictionary chứa kết quả tuning
    """
    print("\n=== LƯU CÁC MÔ HÌNH ĐÃ TUNING ===")
    
    from config import MODELS_DIR, OUTPUT_REPORTS_DIR
    for model_name, model in models_dict.items():
        filename = f"tuned_{model_name.lower().replace(' ', '_')}_model.pkl"
        filepath = os.path.join(MODELS_DIR, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Đã lưu: {filepath}")
    
    # Lưu best parameters
    best_params_df = pd.DataFrame({
        'Model': list(tuning_results_dict.keys()),
        'Best_Params': [str(results['best_params']) for results in tuning_results_dict.values()],
        'Best_Score_MAE': [-results['best_score'] for results in tuning_results_dict.values()],
        'Tuning_Time_Minutes': [results['tuning_time']/60 for results in tuning_results_dict.values()]
    })
    output_path = os.path.join(OUTPUT_REPORTS_DIR, 'tuned_models_best_params.csv')
    best_params_df.to_csv(output_path, index=False)
    print(f"✓ Đã lưu: {output_path}")

def main():
    """Hàm chính để chạy BƯỚC 3"""
    print("="*80)
    print("BƯỚC 3: HYPERPARAMETER TUNING")
    print("="*80)
    
    # Load dữ liệu đã chuẩn bị
    X_train, X_test, y_train, y_test, weights = load_prepared_data()
    
    if X_train is None:
        print("\n❌ Không thể tiếp tục do thiếu dữ liệu")
        return None
    
    # Tạo parameter grids
    param_grids = create_parameter_grids()
    
    # Tuning các mô hình
    models = {}
    tuning_results = {}
    
    # 1. Random Forest
    print("\n" + "="*80)
    rf_model, rf_params, rf_tuning = tune_random_forest(
        X_train, y_train, param_grids['random_forest'], n_iter=30, cv=3
    )
    models['Random Forest'] = rf_model
    tuning_results['Random Forest'] = rf_tuning
    
    # 2. XGBoost
    print("\n" + "="*80)
    xgb_model, xgb_params, xgb_tuning = tune_xgboost(
        X_train, y_train, param_grids['xgboost'], n_iter=30, cv=3
    )
    models['XGBoost'] = xgb_model
    tuning_results['XGBoost'] = xgb_tuning
    
    # Đánh giá các mô hình đã tuning
    print("\n" + "="*80)
    metrics = create_evaluation_metrics()
    evaluation_results = evaluate_tuned_models(models, X_test, y_test, metrics, weights)
    
    # So sánh với baseline
    print("\n" + "="*80)
    comparison_df = compare_with_baseline(evaluation_results)
    
    # Trực quan hóa
    visualize_tuning_results(tuning_results)
    
    # Lưu mô hình
    save_tuned_models(models, tuning_results)
    
    # Lưu kết quả
    from config import OUTPUT_REPORTS_DIR
    evaluation_df = pd.DataFrame(evaluation_results).T
    evaluation_df.to_csv(os.path.join(OUTPUT_REPORTS_DIR, 'tuned_models_evaluation.csv'))
    comparison_df.to_csv(os.path.join(OUTPUT_REPORTS_DIR, 'tuned_vs_baseline_comparison.csv'), index=False)
    
    print(f"\n✓ Đã lưu kết quả:")
    print(f"  - tuned_models_evaluation.csv")
    print(f"  - tuned_vs_baseline_comparison.csv")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH BƯỚC 3: HYPERPARAMETER TUNING")
    print("="*80)
    print("\nCác file đã được tạo:")
    print("  - tuned_random_forest_model.pkl")
    print("  - tuned_xgboost_model.pkl")
    print("  - tuned_models_best_params.csv")
    print("  - tuned_models_evaluation.csv")
    print("  - tuned_vs_baseline_comparison.csv")
    print("  - hyperparameter_tuning_results.png")
    print("\nBây giờ có thể tiếp tục BƯỚC 4 (Model Evaluation & Analysis)")
    
    return {
        'models': models,
        'tuning_results': tuning_results,
        'evaluation_results': evaluation_results,
        'comparison': comparison_df
    }

if __name__ == "__main__":
    results = main()

