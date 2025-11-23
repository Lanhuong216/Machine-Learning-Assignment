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
import warnings
warnings.filterwarnings('ignore')

# Import functions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, OUTPUT_VISUALIZATIONS_DIR, OUTPUT_REPORTS_DIR

def create_evaluation_metrics():
    """
    Tạo dictionary chứa các hàm đánh giá metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    def wmae(y_true, y_pred, weights):
        """Weighted Mean Absolute Error"""
        return np.average(np.abs(y_true - y_pred), weights=weights)
    
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'mae': mean_absolute_error,
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score,
        'wmae': wmae,
        'mape': mape
    }

def load_prepared_data():
    """
    Load dữ liệu đã được chuẩn bị từ preprocessing
    Sử dụng features từ feature_chosen.csv
    """
    print("=== LOAD DỮ LIỆU ĐÃ CHUẨN BỊ ===")
    
    try:
        # Load train_detail và test_detail từ preprocessing
        train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
        test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
        feature_chosen_path = os.path.join(DATA_PROCESSED_DIR, 'feature_chosen.csv')
        
        if not os.path.exists(train_detail_path):
            print(f"❌ Không tìm thấy file: {train_detail_path}")
            print("Vui lòng chạy preprocessing.py trước")
            return None, None, None, None, None
        
        # Load train_detail và test_detail
        train_detail = pd.read_csv(train_detail_path)
        test_detail = pd.read_csv(test_detail_path) if os.path.exists(test_detail_path) else None
        
        # Convert Date to datetime
        if 'Date' in train_detail.columns:
            train_detail['Date'] = pd.to_datetime(train_detail['Date'])
        if test_detail is not None and 'Date' in test_detail.columns:
            test_detail['Date'] = pd.to_datetime(test_detail['Date'])
        
        # Load feature_chosen.csv
        if os.path.exists(feature_chosen_path):
            feature_chosen_df = pd.read_csv(feature_chosen_path)
            feature_names = feature_chosen_df['Feature'].tolist()
            print(f"✓ Đã load {len(feature_names)} features từ feature_chosen.csv")
            print(f"  Features: {feature_names}")
        else:
            # Nếu không có feature_chosen.csv, lấy tất cả features trừ Date và Weekly_Sales
            feature_names = [col for col in train_detail.columns 
                           if col not in ['Date', 'Weekly_Sales']]
            print(f"⚠️ Không tìm thấy feature_chosen.csv, sử dụng tất cả {len(feature_names)} features")
        
        # Kiểm tra các features có trong train_detail không
        available_features = [f for f in feature_names if f in train_detail.columns]
        missing_features = [f for f in feature_names if f not in train_detail.columns]
        
        if missing_features:
            print(f"⚠️ Cảnh báo: Thiếu {len(missing_features)} features: {missing_features}")
        
        if len(available_features) == 0:
            print("❌ Không có features nào khả dụng")
            return None, None, None, None, None
        
        print(f"✓ Sử dụng {len(available_features)} features để train")
        
        # Tạo X và y từ train_detail
        X_train = train_detail[available_features].copy()
        y_train = train_detail['Weekly_Sales'].copy()
        
        # Xử lý missing values
        X_train = X_train.fillna(0)
        
        # Chia train/test theo time series split (80/20)
        # Sắp xếp theo Date để đảm bảo time series order
        if 'Date' in train_detail.columns:
            sorted_indices = train_detail.sort_values('Date').index
            X_train_sorted = X_train.loc[sorted_indices]
            y_train_sorted = y_train.loc[sorted_indices]
        else:
            X_train_sorted = X_train
            y_train_sorted = y_train
        
        # Chia 80/20 (giữ lại 20% cuối làm test set)
        split_idx = int(len(X_train_sorted) * 0.8)
        X_train_final = X_train_sorted.iloc[:split_idx]
        X_test_final = X_train_sorted.iloc[split_idx:]
        y_train_final = y_train_sorted.iloc[:split_idx]
        y_test_final = y_train_sorted.iloc[split_idx:]
        
        # Tạo weights cho WMAE từ IsHoliday
        weights = None
        if 'IsHoliday' in train_detail.columns:
            # Lấy IsHoliday cho test set (sau khi đã sort)
            if 'Date' in train_detail.columns:
                train_detail_sorted = train_detail.sort_values('Date')
                test_holiday = train_detail_sorted.iloc[split_idx:]['IsHoliday']
            else:
                test_holiday = train_detail.iloc[split_idx:]['IsHoliday']
            
            if test_holiday is not None and len(test_holiday) == len(y_test_final):
                # Convert IsHoliday sang numeric nếu cần
                if test_holiday.dtype == bool:
                    weights = np.where(test_holiday.values == True, 5, 1)
                elif test_holiday.dtype == object:
                    # String format (True/False)
                    weights = np.where((test_holiday == True) | (test_holiday == 'True') | (test_holiday == 1), 5, 1)
                else:
                    # Numeric format
                    weights = np.where(test_holiday.values == 1, 5, 1)
                print(f"✓ Đã tạo weights từ IsHoliday: {len(weights)} samples")
            else:
                weights = np.ones(len(y_test_final))
                print("⚠️ Không tìm thấy IsHoliday cho test set, sử dụng weights = 1")
        else:
            weights = np.ones(len(y_test_final))
            print("⚠️ Không có cột IsHoliday, sử dụng weights = 1")
        
        print(f"\n✓ Đã load và chuẩn bị dữ liệu:")
        print(f"  - X_train: {X_train_final.shape}")
        print(f"  - X_test: {X_test_final.shape}")
        print(f"  - y_train: {y_train_final.shape}")
        print(f"  - y_test: {y_test_final.shape}")
        print(f"  - Features: {len(available_features)}")
        
        return X_train_final, X_test_final, y_train_final, y_test_final, weights
    
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
        import traceback
        traceback.print_exc()
        print("Vui lòng chạy preprocessing.py trước")
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

def visualize_tuning_results(tuning_results_dict):
    """
    Trực quan hóa kết quả tuning (chỉ tuning time và best score)
    
    Args:
        tuning_results_dict (dict): Dictionary chứa kết quả tuning
    """
    print("\n=== TRỰC QUAN HÓA KẾT QUẢ TUNING ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Tuning time comparison
    axes[0].bar(tuning_results_dict.keys(), 
           [results['tuning_time']/60 for results in tuning_results_dict.values()],
           color=['steelblue', 'coral'])
    axes[0].set_title('Tuning Time Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Time (minutes)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 2. Best score comparison
    axes[1].bar(tuning_results_dict.keys(), 
           [-results['best_score'] for results in tuning_results_dict.values()],
           color=['steelblue', 'coral'])
    axes[1].set_title('Best Score from Tuning (MAE)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('MAE')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'hyperparameter_tuning_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {output_path}")

def save_tuned_models(models_dict, tuning_results_dict):
    """
    Lưu best parameters vào CSV (không lưu models)
    
    Args:
        models_dict (dict): Dictionary chứa các mô hình (không sử dụng, chỉ để tương thích)
        tuning_results_dict (dict): Dictionary chứa kết quả tuning
    """
    print("\n=== LƯU BEST PARAMETERS ===")
    
    from config import OUTPUT_REPORTS_DIR
    
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
    
    # Trực quan hóa kết quả tuning
    visualize_tuning_results(tuning_results)
    
    # Lưu best parameters (chỉ CSV)
    save_tuned_models(models, tuning_results)
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH BƯỚC 3: HYPERPARAMETER TUNING")
    print("="*80)
    print("\nCác file đã được tạo:")
    print("  - tuned_models_best_params.csv")
    print("  - hyperparameter_tuning_results.png")
    print("\nBây giờ có thể chạy train_with_best_params.py để train models với best parameters")
    
    return {
        'models': models,
        'tuning_results': tuning_results
    }

if __name__ == "__main__":
    results = main()

