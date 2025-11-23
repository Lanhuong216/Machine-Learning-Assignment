"""
K-Fold Cross-Validation cho Random Forest và XGBoost
Chọn model tốt nhất và xuất output
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

warnings.filterwarnings('ignore')

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, OUTPUT_DIR


def load_preprocessed_data():
    """
    Load train_detail đã được preprocess
    
    Returns:
        tuple: (train_detail, feature_names) hoặc (None, None) nếu lỗi
    """
    print("\n LOAD DỮ LIỆU ĐÃ PREPROCESS: ")
    
    train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
    feature_chosen_path = os.path.join(DATA_PROCESSED_DIR, 'feature_chosen.csv')
    
    if not os.path.exists(train_detail_path):
        return None, None
    
    try:
        train_detail = pd.read_csv(train_detail_path)
        
        # Convert Date to datetime
        if 'Date' in train_detail.columns:
            train_detail['Date'] = pd.to_datetime(train_detail['Date'])
        
        # Load feature names đã chọn
        if os.path.exists(feature_chosen_path):
            feature_chosen_df = pd.read_csv(feature_chosen_path)
            feature_names = feature_chosen_df['Feature'].tolist()
        else:
            # Nếu không có feature_chosen.csv, lấy tất cả features trừ Date và Weekly_Sales
            feature_names = [col for col in train_detail.columns 
                           if col not in ['Store', 'Dept', 'Date', 'Weekly_Sales']]
        
        return train_detail, feature_names
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None


def prepare_data_for_ml(train_detail, feature_names):
    """
    Chuẩn bị dữ liệu cho ML
    
    Args:
        train_detail: DataFrame chứa train data
        feature_names: Danh sách feature names
        
    Returns:
        tuple: (X, y) hoặc (None, None) nếu lỗi
    """
    print("\n CHUẨN BỊ DỮ LIỆU CHO ML: ")
    
    try:
        # Kiểm tra các features có trong train_detail không
        available_features = [f for f in feature_names if f in train_detail.columns]
        missing_features = [f for f in feature_names if f not in train_detail.columns]
        
        if len(available_features) == 0:
            return None, None
        
        # Tạo X và y
        X = train_detail[available_features].copy()
        y = train_detail['Weekly_Sales'].copy()
        
        # Xử lý missing values
        X = X.fillna(0)
        
        return X, y, available_features
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None


def create_weights_for_wmae(is_holiday):
    """
    Tạo weights cho WMAE
    
    Args:
        is_holiday: Series hoặc array chứa thông tin IsHoliday
        
    Returns:
        np.ndarray: Weights
    """
    # Convert to boolean/numeric
    if isinstance(is_holiday, pd.Series):
        is_holiday = is_holiday.values
    
    # Handle different formats: True/False, 1/0, 'True'/'False'
    if is_holiday.dtype == bool:
        weights = np.where(is_holiday == True, 5, 1)
    elif is_holiday.dtype == object:
        # String format
        weights = np.where((is_holiday == True) | (is_holiday == 'True') | (is_holiday == 1), 5, 1)
    else:
        # Numeric format
        weights = np.where(is_holiday == 1, 5, 1)
    
    return weights


def calculate_wmae(y_true, y_pred, weights):
    """
    Tính Weighted Mean Absolute Error
    
    Args:
        y_true: True values
        y_pred: Predicted values
        weights: Weights
        
    Returns:
        float: WMAE
    """
    return np.average(np.abs(y_true - y_pred), weights=weights)


def k_fold_cross_validation(X, y, model, model_name, k=5, train_detail=None):
    """
    Thực hiện K-Fold Cross-Validation
    
    Args:
        X: Features
        y: Target
        model: Model object (chưa fit)
        model_name: Tên model
        k: Số folds
        train_detail: DataFrame để lấy IsHoliday cho weights
        
    Returns:
        dict: Kết quả cross-validation
    """
    print(f"\n K-FOLD CROSS-VALIDATION: {model_name} (K={k})")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = {
        'mae': [],
        'rmse': [],
        'r2': [],
        'wmae': [],
        'train_time': []
    }
    
    # Lấy IsHoliday nếu có
    is_holiday = None
    if train_detail is not None and 'IsHoliday' in train_detail.columns:
        is_holiday = train_detail['IsHoliday'].values
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        
        # Split data
        X_train_fold = X.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_train_fold = y.iloc[train_idx]
        y_val_fold = y.iloc[val_idx]
        
        # Train model
        start_time = time.time()
        model.fit(X_train_fold, y_train_fold)
        train_time = time.time() - start_time
        
        # Predict
        y_pred_fold = model.predict(X_val_fold)
        
        # Calculate metrics
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
        r2 = r2_score(y_val_fold, y_pred_fold)
        
        # Calculate WMAE
        if is_holiday is not None:
            weights = create_weights_for_wmae(is_holiday[val_idx])
            wmae = calculate_wmae(y_val_fold, y_pred_fold, weights)
        else:
            wmae = mae  # Fallback to MAE if no IsHoliday
        
        fold_results['mae'].append(mae)
        fold_results['rmse'].append(rmse)
        fold_results['r2'].append(r2)
        fold_results['wmae'].append(wmae)
        fold_results['train_time'].append(train_time)
        
    
    # Tính trung bình và std
    results = {
        'model_name': model_name,
        'mean_mae': np.mean(fold_results['mae']),
        'std_mae': np.std(fold_results['mae']),
        'mean_rmse': np.mean(fold_results['rmse']),
        'std_rmse': np.std(fold_results['rmse']),
        'mean_r2': np.mean(fold_results['r2']),
        'std_r2': np.std(fold_results['r2']),
        'mean_wmae': np.mean(fold_results['wmae']),
        'std_wmae': np.std(fold_results['wmae']),
        'mean_train_time': np.mean(fold_results['train_time']),
        'fold_results': fold_results
    }
    
    
    return results


def train_final_model(X, y, model, model_name, train_detail=None):
    """
    Train model cuối cùng trên toàn bộ dữ liệu
    
    Args:
        X: Features
        y: Target
        model: Model object
        model_name: Tên model
        train_detail: DataFrame để lấy IsHoliday
        
    Returns:
        tuple: (trained_model, metrics)
    """
    print(f"\n TRAIN MODEL CUỐI CÙNG: {model_name}")
    
    start_time = time.time()
    model.fit(X, y)
    train_time = time.time() - start_time
    
    # Predict trên toàn bộ dữ liệu
    y_pred = model.predict(X)
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    # Calculate WMAE
    if train_detail is not None and 'IsHoliday' in train_detail.columns:
        weights = create_weights_for_wmae(train_detail['IsHoliday'])
        wmae = calculate_wmae(y, y_pred, weights)
    else:
        wmae = mae
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'wmae': wmae,
        'train_time': train_time
    }
    

    return model, metrics


def create_submission_from_best_model(best_model, best_model_name, feature_names):
    """
    Tạo submission từ model tốt nhất
    
    Args:
        best_model: Model đã train
        best_model_name: Tên model
        feature_names: Danh sách feature names
    """
    print(f"\n TẠO SUBMISSION TỪ MODEL TỐT NHẤT")
    
    # Load test_detail
    test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
    if not os.path.exists(test_detail_path):
        return None
    
    try:
        test_detail = pd.read_csv(test_detail_path)
        if 'Date' in test_detail.columns:
            test_detail['Date'] = pd.to_datetime(test_detail['Date'])
        
        # Prepare test data
        available_features = [f for f in feature_names if f in test_detail.columns]
        X_test = test_detail[available_features].copy()
        X_test = X_test.fillna(0)
        
        # Predict
        predictions = best_model.predict(X_test)
        predictions = np.maximum(predictions, 0)  # Đảm bảo không âm
        
        # Tạo Id
        date_str = test_detail['Date'].dt.strftime('%Y-%m-%d')
        ids = (test_detail['Store'].astype(str) + '_' + 
               test_detail['Dept'].astype(str) + '_' + 
               date_str)
        
        # Tạo submission
        submission = pd.DataFrame({
            'Id': ids,
            'Weekly_Sales': np.round(predictions, 2)
        })
        
        submission = submission.sort_values('Id').reset_index(drop=True)
        
        # Lưu file
        output_path = os.path.join(OUTPUT_DIR, f'submission_{best_model_name.lower().replace(" ", "_")}.csv')
        submission.to_csv(output_path, index=False)
        
        return submission
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def main():
    """Hàm chính"""
    print("\n K-FOLD CROSS-VALIDATION: RANDOM FOREST vs XGBOOST")
    
    # 1. Load data
    train_detail, feature_names = load_preprocessed_data()
    if train_detail is None:
        return None
    
    # 2. Prepare data
    X, y, available_features = prepare_data_for_ml(train_detail, feature_names)
    if X is None:
        return None
    
    # 3. K-Fold Cross-Validation
    k = 5
    results = {}
    
    # Random Forest
    print("\n MODEL 1: RANDOM FOREST")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_results = k_fold_cross_validation(
        X, y, rf_model, "Random Forest", k=k, train_detail=train_detail
    )
    results['Random Forest'] = rf_results
    
    # XGBoost
    print("\n MODEL 2: XGBOOST")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_results = k_fold_cross_validation(
        X, y, xgb_model, "XGBoost", k=k, train_detail=train_detail
    )
    results['XGBoost'] = xgb_results
    
    # 4. So sánh và chọn model tốt nhất
    print("\n SO SÁNH VÀ CHỌN MODEL TỐT NHẤT")
    
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Mean_WMAE': model_results['mean_wmae'],
            'Std_WMAE': model_results['std_wmae'],
            'Mean_MAE': model_results['mean_mae'],
            'Std_MAE': model_results['std_mae'],
            'Mean_RMSE': model_results['mean_rmse'],
            'Std_RMSE': model_results['std_rmse'],
            'Mean_R2': model_results['mean_r2'],
            'Std_R2': model_results['std_r2'],
            'Mean_Train_Time': model_results['mean_train_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_WMAE')
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Chọn model tốt nhất (WMAE thấp nhất)
    best_model_name = comparison_df.iloc[0]['Model']
    best_wmae = comparison_df.iloc[0]['Mean_WMAE']
    
    # 5. Train model tốt nhất trên toàn bộ dữ liệu
    print("\n TRAIN MODEL TỐT NHẤT TRÊN TOÀN BỘ DỮ LIỆU")
    
    if best_model_name == "Random Forest":
        best_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    else:  # XGBoost
        best_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    best_model_trained, final_metrics = train_final_model(
        X, y, best_model, best_model_name, train_detail=train_detail
    )
    
    # 6. Tạo submission
    submission = create_submission_from_best_model(
        best_model_trained, best_model_name, available_features
    )
    
    # 8. Lưu kết quả so sánh
    comparison_path = os.path.join(OUTPUT_DIR, 'kfold_validation_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nĐã lưu kết quả so sánh: {comparison_path}")
    
    print("\n HOÀN THÀNH K-FOLD VALIDATION!")
    
    return {
        'results': results,
        'comparison': comparison_df,
        'best_model': best_model_trained,
        'best_model_name': best_model_name,
        'submission': submission
    }


if __name__ == "__main__":
    results = main()

