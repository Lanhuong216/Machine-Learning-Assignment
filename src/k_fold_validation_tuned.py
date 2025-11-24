"""
Train models với best parameters từ tuned_models_best_params.csv
Sử dụng K-Fold Cross-Validation và xuất submission
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import ast
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

warnings.filterwarnings('ignore')

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR, OUTPUT_DIR, OUTPUT_REPORTS_DIR


def load_best_params(csv_path=None):
    """
    Load best parameters từ tuned_models_best_params.csv
    
    Args:
        csv_path: Đường dẫn đến file CSV. Nếu None thì dùng default path
        
    Returns:
        dict: Dictionary chứa best parameters cho mỗi model
    """
    if csv_path is None:
        csv_path = os.path.join(OUTPUT_REPORTS_DIR, 'tuned_models_best_params.csv')
    
    print("="*80)
    print("LOAD BEST PARAMETERS")
    print("="*80)
    
    if not os.path.exists(csv_path):
        print(f"❌ Không tìm thấy file: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Đã load {len(df)} models từ {csv_path}")
        
        best_params = {}
        for _, row in df.iterrows():
            model_name = row['Model']
            params_str = row['Best_Params']
            
            # Parse parameters từ string sang dict
            try:
                # Sử dụng ast.literal_eval để parse string thành dict
                params_dict = ast.literal_eval(params_str)
                best_params[model_name] = params_dict
                print(f"✓ {model_name}: {len(params_dict)} parameters")
            except Exception as e:
                print(f"⚠️ Không thể parse parameters cho {model_name}: {e}")
                continue
        
        return best_params
    
    except Exception as e:
        print(f"❌ Lỗi khi load best parameters: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_preprocessed_data():
    """
    Load train_detail đã được preprocess
    
    Returns:
        tuple: (train_detail, feature_names) hoặc (None, None) nếu lỗi
    """
    print("\n" + "="*80)
    print("LOAD DỮ LIỆU ĐÃ PREPROCESS")
    print("="*80)
    
    train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
    feature_chosen_path = os.path.join(DATA_PROCESSED_DIR, 'feature_chosen.csv')
    
    if not os.path.exists(train_detail_path):
        print(f"❌ Không tìm thấy file: {train_detail_path}")
        print("   Vui lòng chạy preprocessing.py trước")
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
                           if col not in ['Date', 'Weekly_Sales']]
            print(f"⚠️ Không tìm thấy feature_chosen.csv, sử dụng tất cả {len(feature_names)} features")
        
        print(f"✓ Đã load train_detail: {train_detail.shape}")
        print(f"✓ Số features: {len(feature_names)}")
        print(f"  Features: {feature_names[:5]}..." if len(feature_names) > 5 else f"  Features: {feature_names}")
        
        return train_detail, feature_names
    
    except Exception as e:
        print(f"❌ Lỗi khi load dữ liệu: {e}")
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
        tuple: (X, y, available_features) hoặc (None, None, None) nếu lỗi
    """
    print("\n" + "="*80)
    print("CHUẨN BỊ DỮ LIỆU CHO ML")
    print("="*80)
    
    try:
        # Kiểm tra các features có trong train_detail không
        available_features = [f for f in feature_names if f in train_detail.columns]
        missing_features = [f for f in feature_names if f not in train_detail.columns]
        
        if missing_features:
            print(f"⚠️ Thiếu {len(missing_features)} features: {missing_features[:3]}...")
        
        if len(available_features) == 0:
            print("❌ Không có features nào khả dụng")
            return None, None, None
        
        # Tạo X và y
        X = train_detail[available_features].copy()
        y = train_detail['Weekly_Sales'].copy()
        
        # Xử lý missing values
        X = X.fillna(0)
        
        print(f"✓ X shape: {X.shape}")
        print(f"✓ y shape: {y.shape}")
        print(f"✓ Sử dụng {len(available_features)} features")
        
        return X, y, available_features
    
    except Exception as e:
        print(f"❌ Lỗi khi chuẩn bị dữ liệu: {e}")
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
    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {model_name} (K={k})")
    print(f"{'='*80}")
    
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
        print(f"\n--- Fold {fold}/{k} ---")
        
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
        
        print(f"  MAE: {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  WMAE: {wmae:.2f}")
        print(f"  Train time: {train_time:.2f}s")
    
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
    
    print(f"\n--- Kết quả trung bình ({k} folds) ---")
    print(f"  MAE: {results['mean_mae']:.2f} ± {results['std_mae']:.2f}")
    print(f"  RMSE: {results['mean_rmse']:.2f} ± {results['std_rmse']:.2f}")
    print(f"  R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    print(f"  WMAE: {results['mean_wmae']:.2f} ± {results['std_wmae']:.2f}")
    print(f"  Train time: {results['mean_train_time']:.2f}s")
    
    return results


def create_model_from_params(model_name, params):
    """
    Tạo model từ parameters
    
    Args:
        model_name: Tên model ('Random Forest' hoặc 'XGBoost')
        params: Dictionary chứa parameters
        
    Returns:
        Model object
    """
    # Set default parameters
    default_params = {
        'random_state': 42,
        'n_jobs': -1
    }
    
    if model_name == 'Random Forest':
        # Merge params với defaults
        model_params = {**default_params, **params}
        # XGBoost không có n_jobs, chỉ Random Forest có
        model = RandomForestRegressor(**model_params)
    elif model_name == 'XGBoost':
        # XGBoost không có n_jobs, có verbosity
        model_params = {**default_params, **params}
        model_params.pop('n_jobs', None)  # Remove n_jobs nếu có
        model_params['verbosity'] = 0  # XGBoost dùng verbosity
        model = xgb.XGBRegressor(**model_params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


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
    print(f"\n{'='*80}")
    print(f"TRAIN MODEL CUỐI CÙNG: {model_name}")
    print(f"{'='*80}")
    
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
    
    print(f"✓ MAE: {mae:.2f}")
    print(f"✓ RMSE: {rmse:.2f}")
    print(f"✓ R²: {r2:.4f}")
    print(f"✓ WMAE: {wmae:.2f}")
    print(f"✓ Train time: {train_time:.2f}s")
    
    return model, metrics


def create_submission_from_model(model, model_name, feature_names):
    """
    Tạo submission từ model
    
    Args:
        model: Model đã train
        model_name: Tên model
        feature_names: Danh sách feature names
    """
    print(f"\n{'='*80}")
    print("TẠO SUBMISSION TỪ MODEL")
    print(f"{'='*80}")
    
    # Load test_detail
    test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
    if not os.path.exists(test_detail_path):
        print(f"❌ Không tìm thấy file: {test_detail_path}")
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
        print("Đang dự đoán...")
        predictions = model.predict(X_test)
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
        output_path = os.path.join(OUTPUT_DIR, f'submission_{model_name.lower().replace(" ", "_")}_best_params.csv')
        submission.to_csv(output_path, index=False)
        
        print(f"✓ Đã tạo submission: {output_path}")
        print(f"  - Số dòng: {len(submission)}")
        print(f"  - Min: {predictions.min():.2f}")
        print(f"  - Max: {predictions.max():.2f}")
        print(f"  - Mean: {np.mean(predictions):.2f}")
        
        return submission
    
    except Exception as e:
        print(f"❌ Lỗi khi tạo submission: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Hàm chính"""
    print("="*80)
    print("TRAIN VỚI BEST PARAMETERS VÀ K-FOLD VALIDATION")
    print("="*80)
    
    # 1. Load best parameters
    best_params = load_best_params()
    if best_params is None or len(best_params) == 0:
        print("❌ Không thể tiếp tục do thiếu best parameters")
        return None
    
    # 2. Load data
    train_detail, feature_names = load_preprocessed_data()
    if train_detail is None:
        return None
    
    # 3. Prepare data
    X, y, available_features = prepare_data_for_ml(train_detail, feature_names)
    if X is None:
        return None
    
    # 4. K-Fold Cross-Validation với best parameters
    k = 5
    results = {}
    
    for model_name, params in best_params.items():
        print("\n" + "="*80)
        print(f"MODEL: {model_name}")
        print("="*80)
        print(f"Parameters: {params}")
        
        # Tạo model với best parameters
        model = create_model_from_params(model_name, params)
        
        # K-Fold Cross-Validation
        cv_results = k_fold_cross_validation(
            X, y, model, model_name, k=k, train_detail=train_detail
        )
        results[model_name] = cv_results
    
    # 5. So sánh và chọn model tốt nhất
    print("\n" + "="*80)
    print("SO SÁNH VÀ CHỌN MODEL TỐT NHẤT")
    print("="*80)
    
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
    
    print(f"\n🏆 MODEL TỐT NHẤT: {best_model_name}")
    print(f"   WMAE: {best_wmae:.2f} ± {comparison_df.iloc[0]['Std_WMAE']:.2f}")
    
    # 6. Train model tốt nhất trên toàn bộ dữ liệu với best parameters
    print("\n" + "="*80)
    print("TRAIN MODEL TỐT NHẤT TRÊN TOÀN BỘ DỮ LIỆU")
    print("="*80)
    
    best_params_dict = best_params[best_model_name]
    best_model = create_model_from_params(best_model_name, best_params_dict)
    
    best_model_trained, final_metrics = train_final_model(
        X, y, best_model, best_model_name, train_detail=train_detail
    )
    
    # 7. Tạo submission
    submission = create_submission_from_model(
        best_model_trained, best_model_name, available_features
    )
    
    # 9. Lưu kết quả so sánh
    comparison_path = os.path.join(OUTPUT_DIR, 'best_params_kfold_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Đã lưu kết quả so sánh: {comparison_path}")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH TRAIN VỚI BEST PARAMETERS!")
    print("="*80)
    print(f"\nModel tốt nhất: {best_model_name}")
    print(f"Submission file: output/submission_{best_model_name.lower().replace(' ', '_')}_best_params.csv")
    
    return {
        'results': results,
        'comparison': comparison_df,
        'best_model': best_model_trained,
        'best_model_name': best_model_name,
        'submission': submission
    }


if __name__ == "__main__":
    results = main()
