import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_PROCESSED_DIR

def load_preprocessed_data(train_detail_path=None, test_detail_path=None):
    if train_detail_path is None:
        train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
    if test_detail_path is None:
        test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')

    print("=== LOAD PREPROCESSED DATA ===")
    
    try:
        train_detail = pd.read_csv(train_detail_path)
        test_detail = pd.read_csv(test_detail_path)
        
        # Convert Date column back to datetime
        if 'Date' in train_detail.columns:
            train_detail['Date'] = pd.to_datetime(train_detail['Date'])
        if 'Date' in test_detail.columns:
            test_detail['Date'] = pd.to_datetime(test_detail['Date'])
        
        return train_detail, test_detail
    
    except FileNotFoundError as e:
        print(f"Không tìm thấy file: {e}")
        return None, None

def handle_missing_values(df, strategy='forward_fill'):
    
    """
    -   Nếu là forward fill, sắp xếp theo thời gian và forward fill theo group sau đó fill các dữ liệu bị thiếu bằng giá trị hợp lệ gần nhất trước đó
    -   Nếu là backward fill, sắp xếp theo thời gian và backward fill theo group sau đó fill các dữ liệu bị thiếu bằng giá trị hợp lệ gần nhất sau đó
    -   Nếu là mean, fill theo mean của group nếu có Store và Dept sau đó fill các dữ liệu bị thiếu bằng giá trị mean của group
    -   Nếu là median, fill theo median của group nếu có Store và Dept sau đó fill các dữ liệu bị thiếu bằng giá trị median của group
    """

    print(f"\n=== XỬ LÝ MISSING VALUES (Strategy: {strategy}) ===")

    
    df_clean = df.copy()
    
    # Analyze missing values
    missing_before = df_clean.isnull().sum().sum()
    print(f"Missing values: {missing_before}")
    
    # Kiểm tra các cột cần thiết
    has_store = 'Store' in df_clean.columns
    has_dept = 'Dept' in df_clean.columns
    has_date = 'Date' in df_clean.columns
    
    # Xử lý theo strategy
    if strategy == 'forward_fill':
        # Forward fill cho time series
        if has_store and has_dept and has_date:
            # Sắp xếp theo thời gian
            df_clean = df_clean.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
            # Forward fill theo group
            df_clean = df_clean.groupby(['Store', 'Dept'], group_keys=False).apply(
                lambda x: x.ffill().bfill()
            )
        else:
            # Nếu không có các cột groupby, chỉ sort theo Date nếu có
            if has_date:
                df_clean = df_clean.sort_values(by='Date').reset_index(drop=True)
                df_clean = df_clean.ffill().bfill()
            else:
                df_clean = df_clean.ffill().bfill()
    
    elif strategy == 'backward_fill':
        if has_store and has_dept and has_date:
            df_clean = df_clean.sort_values(by=['Store', 'Dept', 'Date']).reset_index(drop=True)
            df_clean = df_clean.groupby(['Store', 'Dept'], group_keys=False).apply(
                lambda x: x.bfill().ffill()
            )
        else:
            if has_date:
                df_clean = df_clean.sort_values(by='Date').reset_index(drop=True)
                df_clean = df_clean.bfill().ffill()
            else:
                df_clean = df_clean.bfill().ffill()
    
    elif strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Fill theo mean của group nếu có Store và Dept
                if has_store and has_dept:
                    df_clean[col] = df_clean.groupby(['Store', 'Dept'])[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                    # Fill remaining với overall mean
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                if has_store and has_dept:
                    df_clean[col] = df_clean.groupby(['Store', 'Dept'])[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    elif strategy == 'zero':
        df_clean = df_clean.fillna(0)
    
    # Kiểm tra sau xử lý
    missing_after = df_clean.isnull().sum().sum()
    print(f"Missing values sau xử lý: {missing_after}")
    print(f"✓ Đã xử lý {missing_before - missing_after} missing values")
    
    return df_clean

def prepare_ml_data(train_detail, exclude_cols=None):

    print("\n=== CHUẨN BỊ DỮ LIỆU CHO ML ===")
    
    """
        - Loại bỏ các cột khỏi feature_cols nếu chúng không cần thiết
        - Loại bỏ các cột có quá nhiều missing values (>50%)
        - Tạo X và y
        - Xử lý missing values còn lại
        - Trả về X, y và feature_cols
    """
    if exclude_cols is None:
        # Các cột không cần thiết
        exclude_cols = ['Date', 'Weekly_Sales'] # date và weekly_sales là cột target
    
    # Loại bỏ các cột không cần thiết
    feature_cols = [col for col in train_detail.columns if col not in exclude_cols]
    
    # Loại bỏ các cột có quá nhiều missing values (>50%)
    feature_cols = [col for col in feature_cols 
                   if train_detail[col].isnull().sum() < len(train_detail) * 0.5]
    
    print(f"Tổng số features: {len(feature_cols)}")
    print(f"Features bị loại bỏ: {len(train_detail.columns) - len(feature_cols) - len(exclude_cols)}")
    
    # Tạo X và y
    X = train_detail[feature_cols].copy() # X là các cột features
    y = train_detail['Weekly_Sales'].copy()
    
    # Xử lý missing values còn lại
    X = X.fillna(0)
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y, feature_cols

def time_series_split_data(X, y, test_size=0.2, gap=1):
    print(f"\n=== CHIA DỮ LIỆU THEO THỜI GIAN ===")
    print(f"Test size: {test_size * 100}%") # test_size là tỷ lệ test set (giữ lại tỷ lệ test set là 20%)
    print(f"Gap: {gap} weeks") # gap là số tuần gap giữa train và test
    
    # Đảm bảo dữ liệu được sắp xếp theo thời gian
    sorted_indices = X.index.sort_values()
    X_sorted = X.loc[sorted_indices]
    y_sorted = y.loc[sorted_indices]
    
    # Tính split index
    split_idx = int(len(X_sorted) * (1 - test_size))
    
    # Tạo gap
    gap_start = split_idx - gap # gap_start là index của dữ liệu train
    gap_end = split_idx + gap # gap_end là index của dữ liệu test
    
    """
        Train Set: Dữ liệu <= gap_start
        Test Set: Dữ liệu >= gap_end
        Gap: Dữ liệu giữa gap_start và gap_end 
            (giữ lại gap là 1 tuần để tránh data leakage)
    """
    
    # Chia dữ liệu
    X_train = X_sorted.iloc[:gap_start]
    X_test = X_sorted.iloc[gap_end:]
    y_train = y_sorted.iloc[:gap_start]
    y_test = y_sorted.iloc[gap_end:]
    
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Gap size: {gap_end - gap_start}")
    
    return X_train, X_test, y_train, y_test

def create_evaluation_metrics():

    print("\n=== TẠO EVALUATION METRICS CHO REGRESSION ===")
    
    def mae(y_true, y_pred):
        """Mean Absolute Error"""
        # Trung bình độ lớn giá trị tuyệt đối của sai số giữa giá trị thực và giá trị dự đoán
        return mean_absolute_error(y_true, y_pred)
    
    def mse(y_true, y_pred):
        """Mean Squared Error"""
        # Trung bình bình phương sai số giữa giá trị thực và giá trị dự đoán
        return mean_squared_error(y_true, y_pred)
    
    def rmse(y_true, y_pred):
        """Root Mean Squared Error"""
        # Căn bậc hai của trung bình bình phương sai số giữa giá trị thực và giá trị dự đoán
        # RMSE là thước đo độ phù hợp của mô hình (càng nhỏ càng tốt)
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def r2(y_true, y_pred):
        """R-squared"""
        # Hệ số xác định R^2
        # R^2 là thước đo độ phù hợp của mô hình (càng gần 1 càng tốt)
        return r2_score(y_true, y_pred)
    
    def wmae(y_true, y_pred, weights=None):
        """
        Weighted Mean Absolute Error
        weights = 5 cho holiday weeks, 1 cho normal weeks
        WMAE là trung bình độ lớn giá trị tuyệt đối của sai số giữa giá trị thực và giá trị dự đoán sau khi nhân với weights, với weights là 5 cho holiday weeks và 1 cho normal weeks
        """
        if weights is None:
            weights = np.ones(len(y_true))
        return np.average(np.abs(y_true - y_pred), weights=weights)
    
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error"""
        # Trung bình độ lớn phần trăm sai số trung bình giữa giá trị thực và giá trị dự đoán
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'wmae': wmae,
        'mape': mape
    }
    
    print("Đã tạo các hàm đánh giá metrics")
    
    return metrics

def create_weights_for_wmae(df, is_holiday_col='IsHoliday'):
    """
    Việc sử dụng trọng số là một kỹ thuật quan trọng trong Machine Learning khi bạn có các điểm dữ liệu không đồng đều về mức độ quan trọng:
    -   Tuần Lễ (Trọng số 5): Doanh số bán hàng trong các dịp lễ lớn (như Tạ ơn, Giáng sinh) thường tăng đột biến
        và rất khó dự đoán chính xác. Việc dự đoán sai trong tuần lễ có thể gây ra tổn thất lớn hơn nhiều (do thiếu hàng hoặc dư thừa hàng). 
        Do đó, bằng cách áp dụng trọng số gấp 5 lần, buộc mô hình phải chú trọng và cố gắng giảm thiểu lỗi trong các tuần này.
    -   Tuần thường (Trọng số 1): Sai số trong các tuần bình thường được tính toán với trọng số cơ bản là 1.
    -   Mảng weights này sau đó được truyền vào hàm wmae() để tính toán chỉ số đánh giá cuối cùng.
    """
    
    if is_holiday_col in df.columns:
        weights = np.where(df[is_holiday_col] == 1, 5, 1)
    else:
        weights = np.ones(len(df))
        print("Không tìm thấy cột IsHoliday, sử dụng weights = 1 cho tất cả")
    
    return weights

def feature_selection(X_train, y_train, method='correlation', top_k=50):

    print(f"\n=== CHỌN FEATURES (Method: {method}, Top K: {top_k}) ===")
    
    if method == 'correlation':
        # Chọn features có correlation cao với target (correlation là hệ số tương quan giữa các features và target)
        """
        X_train.corrwith(y_train): Tính hệ số tương quan Pearson giữa mỗi cột đặc trưng trong X_train và cột mục tiêu y_train.
        .abs(): Lấy giá trị tuyệt đối của tương quan. Quan tâm đến độ mạnh của mối quan hệ, không quan tâm nó là tương quan dương hay âm.
        .sort_values(ascending=False): Sắp xếp các giá trị tương quan theo thứ tự giảm dần (từ mạnh nhất đến yếu nhất).
        .head(top_k): Chọn top_k đặc trưng có mối tương quan mạnh nhất với Weekly_Sales.
        """
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        selected_features = correlations.head(top_k).index.tolist()
        print(f"Đã chọn {len(selected_features)} features dựa trên correlation")
    
    elif method == 'variance':
        # Chọn features có variance cao (variance là phương sai của các features)
        """
            X_train.var(): Tính phương sai của mỗi cột đặc trưng trong X_train.
            .sort_values(ascending=False): Sắp xếp các giá trị phương sai theo thứ tự giảm dần (từ mạnh nhất đến yếu nhất).
            .head(top_k): Chọn top_k đặc trưng có phương sai mạnh nhất với Weekly_Sales.
        """
        variances = X_train.var().sort_values(ascending=False)
        selected_features = variances.head(top_k).index.tolist()
        print(f"Đã chọn {len(selected_features)} features dựa trên variance")
    
    elif method == 'all':
        # Giữ tất cả features (không chọn features)
        selected_features = X_train.columns.tolist()
        print(f"Giữ tất cả {len(selected_features)} features")
    
    return selected_features

def scale_features(X_train, X_test, method='standard'):
    print(f"\n=== SCALE FEATURES (Method: {method}) ===")
    
    if method == 'standard':
        """
            Chuẩn hóa Z-Score: Biến đổi dữ liệu sao cho có trung bình μ bằng 0 
            và độ lệch chuẩn σ bằng 1. Thích hợp khi dữ liệu có phân phối xấp xỉ chuẩn (Gaussian).
        """
        scaler = StandardScaler()
    elif method == 'robust':
        """
            Robust Scaler: Biến đổi dữ liệu sao cho có trung vị bằng 0 
            và IQR bằng 1. Thích hợp khi dữ liệu có phân phối không xấp xỉ chuẩn (Gaussian).
        """
        scaler = RobustScaler()
    elif method == 'none':
        print("✓ Không scale features")
        return X_train, X_test, None
    else:
        scaler = StandardScaler()
    
    # Fit và transform
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print(f"Đã scale features bằng {method}")
    
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(y_true, y_pred, model_name="Model", metrics=None, weights=None):
    if metrics is None:
        metrics = create_evaluation_metrics()
    
    results = {}
    for metric_name, metric_func in metrics.items():
        if metric_name == 'wmae' and weights is not None:
            results[metric_name] = metric_func(y_true, y_pred, weights)
        else:
            results[metric_name] = metric_func(y_true, y_pred)
    
    print(f"\n=== KẾT QUẢ {model_name.upper()} ===")
    print(f"MAE: {results['mae']:.2f}")
    print(f"RMSE: {results['rmse']:.2f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"WMAE: {results['wmae']:.2f}")
    print(f"MAPE: {results['mape']:.2f}%")
    
    return results

def save_prepared_data(X_train, X_test, y_train, y_test, feature_names, output_dir=None):
    if output_dir is None:
        from config import DATA_PROCESSED_DIR
        output_dir = DATA_PROCESSED_DIR

    print(f"\n=== LƯU DỮ LIỆU ĐÃ CHUẨN BỊ ===")
    
    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    # Lưu feature names
    pd.Series(feature_names).to_csv(f'{output_dir}/feature_names.csv', index=False)
    
    print(f"Đã lưu dữ liệu vào {output_dir}/")

def main():
    
    # Load data
    train_detail, test_detail = load_preprocessed_data()
    
    if train_detail is None or test_detail is None:
        print("Thiếu dữ liệu train_detail hoặc test_detail")
        return None
    
    # Xử lý missing values
    train_detail = handle_missing_values(train_detail, strategy='forward_fill')
    
    # Chuẩn bị dữ liệu cho ML
    X, y, feature_names = prepare_ml_data(train_detail)
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = time_series_split_data(X, y, test_size=0.2, gap=1)
    
    # Feature selection
    selected_features = feature_selection(X_train, y_train, method='correlation', top_k=50)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print(f"Đã chọn {len(selected_features)} features")
    
    # Scale features (optional - có thể bỏ qua nếu dùng tree-based models)
    print("Bỏ qua feature scaling (không cần cho tree-based models như XGBoost, Random Forest)")
    # X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')
    X_train_scaled, X_test_scaled = X_train, X_test
    
    # Tạo evaluation metrics
    print("\n" + "="*80)
    metrics = create_evaluation_metrics()
    
    # Tạo weights cho WMAE
    # Lấy IsHoliday từ train_detail cho test set
    test_indices = X_test.index
    test_holiday = train_detail.loc[test_indices, 'IsHoliday'] if 'IsHoliday' in train_detail.columns else None
    if test_holiday is not None:
        weights = create_weights_for_wmae(pd.DataFrame({'IsHoliday': test_holiday}))
    else:
        weights = np.ones(len(y_test))
        print("Không tìm thấy IsHoliday, sử dụng weights = 1 cho tất cả")
    
    # Lưu dữ liệu
    save_prepared_data(X_train_scaled, X_test_scaled, y_train, y_test, selected_features)
    
    # Lưu weights
    from config import DATA_PROCESSED_DIR
    weights_path = os.path.join(DATA_PROCESSED_DIR, 'weights.npy')
    np.save(weights_path, weights)
    print(f"✓ Đã lưu weights.npy tại {weights_path}")
    
    print("Hoàn thành data preparation!")
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': selected_features,
        'metrics': metrics,
        'weights': weights
    }

if __name__ == "__main__":
    main()
