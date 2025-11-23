"""
Script để tạo file submission cuối cùng dựa trên model tốt nhất
Format output giống với walmart-sampleSubmission.csv
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUT_DIR


def load_best_model():
    """
    Load model tốt nhất từ best_model.pkl
    
    Returns:
        model: Model object hoặc None nếu lỗi
    """
    print("="*80)
    print("LOAD MODEL TỐT NHẤT")
    print("="*80)
    
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    
    if not os.path.exists(best_model_path):
        print(f"❌ Không tìm thấy file: {best_model_path}")
        print("   Vui lòng chạy model_evaluation_analysis.py trước để tạo best_model.pkl")
        return None
    
    try:
        with open(best_model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"✓ Đã load model: {type(model).__name__}")
        print(f"  Từ file: {best_model_path}")
        return model
    
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None


def load_feature_names():
    """
    Load danh sách feature names đã được chọn
    
    Returns:
        list: Danh sách feature names hoặc None nếu lỗi
    """
    feature_names_path = os.path.join(DATA_PROCESSED_DIR, 'feature_names.csv')
    
    if not os.path.exists(feature_names_path):
        print(f"❌ Không tìm thấy file: {feature_names_path}")
        return None
    
    try:
        feature_names = pd.read_csv(feature_names_path).iloc[:, 0].tolist()
        print(f"✓ Đã load {len(feature_names)} features")
        return feature_names
    
    except Exception as e:
        print(f"❌ Lỗi khi load feature names: {e}")
        return None


def prepare_test_data(test_detail, feature_names):
    """
    Chuẩn bị test data với cùng features như khi train
    
    Args:
        test_detail (pd.DataFrame): Test data đã preprocess
        feature_names (list): Danh sách feature names cần sử dụng
        
    Returns:
        pd.DataFrame: Test data đã được prepare với đúng features
    """
    print("\n" + "="*80)
    print("CHUẨN BỊ TEST DATA")
    print("="*80)
    
    # Kiểm tra các features có trong test_detail không
    missing_features = [f for f in feature_names if f not in test_detail.columns]
    if missing_features:
        print(f"⚠️ Cảnh báo: Thiếu {len(missing_features)} features trong test_detail:")
        for f in missing_features[:5]:
            print(f"  - {f}")
        if len(missing_features) > 5:
            print(f"  ... và {len(missing_features) - 5} features khác")
    
    # Chọn các features có sẵn
    available_features = [f for f in feature_names if f in test_detail.columns]
    print(f"✓ Sử dụng {len(available_features)}/{len(feature_names)} features")
    
    # Tạo X_test với đúng thứ tự features
    X_test = test_detail[available_features].copy()
    
    # Xử lý missing values (giống như khi train)
    X_test = X_test.fillna(0)
    
    # Đảm bảo thứ tự cột giống với feature_names
    # Nếu thiếu features, thêm cột 0
    for feature in feature_names:
        if feature not in X_test.columns:
            X_test[feature] = 0
            print(f"  ⚠️ Thêm feature {feature} với giá trị 0 (không có trong test data)")
    
    # Sắp xếp lại theo thứ tự feature_names
    X_test = X_test[feature_names]
    
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ Features: {list(X_test.columns)[:5]}..." if len(X_test.columns) > 5 else f"✓ Features: {list(X_test.columns)}")
    
    return X_test


def create_submission_ids(test_detail):
    """
    Tạo Id cho submission theo format: Store_Dept_Date
    
    Args:
        test_detail (pd.DataFrame): Test data với các cột Store, Dept, Date
        
    Returns:
        pd.Series: Series chứa các Id
    """
    # Format Date thành YYYY-MM-DD
    if test_detail['Date'].dtype == 'object':
        test_detail['Date'] = pd.to_datetime(test_detail['Date'])
    
    date_str = test_detail['Date'].dt.strftime('%Y-%m-%d')
    
    # Tạo Id: Store_Dept_Date
    ids = (test_detail['Store'].astype(str) + '_' + 
           test_detail['Dept'].astype(str) + '_' + 
           date_str)
    
    return ids


def make_predictions(model, X_test):
    """
    Dự đoán Weekly_Sales cho test data
    
    Args:
        model: Model đã train
        X_test (pd.DataFrame): Test features
        
    Returns:
        np.ndarray: Predictions
    """
    print("\n" + "="*80)
    print("DỰ ĐOÁN")
    print("="*80)
    
    try:
        predictions = model.predict(X_test)
        
        # Đảm bảo predictions không âm (Weekly_Sales không thể âm)
        predictions = np.maximum(predictions, 0)
        
        print(f"✓ Đã dự đoán {len(predictions)} samples")
        print(f"  - Min: {np.min(predictions):.2f}")
        print(f"  - Max: {np.max(predictions):.2f}")
        print(f"  - Mean: {np.mean(predictions):.2f}")
        print(f"  - Median: {np.median(predictions):.2f}")
        
        return predictions
    
    except Exception as e:
        print(f"❌ Lỗi khi dự đoán: {e}")
        return None


def create_submission_file(test_detail, predictions, output_path=None):
    """
    Tạo file submission với format giống walmart-sampleSubmission.csv
    
    Args:
        test_detail (pd.DataFrame): Test data
        predictions (np.ndarray): Predictions
        output_path (str): Đường dẫn file output
        
    Returns:
        pd.DataFrame: Submission dataframe
    """
    print("\n" + "="*80)
    print("TẠO FILE SUBMISSION")
    print("="*80)
    
    # Tạo Id
    ids = create_submission_ids(test_detail)
    
    # Tạo submission dataframe
    # Làm tròn Weekly_Sales đến 2 chữ số thập phân để format đẹp hơn
    submission = pd.DataFrame({
        'Id': ids,
        'Weekly_Sales': np.round(predictions, 2)
    })
    
    # Sắp xếp theo Id để đảm bảo thứ tự giống sample submission
    submission = submission.sort_values('Id').reset_index(drop=True)
    
    # Lưu file
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    
    submission.to_csv(output_path, index=False)
    
    print(f"✓ Đã tạo file submission: {output_path}")
    print(f"  - Số dòng: {len(submission)}")
    print(f"  - Columns: {list(submission.columns)}")
    print(f"\n  Preview (5 dòng đầu):")
    print(submission.head().to_string(index=False))
    print(f"\n  Preview (5 dòng cuối):")
    print(submission.tail().to_string(index=False))
    
    return submission


def verify_submission_format(submission_path, sample_path=None):
    """
    Kiểm tra format của submission file có đúng không
    
    Args:
        submission_path (str): Đường dẫn file submission
        sample_path (str): Đường dẫn file sample submission để so sánh
    """
    print("\n" + "="*80)
    print("KIỂM TRA FORMAT SUBMISSION")
    print("="*80)
    
    try:
        submission = pd.read_csv(submission_path)
        
        # Kiểm tra columns
        expected_cols = ['Id', 'Weekly_Sales']
        if list(submission.columns) != expected_cols:
            print(f"❌ Columns không đúng!")
            print(f"  Expected: {expected_cols}")
            print(f"  Got: {list(submission.columns)}")
            return False
        
        print("✓ Columns đúng")
        
        # Kiểm tra Id format
        sample_ids = submission['Id'].head(10)
        for idx, id_val in enumerate(sample_ids):
            parts = id_val.split('_')
            if len(parts) < 3:
                print(f"❌ Id format không đúng ở dòng {idx+1}: {id_val}")
                print(f"  Expected format: Store_Dept_Date")
                return False
        
        print("✓ Id format đúng (Store_Dept_Date)")
        
        # Kiểm tra Weekly_Sales không âm
        if (submission['Weekly_Sales'] < 0).any():
            negative_count = (submission['Weekly_Sales'] < 0).sum()
            print(f"⚠️ Có {negative_count} giá trị Weekly_Sales âm (đã được set về 0)")
            submission['Weekly_Sales'] = submission['Weekly_Sales'].clip(lower=0)
            submission.to_csv(submission_path, index=False)
        
        print("✓ Weekly_Sales không có giá trị âm")
        
        # So sánh với sample nếu có
        if sample_path and os.path.exists(sample_path):
            sample = pd.read_csv(sample_path)
            if len(submission) != len(sample):
                print(f"⚠️ Số dòng khác nhau: submission={len(submission)}, sample={len(sample)}")
            else:
                print(f"✓ Số dòng giống sample: {len(submission)}")
            
            # Kiểm tra Id có giống không
            if 'Id' in sample.columns:
                sample_ids_set = set(sample['Id'])
                submission_ids_set = set(submission['Id'])
                if sample_ids_set == submission_ids_set:
                    print("✓ Tất cả Id giống với sample")
                else:
                    missing = sample_ids_set - submission_ids_set
                    extra = submission_ids_set - sample_ids_set
                    if missing:
                        print(f"⚠️ Thiếu {len(missing)} Id so với sample")
                    if extra:
                        print(f"⚠️ Thừa {len(extra)} Id so với sample")
        
        print("\n✓ Format submission đúng!")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi kiểm tra format: {e}")
        return False


def main():
    """Hàm chính"""
    print("="*80)
    print("TẠO FILE SUBMISSION CUỐI CÙNG")
    print("="*80)
    
    # 1. Load best model
    model = load_best_model()
    if model is None:
        return None
    
    # 2. Load feature names
    feature_names = load_feature_names()
    if feature_names is None:
        return None
    
    # 3. Load test_detail
    print("\n" + "="*80)
    print("LOAD TEST DATA")
    print("="*80)
    
    test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
    if not os.path.exists(test_detail_path):
        print(f"❌ Không tìm thấy file: {test_detail_path}")
        print("   Vui lòng chạy preprocessing.py trước")
        return None
    
    try:
        test_detail = pd.read_csv(test_detail_path)
        # Convert Date to datetime
        if 'Date' in test_detail.columns:
            test_detail['Date'] = pd.to_datetime(test_detail['Date'])
        
        print(f"✓ Đã load test_detail: {test_detail.shape}")
        print(f"  Columns: {list(test_detail.columns)[:10]}...")
    except Exception as e:
        print(f"❌ Lỗi khi load test_detail: {e}")
        return None
    
    # 4. Prepare test data
    X_test = prepare_test_data(test_detail, feature_names)
    
    # 5. Make predictions
    predictions = make_predictions(model, X_test)
    if predictions is None:
        return None
    
    # 6. Create submission file
    submission = create_submission_file(test_detail, predictions)
    
    # 7. Verify format
    submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    sample_path = os.path.join(DATA_RAW_DIR, 'walmart-sampleSubmission.csv')
    verify_submission_format(submission_path, sample_path)
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH TẠO SUBMISSION!")
    print("="*80)
    print(f"\nFile submission đã được lưu tại: {os.path.join(OUTPUT_DIR, 'submission.csv')}")
    
    return submission


if __name__ == "__main__":
    submission = main()

