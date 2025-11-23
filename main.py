"""
MAIN SCRIPT - Chạy toàn bộ pipeline từ đầu đến cuối
Từ preprocessing đến tạo file submission cuối cùng
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR
import pandas as pd
import numpy as np


def run_preprocessing():
    """
    BƯỚC 1: PREPROCESSING
    Load và merge dữ liệu raw, tạo train_detail và test_detail
    """
    print("\n" + "="*80)
    print("BƯỚC 1: PREPROCESSING")
    print("="*80)
    
    # Kiểm tra xem đã có train_detail và test_detail chưa
    train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
    test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
    
    if os.path.exists(train_detail_path) and os.path.exists(test_detail_path):
        print("✓ Đã có train_detail.csv và test_detail.csv, bỏ qua preprocessing")
        return True
    
    print("Chạy preprocessing...")
    
    try:
        # Chạy preprocessing.py như một script riêng
        import subprocess
        preprocessing_script = os.path.join('src', 'preprocessing.py')
        
        if not os.path.exists(preprocessing_script):
            print(f"❌ Không tìm thấy file: {preprocessing_script}")
            return False
        
        # Chạy script
        result = subprocess.run(
            [sys.executable, preprocessing_script],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ Preprocessing thất bại với exit code: {result.returncode}")
            return False
        
        # Kiểm tra lại sau khi chạy
        if os.path.exists(train_detail_path) and os.path.exists(test_detail_path):
            print("✓ Hoàn thành preprocessing")
            return True
        else:
            print("❌ Preprocessing không tạo được files")
            return False
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_ml_data_preparation():
    """
    BƯỚC 2: ML DATA PREPARATION
    Chuẩn bị dữ liệu cho machine learning
    """
    print("\n" + "="*80)
    print("BƯỚC 2: ML DATA PREPARATION")
    print("="*80)
    
    try:
        from ml_data_preparation import main as ml_prep_main
        result = ml_prep_main()
        
        if result is None:
            print("❌ ML Data Preparation thất bại")
            return False
        
        print("✓ Hoàn thành ML Data Preparation")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy ML Data Preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_baseline_models():
    """
    BƯỚC 3: BASELINE MODELS
    Train các mô hình cơ sở (Linear Regression, Random Forest, XGBoost)
    """
    print("\n" + "="*80)
    print("BƯỚC 3: BASELINE MODELS")
    print("="*80)
    
    try:
        from baseline_models import main as baseline_main
        result = baseline_main()
        
        if result is None:
            print("❌ Baseline Models thất bại")
            return False
        
        print("✓ Hoàn thành Baseline Models")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy Baseline Models: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_hyperparameter_tuning():
    """
    BƯỚC 4: HYPERPARAMETER TUNING
    Tuning hyperparameters cho Random Forest và XGBoost
    """
    print("\n" + "="*80)
    print("BƯỚC 4: HYPERPARAMETER TUNING")
    print("="*80)
    
    try:
        from hyperparameter_tuning import main as tuning_main
        result = tuning_main()
        
        if result is None:
            print("❌ Hyperparameter Tuning thất bại")
            return False
        
        print("✓ Hoàn thành Hyperparameter Tuning")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy Hyperparameter Tuning: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_evaluation():
    """
    BƯỚC 5: MODEL EVALUATION & ANALYSIS
    Đánh giá tất cả mô hình và chọn mô hình tốt nhất
    """
    print("\n" + "="*80)
    print("BƯỚC 5: MODEL EVALUATION & ANALYSIS")
    print("="*80)
    
    try:
        from model_evaluation_analysis import main as eval_main
        result = eval_main()
        
        if result is None:
            print("❌ Model Evaluation thất bại")
            return False
        
        print("✓ Hoàn thành Model Evaluation & Analysis")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy Model Evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_create_submission():
    """
    BƯỚC 6: CREATE SUBMISSION
    Tạo file submission cuối cùng từ model tốt nhất
    """
    print("\n" + "="*80)
    print("BƯỚC 6: CREATE SUBMISSION")
    print("="*80)
    
    try:
        from create_submission import main as submission_main
        result = submission_main()
        
        if result is None:
            print("❌ Create Submission thất bại")
            return False
        
        print("✓ Hoàn thành Create Submission")
        return True
    
    except Exception as e:
        print(f"❌ Lỗi khi chạy Create Submission: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(skip_steps=None):
    """
    Hàm chính để chạy toàn bộ pipeline
    
    Args:
        skip_steps (list): Danh sách các bước cần bỏ qua (1-6)
                           Ví dụ: [1, 2] để bỏ qua preprocessing và ml_data_preparation
    """
    print("="*80)
    print("WALMART SALES FORECASTING - FULL PIPELINE")
    print("="*80)
    print("\nPipeline sẽ chạy các bước sau:")
    print("  1. Preprocessing")
    print("  2. ML Data Preparation")
    print("  3. Baseline Models")
    print("  4. Hyperparameter Tuning")
    print("  5. Model Evaluation & Analysis")
    print("  6. Create Submission")
    
    if skip_steps:
        print(f"\n⚠️ Bỏ qua các bước: {skip_steps}")
    
    print("\n" + "="*80)
    
    steps = [
        ("Preprocessing", run_preprocessing),
        ("ML Data Preparation", run_ml_data_preparation),
        ("Baseline Models", run_baseline_models),
        ("Hyperparameter Tuning", run_hyperparameter_tuning),
        ("Model Evaluation", run_model_evaluation),
        ("Create Submission", run_create_submission)
    ]
    
    results = {}
    
    for step_num, (step_name, step_func) in enumerate(steps, 1):
        if skip_steps and step_num in skip_steps:
            print(f"\n⏭️ Bỏ qua bước {step_num}: {step_name}")
            results[step_num] = "Skipped"
            continue
        
        print(f"\n{'='*80}")
        print(f"BẮT ĐẦU BƯỚC {step_num}: {step_name}")
        print(f"{'='*80}")
        
        success = step_func()
        results[step_num] = "Success" if success else "Failed"
        
        if not success:
            print(f"\n❌ Bước {step_num} ({step_name}) thất bại!")
            print("Dừng pipeline.")
            break
    
    # Tóm tắt kết quả
    print("\n" + "="*80)
    print("TÓM TẮT KẾT QUẢ")
    print("="*80)
    
    for step_num, (step_name, _) in enumerate(steps, 1):
        status = results.get(step_num, "Not run")
        status_icon = "✓" if status == "Success" else "⏭️" if status == "Skipped" else "❌"
        print(f"{status_icon} Bước {step_num}: {step_name} - {status}")
    
    all_success = all(v == "Success" or v == "Skipped" for v in results.values())
    
    if all_success:
        print("\n" + "="*80)
        print("🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("="*80)
        print("\nCác file output quan trọng:")
        print("  - output/submission.csv (File submission cuối cùng)")
        print("  - models/best_model.pkl (Model tốt nhất)")
        print("  - output/reports/final_model_comparison.csv (So sánh các models)")
        print("  - output/reports/final_report.md (Báo cáo cuối cùng)")
    else:
        print("\n" + "="*80)
        print("⚠️ PIPELINE KHÔNG HOÀN THÀNH ĐẦY ĐỦ")
        print("="*80)
        print("Vui lòng kiểm tra lỗi ở các bước trên và chạy lại.")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Chạy toàn bộ pipeline Walmart Sales Forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python main.py                    # Chạy tất cả các bước
  python main.py --skip 1 2          # Bỏ qua preprocessing và ml_data_preparation
  python main.py --from-step 3       # Bắt đầu từ bước 3
        """
    )
    
    parser.add_argument(
        '--skip',
        type=int,
        nargs='+',
        help='Danh sách các bước cần bỏ qua (1-6)'
    )
    
    parser.add_argument(
        '--from-step',
        type=int,
        help='Bắt đầu từ bước này (1-6). Các bước trước đó sẽ được bỏ qua.'
    )
    
    args = parser.parse_args()
    
    skip_steps = args.skip or []
    
    if args.from_step:
        # Bỏ qua tất cả các bước trước from_step
        skip_steps = list(range(1, args.from_step)) + skip_steps
        skip_steps = list(set(skip_steps))  # Remove duplicates
    
    results = main(skip_steps=skip_steps if skip_steps else None)

