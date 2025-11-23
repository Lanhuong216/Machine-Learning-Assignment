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
    print("\n BƯỚC 1: PREPROCESSING")
    
    # Kiểm tra xem đã có train_detail và test_detail chưa
    train_detail_path = os.path.join(DATA_PROCESSED_DIR, 'train_detail.csv')
    test_detail_path = os.path.join(DATA_PROCESSED_DIR, 'test_detail.csv')
    
    if os.path.exists(train_detail_path) and os.path.exists(test_detail_path):
        return True
    
    try:
        # Chạy preprocessing.py như một script riêng
        import subprocess
        preprocessing_script = os.path.join('src', 'preprocessing.py')
        
        if not os.path.exists(preprocessing_script):
            return False
        
        # Chạy script
        result = subprocess.run(
            [sys.executable, preprocessing_script],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Preprocessing thất bại với exit code: {result.returncode}")
            return False
        
        # Kiểm tra lại sau khi chạy
        if os.path.exists(train_detail_path) and os.path.exists(test_detail_path):
            print("Hoàn thành preprocessing")
            return True
        else:
            print("Preprocessing không tạo được files")
            return False
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def run_k_fold_validation():
    """
    BƯỚC 2: K-FOLD VALIDATION (UNTUNED MODELS)
    Chạy k-fold validation cho Random Forest và XGBoost chưa tuning
    """
    print("\n BƯỚC 2: K-FOLD VALIDATION (UNTUNED MODELS)")
    
    try:
        from k_fold_validation_untuned import main as kfold_main
        result = kfold_main()
        
        if result is None:
            print("K-Fold Validation thất bại")
            return False
        
        print("Hoàn thành K-Fold Validation")
        return True
    
    except Exception as e:
        print(f"Lỗi khi chạy K-Fold Validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_hyperparameter_tuning():
    """
    BƯỚC 3: HYPERPARAMETER TUNING
    Tuning hyperparameters cho Random Forest và XGBoost
    Xuất ra tuned_models_best_params.csv
    """
    print("\n BƯỚC 3: HYPERPARAMETER TUNING")
    
    try:
        from hyperparameter_tuning import main as tuning_main
        result = tuning_main()
        
        if result is None:
            print("Hyperparameter Tuning thất bại")
            return False
        
        print("Hoàn thành Hyperparameter Tuning")
        return True
    
    except Exception as e:
        print(f"Lỗi khi chạy Hyperparameter Tuning: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_train_with_best_params():
    """
    BƯỚC 4: TRAIN WITH BEST PARAMS
    Train models với best parameters từ tuned_models_best_params.csv
    Sử dụng k-fold validation và xuất submission
    """
    print("\n BƯỚC 4: TRAIN WITH BEST PARAMS")
    
    try:
        from k_fold_validation_tuned import main as train_main
        result = train_main()
        
        if result is None:
            print("Train With Best Params thất bại")
            return False
        
        print("Hoàn thành Train With Best Params")
        return True
    
    except Exception as e:
        print(f"Lỗi khi chạy Train With Best Params: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_model_evaluation():
    """
    BƯỚC 5: MODEL EVALUATION & ANALYSIS
    So sánh models từ k-fold validation results (untuned vs tuned)
    """
    print("\n BƯỚC 5: MODEL EVALUATION & ANALYSIS")
    
    try:
        from model_evaluation_analysis import main as eval_main
        result = eval_main()
        
        if result is None:
            print("Model Evaluation thất bại")
            return False
        
        print("Hoàn thành Model Evaluation & Analysis")
        return True
    
    except Exception as e:
        print(f"Lỗi khi chạy Model Evaluation: {e}")
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
    print("\n WALMART   SALES FORECASTING - FULL PIPELINE")
    print("\nPipeline sẽ chạy các bước sau:")
    print("  1. Preprocessing → train_detail.csv, test_detail.csv")
    print("  2. K-Fold Validation (Untuned) → kfold_validation_comparison.csv")
    print("  3. Hyperparameter Tuning → tuned_models_best_params.csv")
    print("  4. Train With Best Params → best_params_kfold_comparison.csv + submission")
    print("  5. Model Evaluation & Analysis → final_report.md")
    
    if skip_steps:
        print(f"\n Bỏ qua các bước: {skip_steps}")
    
    steps = [
        ("Preprocessing", run_preprocessing),
        ("K-Fold Validation (Untuned)", run_k_fold_validation),
        ("Hyperparameter Tuning", run_hyperparameter_tuning),
        ("Train With Best Params", run_train_with_best_params),
        ("Model Evaluation & Analysis", run_model_evaluation)
    ]
    
    results = {}
    
    for step_num, (step_name, step_func) in enumerate(steps, 1):
        if skip_steps and step_num in skip_steps:
            print(f"\n Bỏ qua bước {step_num}: {step_name}")
            results[step_num] = "Skipped"
            continue
        
        print(f"\n BẮT ĐẦU BƯỚC {step_num}: {step_name}")
        
        success = step_func()
        results[step_num] = "Success" if success else "Failed"
        
        if not success:
            print(f"\n Không thành công bước {step_num} ({step_name})!")
            print("Dừng pipeline.")
            break
    
    # Tóm tắt kết quả
    print("\n TÓM TẮT KẾT QUẢ")
    
    for step_num, (step_name, _) in enumerate(steps, 1):
        status = results.get(step_num, "Not run")
        status_icon = "Success" if status == "Success" else "Skipped" if status == "Skipped" else "Failed"
        print(f"Bước {step_num}: {step_name} - {status}")
    
    all_success = all(v == "Success" or v == "Skipped" for v in results.values())
    
    if all_success:
        print("\n HOÀN THÀNH TOÀN BỘ PIPELINE!")
        print("\nCác file output quan trọng:")
        print("  - output/submission_*.csv (File submission từ models)")
        print("  - output/kfold_validation_comparison.csv (Kết quả untuned models)")
        print("  - output/best_params_kfold_comparison.csv (Kết quả tuned models)")
        print("  - output/reports/tuned_models_best_params.csv (Best parameters)")
        print("  - output/reports/final_model_comparison.csv (So sánh các models)")
        print("  - output/reports/final_report.md (Báo cáo cuối cùng)")
    else:
        print("\n PIPELINE KHÔNG HOÀN THÀNH ĐẦY ĐỦ")
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
        help='Danh sách các bước cần bỏ qua (1-5)'
    )
    
    parser.add_argument(
        '--from-step',
        type=int,
        help='Bắt đầu từ bước này (1-5). Các bước trước đó sẽ được bỏ qua.'
    )
    
    args = parser.parse_args()
    
    skip_steps = args.skip or []
    
    if args.from_step:
        # Bỏ qua tất cả các bước trước from_step
        skip_steps = list(range(1, args.from_step)) + skip_steps
        skip_steps = list(set(skip_steps))  # Remove duplicates
    
    results = main(skip_steps=skip_steps if skip_steps else None)

