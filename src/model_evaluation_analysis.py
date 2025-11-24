# BƯỚC 4: MODEL EVALUATION & ANALYSIS
# So sánh models từ k-fold validation results

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import functions
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, OUTPUT_VISUALIZATIONS_DIR, OUTPUT_REPORTS_DIR, DATA_PROCESSED_DIR

def load_kfold_comparison_results():
    """
    Load kết quả k-fold validation từ CSV files
    
    Returns:
        tuple: (untuned_results_df, tuned_results_df) hoặc (None, None) nếu lỗi
    """
    print("=== LOAD KẾT QUẢ K-FOLD VALIDATION ===")
    
    untuned_path = os.path.join(OUTPUT_DIR, 'kfold_validation_comparison.csv')
    tuned_path = os.path.join(OUTPUT_DIR, 'best_params_kfold_comparison.csv')
    
    untuned_df = None
    tuned_df = None
    
    # Load untuned results
    if os.path.exists(untuned_path):
        try:
            untuned_df = pd.read_csv(untuned_path)
            print(f"✓ Đã load untuned results: {len(untuned_df)} models")
        except Exception as e:
            print(f"Lỗi khi load {untuned_path}: {e}")
    else:
        print(f"Không tìm thấy: {untuned_path}")
    
    # Load tuned results
    if os.path.exists(tuned_path):
        try:
            tuned_df = pd.read_csv(tuned_path)
            print(f"✓ Đã load tuned results: {len(tuned_df)} models")
        except Exception as e:
            print(f"Lỗi khi load {tuned_path}: {e}")
    else:
        print(f"Không tìm thấy: {tuned_path}")
    
    if untuned_df is None and tuned_df is None:
        print("Không có kết quả nào để so sánh")
        return None, None
    
    return untuned_df, tuned_df

def create_comparison_from_csv(untuned_df, tuned_df):
    """
    Tạo bảng so sánh từ kết quả CSV
    
    Args:
        untuned_df: DataFrame chứa kết quả models chưa tuning
        tuned_df: DataFrame chứa kết quả models đã tuning
        
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    print("\n=== TẠO BẢNG SO SÁNH ===")
    
    comparison_data = []
    
    # Thêm untuned models
    if untuned_df is not None:
        for _, row in untuned_df.iterrows():
            comparison_data.append({
                'Model': f"{row['Model']} (Untuned)",
                'Mean_WMAE': row['Mean_WMAE'],
                'Std_WMAE': row['Std_WMAE'],
                'Mean_MAE': row['Mean_MAE'],
                'Std_MAE': row['Std_MAE'],
                'Mean_RMSE': row['Mean_RMSE'],
                'Std_RMSE': row['Std_RMSE'],
                'Mean_R2': row['Mean_R2'],
                'Std_R2': row['Std_R2'],
                'Mean_Train_Time': row['Mean_Train_Time']
            })
    
    # Thêm tuned models
    if tuned_df is not None:
        for _, row in tuned_df.iterrows():
            comparison_data.append({
                'Model': f"{row['Model']} (Tuned)",
                'Mean_WMAE': row['Mean_WMAE'],
                'Std_WMAE': row['Std_WMAE'],
                'Mean_MAE': row['Mean_MAE'],
                'Std_MAE': row['Std_MAE'],
                'Mean_RMSE': row['Mean_RMSE'],
                'Std_RMSE': row['Std_RMSE'],
                'Mean_R2': row['Mean_R2'],
                'Std_R2': row['Std_R2'],
                'Mean_Train_Time': row['Mean_Train_Time']
            })
    
    if len(comparison_data) == 0:
        print("Không có dữ liệu để so sánh")
        return None
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Mean_WMAE')
    
    return comparison_df

def create_final_comparison(comparison_df):
    """Tạo bảng so sánh cuối cùng từ comparison_df"""
    print("\n=== BẢNG SO SÁNH CUỐI CÙNG ===")
    
    if comparison_df is None or len(comparison_df) == 0:
        print("Không có dữ liệu để so sánh")
        return None, None
    
    # Format để hiển thị đẹp
    display_df = comparison_df.copy()
    display_df = display_df.round(4)
    
    print("\n" + "="*100)
    print(display_df.to_string(index=False))
    print("="*100)
    
    # Tìm mô hình tốt nhất
    best_model = comparison_df.iloc[0]
    print(f"\n MÔ HÌNH TỐT NHẤT: {best_model['Model']}")
    print(f"   Mean WMAE: {best_model['Mean_WMAE']:.2f} ± {best_model['Std_WMAE']:.2f}")
    print(f"   Mean MAE: {best_model['Mean_MAE']:.2f} ± {best_model['Std_MAE']:.2f}")
    print(f"   Mean R²: {best_model['Mean_R2']:.4f} ± {best_model['Std_R2']:.4f}")
    
    return comparison_df, best_model

def visualize_comparison(comparison_df):
    """Trực quan hóa kết quả so sánh"""
    print("\n=== TRỰC QUAN HÓA KẾT QUẢ SO SÁNH ===")
    
    if comparison_df is None or len(comparison_df) == 0:
        print("⚠️ Không có dữ liệu để visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = comparison_df['Model'].tolist()
    
    # 1. WMAE Comparison
    mean_wmae = comparison_df['Mean_WMAE'].values
    std_wmae = comparison_df['Std_WMAE'].values
    axes[0, 0].barh(model_names, mean_wmae, xerr=std_wmae, capsize=5, color='steelblue')
    axes[0, 0].set_xlabel('WMAE', fontsize=12)
    axes[0, 0].set_title('WMAE Comparison (K-Fold CV)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    axes[0, 0].invert_yaxis()
    
    # 2. R² Comparison
    mean_r2 = comparison_df['Mean_R2'].values
    std_r2 = comparison_df['Std_R2'].values
    axes[0, 1].barh(model_names, mean_r2, xerr=std_r2, capsize=5, color='coral')
    axes[0, 1].set_xlabel('R²', fontsize=12)
    axes[0, 1].set_title('R² Comparison (K-Fold CV)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    axes[0, 1].invert_yaxis()
    
    # 3. MAE Comparison
    mean_mae = comparison_df['Mean_MAE'].values
    std_mae = comparison_df['Std_MAE'].values
    axes[1, 0].barh(model_names, mean_mae, xerr=std_mae, capsize=5, color='lightgreen')
    axes[1, 0].set_xlabel('MAE', fontsize=12)
    axes[1, 0].set_title('MAE Comparison (K-Fold CV)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # 4. Train Time Comparison
    mean_time = comparison_df['Mean_Train_Time'].values
    axes[1, 1].barh(model_names, mean_time, color='gold')
    axes[1, 1].set_xlabel('Train Time (seconds)', fontsize=12)
    axes[1, 1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'model_comparison_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu biểu đồ: {output_path}")

def create_final_report(comparison_df, best_model):
    """Tạo báo cáo cuối cùng"""
    print("\n=== TẠO BÁO CÁO CUỐI CÙNG ===")
    
    from datetime import datetime
    
    if comparison_df is None or best_model is None:
        print("⚠️ Không có dữ liệu để tạo báo cáo")
        return None
    
    # Format comparison table cho report
    display_df = comparison_df.copy()
    display_df = display_df.round(4)
    
    # Bắt đầu tạo report
    report_lines = [
        "# BÁO CÁO CUỐI CÙNG - WALMART SALES FORECASTING",
        "",
        f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## TỔNG QUAN",
        "",
        "Dự án dự báo doanh số Walmart sử dụng Machine Learning truyền thống (KHÔNG sử dụng Deep Learning).",
        "Sử dụng K-Fold Cross-Validation (K=5) để đánh giá và so sánh các mô hình một cách robust.",
        "",
        "## WORKFLOW",
        "",
        "Pipeline được thực hiện theo các bước sau:",
        "",
        "1. **Preprocessing** (`preprocessing.py`):",
        "   - Load và merge dữ liệu từ các file raw",
        "   - Feature engineering (Week, Month, Year, Day)",
        "   - Chọn features và lưu `train_detail.csv`, `test_detail.csv`, `feature_chosen.csv`",
        "",
        "2. **K-Fold Validation - Untuned Models** (`k_fold_validation.py`):",
        "   - Chạy K-Fold Cross-Validation cho Random Forest và XGBoost với default parameters",
        "   - Lưu kết quả vào `kfold_validation_comparison.csv`",
        "",
        "3. **Hyperparameter Tuning** (`hyperparameter_tuning.py`):",
        "   - Sử dụng RandomizedSearchCV với TimeSeriesSplit để tìm best parameters",
        "   - Lưu best parameters vào `tuned_models_best_params.csv`",
        "",
        "4. **Train With Best Params** (`train_with_best_params.py`):",
        "   - Load best parameters từ CSV",
        "   - Chạy K-Fold Cross-Validation với best parameters",
        "   - Train model tốt nhất và tạo submission file",
        "   - Lưu kết quả vào `best_params_kfold_comparison.csv`",
        "",
        "5. **Model Evaluation & Analysis** (`model_evaluation_analysis.py`):",
        "   - So sánh kết quả untuned vs tuned models",
        "   - Tạo visualization và final report",
        "",
        "## KẾT QUẢ CÁC MÔ HÌNH (K-FOLD CROSS-VALIDATION)",
        "",
        "```",
        display_df.to_string(index=False),
        "```",
        "",
        "## MÔ HÌNH TỐT NHẤT",
        "",
        f"**Model:** {best_model['Model']}",
        f"- **Mean WMAE:** {best_model['Mean_WMAE']:.2f} ± {best_model['Std_WMAE']:.2f} (Metric chính của cuộc thi)",
        f"- **Mean MAE:** {best_model['Mean_MAE']:.2f} ± {best_model['Std_MAE']:.2f}",
        f"- **Mean RMSE:** {best_model['Mean_RMSE']:.2f} ± {best_model['Std_RMSE']:.2f}",
        f"- **Mean R²:** {best_model['Mean_R2']:.4f} ± {best_model['Std_R2']:.4f}",
        f"- **Mean Train Time:** {best_model['Mean_Train_Time']:.2f}s",
        "",
        "## PHÂN TÍCH KẾT QUẢ",
        "",
        "### 1. Models Chưa Tuning (Untuned)"
    ]
    
    # Thêm thông tin untuned models
    untuned_models = comparison_df[comparison_df['Model'].str.contains('Untuned', case=False, na=False)]
    if not untuned_models.empty:
        for _, row in untuned_models.iterrows():
            model_name = row['Model'].replace(' (Untuned)', '')
            report_lines.extend([
                f"- **{model_name} (Untuned):** Mean WMAE = {row['Mean_WMAE']:.2f} ± {row['Std_WMAE']:.2f}",
                f"  - Mean R²: {row['Mean_R2']:.4f} ± {row['Std_R2']:.4f}",
                ""
            ])
    
    report_lines.extend([
        "### 2. Models Đã Tuning (Tuned)"
    ])
    
    # Thêm thông tin tuned models
    tuned_models = comparison_df[comparison_df['Model'].str.contains('Tuned', case=False, na=False)]
    if not tuned_models.empty:
        for _, row in tuned_models.iterrows():
            model_name = row['Model'].replace(' (Tuned)', '')
            # Tìm untuned tương ứng để so sánh
            untuned_row = untuned_models[untuned_models['Model'].str.contains(model_name, case=False, na=False)]
            if not untuned_row.empty:
                improvement = ((untuned_row['Mean_WMAE'].iloc[0] - row['Mean_WMAE']) / untuned_row['Mean_WMAE'].iloc[0] * 100)
                report_lines.extend([
                    f"- **{model_name} (Tuned):** Mean WMAE = {row['Mean_WMAE']:.2f} ± {row['Std_WMAE']:.2f}",
                    f"  - Mean R²: {row['Mean_R2']:.4f} ± {row['Std_R2']:.4f}",
                    f"  - Cải thiện: {improvement:.2f}% so với Untuned",
                    ""
                ])
            else:
                report_lines.extend([
                    f"- **{model_name} (Tuned):** Mean WMAE = {row['Mean_WMAE']:.2f} ± {row['Std_WMAE']:.2f}",
                    f"  - Mean R²: {row['Mean_R2']:.4f} ± {row['Std_R2']:.4f}",
                    ""
                ])
    
    report_lines.extend([
        "## KHUYẾN NGHỊ",
        "",
        "### Cho Production:",
        f"1. **Sử dụng mô hình:** {best_model['Model']}",
        f"2. **Mean WMAE đạt được:** {best_model['Mean_WMAE']:.2f} ± {best_model['Std_WMAE']:.2f}",
        "3. **Monitor performance:** Theo dõi WMAE trên dữ liệu mới",
        "4. **Retrain định kỳ:** Cập nhật mô hình với dữ liệu mới",
        "",
        "## KẾT LUẬN",
        "",
        f"Dự án đã thành công trong việc xây dựng các mô hình dự báo doanh số Walmart với hiệu suất cao.",
        f"Mô hình tốt nhất đạt được Mean WMAE = {best_model['Mean_WMAE']:.2f} ± {best_model['Std_WMAE']:.2f},",
        "cho thấy khả năng dự báo chính xác và đáng tin cậy.",
        "",
        "Việc sử dụng K-Fold Cross-Validation và tập trung vào WMAE metric đảm bảo",
        "mô hình sẽ hoạt động tốt trong thực tế."
    ])
    
    report = "\n".join(report_lines)
    
    output_path = os.path.join(OUTPUT_REPORTS_DIR, 'final_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Đã lưu báo cáo: {output_path}")
    return report

def main():
    """Hàm chính để chạy BƯỚC 4"""
    print("\n BƯỚC 4: MODEL EVALUATION & ANALYSIS")
    
    # Load kết quả k-fold validation từ CSV
    untuned_df, tuned_df = load_kfold_comparison_results()
    if untuned_df is None and tuned_df is None:
        print("\n Không có kết quả để so sánh")
        print("   Vui lòng chạy k_fold_validation.py và train_with_best_params.py trước")
        return None
    
    # Tạo bảng so sánh
    comparison_df = create_comparison_from_csv(untuned_df, tuned_df)
    if comparison_df is None:
        return None
    
    # Tạo bảng so sánh cuối cùng
    comparison_df_final, best_model = create_final_comparison(comparison_df)
    
    # Trực quan hóa
    visualize_comparison(comparison_df)
    
    # Tạo báo cáo cuối cùng
    final_report = create_final_report(comparison_df, best_model)
    
    # Lưu kết quả
    comparison_path = os.path.join(OUTPUT_REPORTS_DIR, 'final_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nĐã lưu: {comparison_path}")
    
    print("\n HOÀN THÀNH BƯỚC 4: MODEL EVALUATION & ANALYSIS")
    print("\nCác file đã được tạo:")
    print("  - final_model_comparison.csv")
    print("  - final_report.md")
    print("  - model_comparison_visualization.png")
    print("\n HOÀN THÀNH DỰ ÁN!")
    
    return {
        'comparison': comparison_df,
        'best_model': best_model,
        'report': final_report
    }

if __name__ == "__main__":
    results = main()
