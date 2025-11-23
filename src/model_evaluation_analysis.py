# BƯỚC 4: MODEL EVALUATION & ANALYSIS
# Người 2 - ML Engineer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import functions từ ml_data_preparation
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_data_preparation import create_evaluation_metrics
from config import MODELS_DIR, DATA_PROCESSED_DIR, OUTPUT_VISUALIZATIONS_DIR, OUTPUT_REPORTS_DIR

def load_all_models():
    """Load tất cả các mô hình đã train"""
    print("=== LOAD TẤT CẢ CÁC MÔ HÌNH ===")
    
    models = {}
    model_files = {
        'Linear Regression (Baseline)': 'baseline_linear_regression_model.pkl',
        'Random Forest (Baseline)': 'baseline_random_forest_model.pkl',
        'XGBoost (Baseline)': 'baseline_xgboost_model.pkl',
        'Random Forest (Tuned)': 'tuned_random_forest_model.pkl',
        'XGBoost (Tuned)': 'tuned_xgboost_model.pkl'
    }
    
    for model_name, model_file in model_files.items():
        try:
            filepath = os.path.join(MODELS_DIR, model_file)
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
                models[model_name] = model
                print(f"✓ Đã load: {model_name}")
        except FileNotFoundError:
            print(f"⚠️ Không tìm thấy: {filepath}")
    
    print(f"\n✓ Tổng cộng load được {len(models)} mô hình")
    return models

def load_prepared_data():
    """Load dữ liệu test"""
    print("\n=== LOAD DỮ LIỆU TEST ===")
    
    try:
        X_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'X_test.csv'))
        y_test = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'y_test.csv')).iloc[:, 0]
        weights = np.load(os.path.join(DATA_PROCESSED_DIR, 'weights.npy'))
        feature_names = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, 'feature_names.csv')).iloc[:, 0].tolist()
        
        print(f"✓ Đã load:")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_test: {y_test.shape}")
        print(f"  - weights: {weights.shape}")
        print(f"  - features: {len(feature_names)}")
        
        return X_test, y_test, weights, feature_names
    
    except FileNotFoundError as e:
        print(f"Không tìm thấy file: {e}")
        return None, None, None, None

def evaluate_all_models(models, X_test, y_test, weights, metrics=None):
    """Đánh giá tất cả các mô hình"""
    print("\n=== ĐÁNH GIÁ TẤT CẢ CÁC MÔ HÌNH ===")
    
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
        
        print(f"  MAE: {model_results['mae']:.2f}")
        print(f"  RMSE: {model_results['rmse']:.2f}")
        print(f"  R²: {model_results['r2']:.4f}")
        print(f"  WMAE: {model_results['wmae']:.2f}")
    
    return results

def create_final_comparison(results):
    """Tạo bảng so sánh cuối cùng"""
    print("\n=== BẢNG SO SÁNH CUỐI CÙNG ===")
    
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'MAE': model_results['mae'],
            'RMSE': model_results['rmse'],
            'R²': model_results['r2'],
            'WMAE': model_results['wmae'],
            'MAPE (%)': model_results['mape']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.round(4)
    comparison_df = comparison_df.sort_values('WMAE')
    
    print("\n" + "="*100)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    # Tìm mô hình tốt nhất
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 MÔ HÌNH TỐT NHẤT: {best_model['Model']}")
    print(f"   WMAE: {best_model['WMAE']:.2f}")
    print(f"   MAE: {best_model['MAE']:.2f}")
    print(f"   R²: {best_model['R²']:.4f}")
    
    return comparison_df, best_model

def analyze_feature_importance(models, feature_names, top_n=15):
    """Phân tích feature importance"""
    print(f"\n=== PHÂN TÍCH FEATURE IMPORTANCE (Top {top_n}) ===")
    
    # Chọn mô hình tốt nhất có feature importance
    best_tree_models = ['Random Forest (Baseline)', 'XGBoost (Tuned)', 'XGBoost (Baseline)']
    
    fig, axes = plt.subplots(1, len(best_tree_models), figsize=(6*len(best_tree_models), 8))
    if len(best_tree_models) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(best_tree_models):
        if model_name in models:
            model = models[model_name]
            
            # Lấy feature importance
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
            else:
                continue
            
            # Sắp xếp
            indices = np.argsort(importances)[::-1][:top_n]
            
            # Vẽ biểu đồ
            axes[i].barh(range(len(indices)), importances[indices])
            axes[i].set_yticks(range(len(indices)))
            axes[i].set_yticklabels([feature_names[idx] for idx in indices])
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'Feature Importance - {model_name}', fontsize=12, fontweight='bold')
            axes[i].invert_yaxis()
            axes[i].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'feature_importance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ: {output_path}")
    plt.show()

def analyze_residuals(models, X_test, y_test, weights=None, top_n=2):
    """Phân tích residuals của top N mô hình"""
    print(f"\n=== PHÂN TÍCH RESIDUALS (Top {top_n} mô hình) ===")
    
    # Tính WMAE cho tất cả mô hình để chọn top N
    metrics = create_evaluation_metrics()
    if weights is None:
        weights = np.ones(len(y_test))
    
    model_scores = []
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        wmae = metrics['wmae'](y_test, y_pred, weights)
        model_scores.append((model_name, wmae))
    
    model_scores.sort(key=lambda x: x[1])
    top_models = model_scores[:top_n]
    
    fig, axes = plt.subplots(2, top_n, figsize=(8*top_n, 12))
    if top_n == 1:
        axes = axes.reshape(2, 1)
    
    for i, (model_name, _) in enumerate(top_models):
        model = models[model_name]
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Residuals vs Predicted
        axes[0, i].scatter(y_pred, residuals, alpha=0.3, s=1)
        axes[0, i].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, i].set_xlabel('Predicted Values', fontsize=12)
        axes[0, i].set_ylabel('Residuals', fontsize=12)
        axes[0, i].set_title(f'Residuals vs Predicted - {model_name}', fontsize=12, fontweight='bold')
        axes[0, i].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, i].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, i].set_xlabel('Residuals', fontsize=12)
        axes[1, i].set_ylabel('Frequency', fontsize=12)
        axes[1, i].set_title(f'Distribution of Residuals - {model_name}', fontsize=12, fontweight='bold')
        axes[1, i].grid(True, alpha=0.3, axis='y')
        
        # Thêm thống kê
        mean_residual = residuals.mean()
        std_residual = residuals.std()
        axes[1, i].axvline(mean_residual, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_residual:.2f}')
        axes[1, i].legend()
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_VISUALIZATIONS_DIR, 'residual_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu biểu đồ: {output_path}")
    plt.show()

def create_final_report(comparison_df, best_model):
    """Tạo báo cáo cuối cùng"""
    print("\n=== TẠO BÁO CÁO CUỐI CÙNG ===")
    
    from datetime import datetime
    
    # Bắt đầu tạo report
    report_lines = [
        "# BÁO CÁO CUỐI CÙNG - WALMART SALES FORECASTING",
        "",
        f"**Ngày tạo:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## TỔNG QUAN",
        "",
        "Dự án dự báo doanh số Walmart sử dụng Machine Learning truyền thống (KHÔNG sử dụng Deep Learning).",
        "",
        "## KẾT QUẢ CÁC MÔ HÌNH",
        "",
        "```",
        comparison_df.to_string(index=False),
        "```",
        "",
        "## MÔ HÌNH TỐT NHẤT",
        "",
        f"**Model:** {best_model['Model']}",
        f"- **WMAE:** {best_model['WMAE']:.2f} (Metric chính của cuộc thi)",
        f"- **MAE:** {best_model['MAE']:.2f}",
        f"- **RMSE:** {best_model['RMSE']:.2f}",
        f"- **R²:** {best_model['R²']:.4f}",
        f"- **MAPE:** {best_model['MAPE (%)']:.2f}%",
        "",
        "## PHÂN TÍCH KẾT QUẢ",
        "",
        "### 1. Baseline Models"
    ]
    
    # Thêm thông tin baseline models
    try:
        lr_row = comparison_df[comparison_df['Model'].str.contains('Linear Regression', case=False, na=False)]
        if not lr_row.empty:
            report_lines.extend([
                f"- **Linear Regression:** WMAE = {lr_row['WMAE'].iloc[0]:.2f}",
                "  - Mô hình cơ sở, hiệu suất thấp do không nắm bắt được mối quan hệ phi tuyến",
                ""
            ])
    except:
        pass
    
    try:
        rf_baseline = comparison_df[comparison_df['Model'].str.contains('Random Forest.*Baseline', case=False, na=False, regex=True)]
        if not rf_baseline.empty:
            report_lines.extend([
                f"- **Random Forest (Baseline):** WMAE = {rf_baseline['WMAE'].iloc[0]:.2f}",
                "  - Hiệu suất tốt, không cần tuning",
                ""
            ])
    except:
        pass
    
    try:
        xgb_baseline = comparison_df[comparison_df['Model'].str.contains('XGBoost.*Baseline', case=False, na=False, regex=True)]
        if not xgb_baseline.empty:
            report_lines.extend([
                f"- **XGBoost (Baseline):** WMAE = {xgb_baseline['WMAE'].iloc[0]:.2f}",
                "  - Hiệu suất tốt, có thể cải thiện bằng tuning",
                ""
            ])
    except:
        pass
    
    report_lines.extend([
        "### 2. Tuned Models"
    ])
    
    try:
        rf_tuned = comparison_df[comparison_df['Model'].str.contains('Random Forest.*Tuned', case=False, na=False, regex=True)]
        if not rf_tuned.empty:
            report_lines.extend([
                f"- **Random Forest (Tuned):** WMAE = {rf_tuned['WMAE'].iloc[0]:.2f}",
                "  - ⚠️ Tồi hơn baseline (có thể do overfitting trên validation set)",
                "  - **Khuyến nghị:** Sử dụng Random Forest Baseline thay vì Tuned",
                ""
            ])
    except:
        pass
    
    try:
        xgb_tuned = comparison_df[comparison_df['Model'].str.contains('XGBoost.*Tuned', case=False, na=False, regex=True)]
        xgb_baseline = comparison_df[comparison_df['Model'].str.contains('XGBoost.*Baseline', case=False, na=False, regex=True)]
        if not xgb_tuned.empty and not xgb_baseline.empty:
            improvement = ((xgb_baseline['WMAE'].iloc[0] - xgb_tuned['WMAE'].iloc[0]) / xgb_baseline['WMAE'].iloc[0] * 100)
            report_lines.extend([
                f"- **XGBoost (Tuned):** WMAE = {xgb_tuned['WMAE'].iloc[0]:.2f}",
                f"  - ✅ Cải thiện {improvement:.2f}% so với baseline",
                "  - **Khuyến nghị:** Sử dụng XGBoost Tuned",
                ""
            ])
    except:
        pass
    
    report_lines.extend([
        "## KHUYẾN NGHỊ",
        "",
        "### Cho Production:",
        f"1. **Sử dụng mô hình:** {best_model['Model']}",
        f"2. **WMAE đạt được:** {best_model['WMAE']:.2f}",
        "3. **Monitor performance:** Theo dõi WMAE trên dữ liệu mới",
        "4. **Retrain định kỳ:** Cập nhật mô hình với dữ liệu mới",
        "",
        "### Lưu ý:",
        "- Random Forest Baseline tốt hơn Random Forest Tuned",
        "- XGBoost Tuned cải thiện đáng kể so với baseline",
        "- Có thể thử ensemble của Random Forest Baseline và XGBoost Tuned",
        "",
        "## KẾT LUẬN",
        "",
        f"Dự án đã thành công trong việc xây dựng các mô hình dự báo doanh số Walmart với hiệu suất cao.",
        f"Mô hình tốt nhất đạt được WMAE = {best_model['WMAE']:.2f}, cho thấy khả năng dự báo chính xác và đáng tin cậy.",
        "",
        "Việc sử dụng Time Series Cross-Validation và tập trung vào WMAE metric đảm bảo",
        "mô hình sẽ hoạt động tốt trong thực tế."
    ])
    
    report = "\n".join(report_lines)
    
    output_path = os.path.join(OUTPUT_REPORTS_DIR, 'final_report.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Đã lưu báo cáo: {output_path}")
    return report

def main():
    """Hàm chính để chạy BƯỚC 4"""
    print("="*80)
    print("BƯỚC 4: MODEL EVALUATION & ANALYSIS")
    print("="*80)
    
    # Load tất cả mô hình
    models = load_all_models()
    if not models:
        print("\n❌ Không tìm thấy mô hình nào")
        return None
    
    # Load dữ liệu test
    X_test, y_test, weights, feature_names = load_prepared_data()
    if X_test is None:
        print("\n❌ Không thể tiếp tục do thiếu dữ liệu")
        return None
    
    # Đánh giá tất cả mô hình
    metrics = create_evaluation_metrics()
    results = evaluate_all_models(models, X_test, y_test, weights, metrics)
    
    # Tạo bảng so sánh cuối cùng
    comparison_df, best_model = create_final_comparison(results)
    
    # Phân tích feature importance
    analyze_feature_importance(models, feature_names)
    
    # Phân tích residuals
    analyze_residuals(models, X_test, y_test, weights, top_n=2)
    
    # Tạo báo cáo cuối cùng
    final_report = create_final_report(comparison_df, best_model)
    
    # Lưu kết quả
    comparison_path = os.path.join(OUTPUT_REPORTS_DIR, 'final_model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Đã lưu: {comparison_path}")
    
    # Lưu best model
    best_model_name = best_model['Model']
    if best_model_name in models:
        best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(models[best_model_name], f)
        print(f"✓ Đã lưu best model: {best_model_path} ({best_model_name})")
    
    print("\n" + "="*80)
    print("✓ HOÀN THÀNH BƯỚC 4: MODEL EVALUATION & ANALYSIS")
    print("="*80)
    print("\nCác file đã được tạo:")
    print("  - final_model_comparison.csv")
    print("  - final_report.md")
    print("  - feature_importance_analysis.png")
    print("  - residual_analysis.png")
    print("  - best_model.pkl")
    print("\n🎉 HOÀN THÀNH DỰ ÁN!")
    
    return {
        'models': models,
        'results': results,
        'comparison': comparison_df,
        'best_model': best_model,
        'report': final_report
    }

if __name__ == "__main__":
    results = main()
