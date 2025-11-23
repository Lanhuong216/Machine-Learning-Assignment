# BÁO CÁO CUỐI CÙNG - WALMART SALES FORECASTING

**Ngày tạo:** 2025-11-23 23:40:46

## TỔNG QUAN

Dự án dự báo doanh số Walmart sử dụng Machine Learning truyền thống (KHÔNG sử dụng Deep Learning).
Sử dụng K-Fold Cross-Validation (K=5) để đánh giá và so sánh các mô hình một cách robust.

## WORKFLOW

Pipeline được thực hiện theo các bước sau:

1. **Preprocessing** (`preprocessing.py`):
   - Load và merge dữ liệu từ các file raw
   - Feature engineering (Week, Month, Year, Day)
   - Chọn features và lưu `train_detail.csv`, `test_detail.csv`, `feature_chosen.csv`

2. **K-Fold Validation - Untuned Models** (`k_fold_validation.py`):
   - Chạy K-Fold Cross-Validation cho Random Forest và XGBoost với default parameters
   - Lưu kết quả vào `kfold_validation_comparison.csv`

3. **Hyperparameter Tuning** (`hyperparameter_tuning.py`):
   - Sử dụng RandomizedSearchCV với TimeSeriesSplit để tìm best parameters
   - Lưu best parameters vào `tuned_models_best_params.csv`

4. **Train With Best Params** (`train_with_best_params.py`):
   - Load best parameters từ CSV
   - Chạy K-Fold Cross-Validation với best parameters
   - Train model tốt nhất và tạo submission file
   - Lưu kết quả vào `best_params_kfold_comparison.csv`

5. **Model Evaluation & Analysis** (`model_evaluation_analysis.py`):
   - So sánh kết quả untuned vs tuned models
   - Tạo visualization và final report

## KẾT QUẢ CÁC MÔ HÌNH (K-FOLD CROSS-VALIDATION)

```
                  Model  Mean_WMAE  Std_WMAE  Mean_MAE  Std_MAE  Mean_RMSE  Std_RMSE  Mean_R2  Std_R2  Mean_Train_Time
        XGBoost (Tuned)  1246.9071    8.0617 1134.1198   6.0946  2511.5140   54.0841   0.9878  0.0005          13.8660
Random Forest (Untuned)  1535.2139   13.8873 1380.6197  11.3110  3310.4627  135.3429   0.9787  0.0016          14.3484
  Random Forest (Tuned)  1552.3776   14.8446 1360.9112   8.7214  3500.3502   84.0591   0.9762  0.0010          29.2180
      XGBoost (Untuned)  4065.6836   22.9553 3900.7808  22.7252  6985.2421   51.0073   0.9054  0.0015           1.0070
```

## MÔ HÌNH TỐT NHẤT

**Model:** XGBoost (Tuned)
- **Mean WMAE:** 1246.91 ± 8.06 (Metric chính của cuộc thi)
- **Mean MAE:** 1134.12 ± 6.09
- **Mean RMSE:** 2511.51 ± 54.08
- **Mean R²:** 0.9878 ± 0.0005
- **Mean Train Time:** 13.87s

## PHÂN TÍCH KẾT QUẢ

### 1. Models Chưa Tuning (Untuned)
- **Random Forest (Untuned):** Mean WMAE = 1535.21 ± 13.89
  - Mean R²: 0.9787 ± 0.0016

- **XGBoost (Untuned):** Mean WMAE = 4065.68 ± 22.96
  - Mean R²: 0.9054 ± 0.0015

### 2. Models Đã Tuning (Tuned)
- **XGBoost (Tuned):** Mean WMAE = 1246.91 ± 8.06
  - Mean R²: 0.9878 ± 0.0005
  - Cải thiện: 69.33% so với Untuned

- **Random Forest (Untuned) (Tuned):** Mean WMAE = 1535.21 ± 13.89
  - Mean R²: 0.9787 ± 0.0016

- **Random Forest (Tuned):** Mean WMAE = 1552.38 ± 14.84
  - Mean R²: 0.9762 ± 0.0010
  - Cải thiện: -1.12% so với Untuned

- **XGBoost (Untuned) (Tuned):** Mean WMAE = 4065.68 ± 22.96
  - Mean R²: 0.9054 ± 0.0015

## KHUYẾN NGHỊ

### Cho Production:
1. **Sử dụng mô hình:** XGBoost (Tuned)
2. **Mean WMAE đạt được:** 1246.91 ± 8.06
3. **Monitor performance:** Theo dõi WMAE trên dữ liệu mới
4. **Retrain định kỳ:** Cập nhật mô hình với dữ liệu mới

## KẾT LUẬN

Dự án đã thành công trong việc xây dựng các mô hình dự báo doanh số Walmart với hiệu suất cao.
Mô hình tốt nhất đạt được Mean WMAE = 1246.91 ± 8.06,
cho thấy khả năng dự báo chính xác và đáng tin cậy.

Việc sử dụng K-Fold Cross-Validation và tập trung vào WMAE metric đảm bảo
mô hình sẽ hoạt động tốt trong thực tế.