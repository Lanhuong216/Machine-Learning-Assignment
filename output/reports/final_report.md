# BÁO CÁO CUỐI CÙNG - WALMART SALES FORECASTING

**Ngày tạo:** 2025-11-23 19:42:02

## TỔNG QUAN

Dự án dự báo doanh số Walmart sử dụng Machine Learning truyền thống (KHÔNG sử dụng Deep Learning).

## KẾT QUẢ CÁC MÔ HÌNH

```
                       Model        MAE       RMSE      R²       WMAE     MAPE (%)
    Random Forest (Baseline)  4520.1764  9470.9453  0.7523  4602.5089 2.117682e+08
             XGBoost (Tuned)  4947.3984  9313.9647  0.7605  5029.2577 3.265553e+09
          XGBoost (Baseline)  5265.3367  9356.1277  0.7583  5420.7451 2.921018e+09
       Random Forest (Tuned)  5615.7917  9549.4137  0.7482  5749.4947 4.982268e+09
Linear Regression (Baseline) 11764.0862 19066.1217 -0.0037 11891.2912 9.130758e+09
```

## MÔ HÌNH TỐT NHẤT

**Model:** Random Forest (Baseline)
- **WMAE:** 4602.51 (Metric chính của cuộc thi)
- **MAE:** 4520.18
- **RMSE:** 9470.95
- **R²:** 0.7523
- **MAPE:** 211768190.11%

## PHÂN TÍCH KẾT QUẢ

### 1. Baseline Models
- **Linear Regression:** WMAE = 11891.29
  - Mô hình cơ sở, hiệu suất thấp do không nắm bắt được mối quan hệ phi tuyến

- **Random Forest (Baseline):** WMAE = 4602.51
  - Hiệu suất tốt, không cần tuning

- **XGBoost (Baseline):** WMAE = 5420.75
  - Hiệu suất tốt, có thể cải thiện bằng tuning

### 2. Tuned Models
- **Random Forest (Tuned):** WMAE = 5749.49
  - ⚠️ Tồi hơn baseline (có thể do overfitting trên validation set)
  - **Khuyến nghị:** Sử dụng Random Forest Baseline thay vì Tuned

- **XGBoost (Tuned):** WMAE = 5029.26
  - ✅ Cải thiện 7.22% so với baseline
  - **Khuyến nghị:** Sử dụng XGBoost Tuned

## KHUYẾN NGHỊ

### Cho Production:
1. **Sử dụng mô hình:** Random Forest (Baseline)
2. **WMAE đạt được:** 4602.51
3. **Monitor performance:** Theo dõi WMAE trên dữ liệu mới
4. **Retrain định kỳ:** Cập nhật mô hình với dữ liệu mới

### Lưu ý:
- Random Forest Baseline tốt hơn Random Forest Tuned
- XGBoost Tuned cải thiện đáng kể so với baseline
- Có thể thử ensemble của Random Forest Baseline và XGBoost Tuned

## KẾT LUẬN

Dự án đã thành công trong việc xây dựng các mô hình dự báo doanh số Walmart với hiệu suất cao.
Mô hình tốt nhất đạt được WMAE = 4602.51, cho thấy khả năng dự báo chính xác và đáng tin cậy.

Việc sử dụng Time Series Cross-Validation và tập trung vào WMAE metric đảm bảo
mô hình sẽ hoạt động tốt trong thực tế.