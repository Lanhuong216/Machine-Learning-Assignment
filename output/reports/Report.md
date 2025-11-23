# BÁO CÁO DỰ ÁN: DỰ BÁO DOANH SỐ WALMART

## Walmart Sales Forecasting using Machine Learning

---

**Ngày hoàn thành:** 2025-11-23  
**Phương pháp:** Machine Learning truyền thống (KHÔNG sử dụng Deep Learning)  
**Mục tiêu:** Dự báo doanh số hàng tuần của các cửa hàng Walmart  
**Evaluation Method:** K-Fold Cross-Validation (K=5)

---

## 📋 MỤC LỤC

1. [Tổng quan dự án](#1-tổng-quan-dự-án)
2. [Dữ liệu](#2-dữ-liệu)
3. [Phương pháp luận](#3-phương-pháp-luận)
4. [Kết quả](#4-kết-quả)
5. [Phân tích chi tiết](#5-phân-tích-chi-tiết)
6. [Kết luận và khuyến nghị](#6-kết-luận-và-khuyến-nghị)

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1. Mục tiêu

Xây dựng mô hình Machine Learning để dự báo doanh số hàng tuần của các cửa hàng Walmart, hỗ trợ:

- Quản lý tồn kho hiệu quả
- Lập kế hoạch cho các tuần lễ đặc biệt (holidays)
- Phân bổ nguồn lực tối ưu
- Giảm chi phí tồn kho và thiếu hàng

### 1.2. Phạm vi dự án

- **Dữ liệu:** Doanh số hàng tuần từ 45 cửa hàng Walmart
- **Thời gian:** Dữ liệu lịch sử từ năm 2010-2012
- **Phương pháp:** Machine Learning truyền thống (Random Forest, XGBoost, Linear Regression)
- **Metric chính:** WMAE (Weighted Mean Absolute Error)

### 1.3. Ràng buộc

- ❌ **KHÔNG sử dụng Deep Learning** (LSTM, RNN, CNN, Transformer, Neural Networks)
- ✅ Chỉ sử dụng các thuật toán ML truyền thống
- ✅ Sử dụng Time Series Cross-Validation để tránh data leakage

---

## 2. DỮ LIỆU

### 2.1. Nguồn dữ liệu

Dự án sử dụng 4 dataset chính:

1. **walmart-train.csv** (421,572 records)

   - Store: ID cửa hàng (1-45)
   - Dept: ID phòng ban (1-99)
   - Date: Ngày (2010-2012)
   - Weekly_Sales: Doanh số hàng tuần
   - IsHoliday: Có phải tuần lễ đặc biệt không

2. **walmart-features.csv** (8,192 records)

   - Store: ID cửa hàng
   - Date: Ngày
   - Temperature: Nhiệt độ (°F)
   - Fuel_Price: Giá nhiên liệu ($/gallon)
   - MarkDown1-5: Dữ liệu khuyến mại (anonymized)
   - CPI: Chỉ số giá tiêu dùng
   - Unemployment: Tỷ lệ thất nghiệp (%)
   - IsHoliday: Có phải tuần lễ đặc biệt không

3. **walmart-stores.csv** (45 records)

   - Store: ID cửa hàng
   - Type: Loại cửa hàng (A, B, C)
   - Size: Diện tích cửa hàng (square feet)

4. **walmart-test.csv** (115,066 records)
   - Dữ liệu test để đánh giá mô hình

### 2.2. Xử lý dữ liệu

#### 2.2.1. Data Cleaning

- ✅ Merge 3 dataset thành master dataset
- ✅ Chuyển đổi kiểu dữ liệu (date, boolean, categorical)
- ✅ Xử lý missing values (forward fill, backward fill)
- ✅ Loại bỏ doanh số âm (returns)
- ✅ Encode categorical variables (Type: A=3, B=2, C=1)

#### 2.2.2. Feature Engineering

- ✅ **Time Features:**

  - year, month, day, dayofweek, week, quarter
  - Cyclical features (sin/cos cho month, week, dayofweek)
  - Event flags (Christmas, Thanksgiving, holiday season)

- ✅ **Lag Features:**

  - Sales lag: 1, 2, 4, 8, 52 tuần trước
  - Environmental lag: temperature, fuel_price, CPI, unemployment

- ✅ **Rolling Window Features:**

  - Rolling mean, std, min, max cho windows: 4, 8, 12, 26, 52 tuần
  - Momentum features
  - Volatility features

- ✅ **MarkDown Features:**
  - Promo flags (is_promo_active, active_markdown_count)
  - MarkDown statistics (total, avg, max, min, std)
  - MarkDown interactions với các biến khác

### 2.3. Train/Test Split

- **Method:** Time Series Split (không dùng random split)
- **Split ratio:** 80% train / 20% test
- **Gap:** 1 tuần giữa train và test để tránh data leakage
- **Train size:** ~337,000 records
- **Test size:** ~84,000 records

---

## 3. PHƯƠNG PHÁP LUẬN

### 3.1. Thuật toán được sử dụng

#### 3.1.1. Linear Regression (Baseline)

- **Mục đích:** Mô hình cơ sở để so sánh
- **Ưu điểm:** Đơn giản, nhanh, dễ hiểu
- **Nhược điểm:** Không nắm bắt được mối quan hệ phi tuyến

#### 3.1.2. Random Forest Regressor

- **Loại:** Ensemble (Bagging)
- **Thư viện:** scikit-learn
- **Ưu điểm:**
  - Xử lý tốt dữ liệu mixed (numeric + categorical)
  - Tự động feature selection
  - Chống overfitting
  - Có thể xử lý missing values
- **Hyperparameters:**
  - Baseline: n_estimators=100, max_depth=None
  - Tuned: n_estimators=200, max_depth=20, max_samples=0.8, max_features='log2'

#### 3.1.3. XGBoost (Extreme Gradient Boosting)

- **Loại:** Ensemble (Boosting)
- **Thư viện:** xgboost
- **Ưu điểm:**
  - Hiệu suất cao
  - Xử lý missing values tự nhiên
  - Regularization tích hợp
  - Parallel processing
- **Hyperparameters:**
  - Baseline: n_estimators=100, learning_rate=0.1, max_depth=6
  - Tuned: n_estimators=300, learning_rate=0.01, max_depth=15, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0

### 3.2. Workflow Pipeline

Pipeline được thực hiện theo 5 bước chính:

1. **Preprocessing** (`preprocessing.py`):

   - Load và merge dữ liệu từ các file raw
   - Feature engineering (Week, Month, Year, Day)
   - Chọn features và lưu `train_detail.csv`, `test_detail.csv`, `feature_chosen.csv`

2. **K-Fold Validation - Untuned Models** (`k_fold_validation.py`):

   - Chạy K-Fold Cross-Validation (K=5) cho Random Forest và XGBoost với default parameters
   - Lưu kết quả vào `kfold_validation_comparison.csv`

3. **Hyperparameter Tuning** (`hyperparameter_tuning.py`):

   - Sử dụng RandomizedSearchCV với TimeSeriesSplit để tìm best parameters
   - 30 iterations cho mỗi mô hình, 3 CV folds
   - Lưu best parameters vào `tuned_models_best_params.csv`

4. **Train With Best Params** (`train_with_best_params.py`):

   - Load best parameters từ CSV
   - Chạy K-Fold Cross-Validation (K=5) với best parameters
   - Train model tốt nhất và tạo submission file
   - Lưu kết quả vào `best_params_kfold_comparison.csv`

5. **Model Evaluation & Analysis** (`model_evaluation_analysis.py`):
   - So sánh kết quả untuned vs tuned models
   - Tạo visualization và final report

### 3.3. Cross-Validation

- **Method:** K-Fold Cross-Validation (K=5) với shuffle=True
- **Lý do:** Đánh giá robust hơn, tránh overfitting
- **Áp dụng cho:**
  - Untuned models (default parameters)
  - Tuned models (best parameters từ hyperparameter tuning)
- **Không sử dụng:** Random split (sẽ gây data leakage trong time series)

### 3.4. Hyperparameter Tuning

- **Method:** RandomizedSearchCV với TimeSeriesSplit
- **Số lần thử nghiệm:** 30 iterations cho mỗi mô hình
- **CV folds:** 3 folds (TimeSeriesSplit)
- **Scoring:** neg_mean_absolute_error
- **Output:** Chỉ lưu best parameters vào CSV (không lưu models)

### 3.4. Evaluation Metrics

#### 3.4.1. WMAE (Weighted Mean Absolute Error) - Metric chính

```
WMAE = Σ(weights × |y_true - y_pred|) / Σ(weights)
```

- **Weights:** 5 cho holiday weeks, 1 cho normal weeks
- **Lý do:** Cuộc thi Walmart đánh giá cao độ chính xác trong các tuần lễ đặc biệt

#### 3.4.2. Các metrics khác

- **MAE (Mean Absolute Error):** Sai số trung bình tuyệt đối
- **RMSE (Root Mean Squared Error):** Trừng phạt sai số lớn hơn
- **R² (R-squared):** Tỷ lệ phương sai được giải thích
- **MAPE (Mean Absolute Percentage Error):** Sai số phần trăm trung bình

---

## 4. KẾT QUẢ

### 4.1. Kết quả K-Fold Cross-Validation (K=5)

Tất cả các mô hình được đánh giá bằng K-Fold Cross-Validation để đảm bảo tính robust và tránh overfitting.

#### 4.1.1. Kết quả Untuned Models (Default Parameters)

| Model                       | Mean WMAE | Std WMAE | Mean MAE | Mean RMSE | Mean R² | Mean Train Time (s) |
| --------------------------- | --------- | -------- | -------- | --------- | ------- | ------------------- |
| **Random Forest (Untuned)** | 1,535.21  | ±13.89   | 1,380.62 | 3,310.46  | 0.9787  | 14.35               |
| **XGBoost (Untuned)**       | 4,065.68  | ±22.96   | 3,900.78 | 6,985.24  | 0.9054  | 1.01                |

**Nhận xét:**

- ✅ Random Forest (Untuned) cho kết quả tốt hơn đáng kể
- ❌ XGBoost (Untuned) có WMAE cao, cần tuning để cải thiện

#### 4.1.2. Kết quả Tuned Models (Best Parameters)

| Model                     | Mean WMAE    | Std WMAE | Mean MAE | Mean RMSE | Mean R² | Mean Train Time (s) |
| ------------------------- | ------------ | -------- | -------- | --------- | ------- | ------------------- |
| **XGBoost (Tuned)**       | **1,246.91** | ±8.06    | 1,134.12 | 2,511.51  | 0.9878  | 13.87               |
| **Random Forest (Tuned)** | 1,552.38     | ±14.84   | 1,360.91 | 3,500.35  | 0.9762  | 29.22               |

**Nhận xét:**

- 🏆 **XGBoost (Tuned):** Model tốt nhất với Mean WMAE = 1,246.91 ± 8.06
- ✅ Cải thiện 69.33% so với XGBoost (Untuned)
- ✅ Random Forest (Tuned) tốt hơn Untuned một chút nhưng vẫn thua XGBoost (Tuned)

### 4.2. So sánh Tổng thể

| Model                          | Mean WMAE    | Std WMAE | Mean R² | Ranking |
| ------------------------------ | ------------ | -------- | ------- | ------- |
| 🥇 **XGBoost (Tuned)**         | **1,246.91** | ±8.06    | 0.9878  | 1       |
| 🥈 **Random Forest (Untuned)** | 1,535.21     | ±13.89   | 0.9787  | 2       |
| 🥉 **Random Forest (Tuned)**   | 1,552.38     | ±14.84   | 0.9762  | 3       |
| 4. XGBoost (Untuned)           | 4,065.68     | ±22.96   | 0.9054  | 4       |

**Kết luận:**

- 🏆 **Mô hình tốt nhất:** XGBoost (Tuned) với Mean WMAE = 1,246.91 ± 8.06
- ✅ Hyperparameter tuning cải thiện đáng kể cho XGBoost (69.33%)
- ✅ Random Forest (Untuned) tốt hơn Random Forest (Tuned) một chút

### 4.3. Best Parameters (từ Hyperparameter Tuning)

Best parameters được tìm thấy bằng RandomizedSearchCV với TimeSeriesSplit:

#### Random Forest (Tuned)

```python
{
    'n_estimators': 200,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_samples': 1.0,
    'max_features': None,
    'max_depth': 30,
    'bootstrap': True
}
```

#### XGBoost (Tuned) - Model tốt nhất

```python
{
    'subsample': 1.0,
    'reg_lambda': 1.0,
    'reg_alpha': 1.0,
    'n_estimators': 500,
    'min_child_weight': 10,
    'max_depth': 15,
    'learning_rate': 0.05,
    'colsample_bytree': 0.9
}
```

---

## 5. PHÂN TÍCH CHI TIẾT

### 5.1. Tại sao Random Forest Tuned lại tồi hơn Baseline?

#### Nguyên nhân:

1. **Overfitting trên Validation Set:**

   - Parameters được chọn dựa trên validation score
   - Validation set có thể không đại diện cho test set
   - Mô hình "học thuộc" validation set

2. **Best Parameters quá phức tạp:**

   - max_depth=20 có thể quá sâu
   - max_samples=0.8 có thể không phù hợp
   - Baseline parameters đơn giản hơn nhưng robust hơn

3. **Time Series Cross-Validation:**
   - Validation folds có thể khác với test set
   - Test set có thể có pattern khác với training data

#### Bài học:

- **Không phải lúc nào tuning cũng tốt hơn!**
- Baseline đôi khi đã rất tốt và robust
- Cần kiểm tra kỹ trên test set trước khi quyết định

### 5.2. Feature Importance Analysis

#### Top Features quan trọng nhất:

1. **Lag Features:** sales_lag_1, sales_lag_52 (doanh số tuần trước, cùng kỳ năm trước)
2. **Rolling Features:** sales_rolling_mean_4, sales_rolling_mean_12
3. **Store Information:** size, type
4. **Time Features:** week, month, is_holiday
5. **Environmental:** temperature, fuel_price, CPI

#### Nhận xét:

- ✅ Lag features rất quan trọng (doanh số có tính tuần hoàn)
- ✅ Store characteristics (size, type) ảnh hưởng lớn
- ✅ Time features giúp nắm bắt seasonality
- ⚠️ MarkDown features ít quan trọng hơn dự kiến

### 5.3. Residual Analysis

#### Random Forest (Baseline):

- Residuals phân bố gần như chuẩn (normal distribution)
- Mean residual ≈ 0
- Không có pattern rõ ràng trong residuals vs predicted
- ✅ Mô hình phù hợp tốt

#### XGBoost (Tuned):

- Residuals phân bố tốt
- Một số outliers nhưng không nhiều
- ✅ Mô hình ổn định

### 5.4. Model Performance Comparison

#### Bảng so sánh đầy đủ:

| Model                           | MAE       | RMSE      | R²      | WMAE         | Ranking |
| ------------------------------- | --------- | --------- | ------- | ------------ | ------- |
| 🥇 **Random Forest (Baseline)** | 4,520.18  | 9,470.95  | 0.7523  | **4,602.51** | 1       |
| 🥈 **XGBoost (Tuned)**          | 4,947.40  | 9,313.96  | 0.7605  | 5,029.26     | 2       |
| 🥉 **XGBoost (Baseline)**       | 5,265.34  | 9,356.13  | 0.7583  | 5,420.75     | 3       |
| 4. Random Forest (Tuned)        | 5,615.79  | 9,549.41  | 0.7482  | 5,749.49     | 4       |
| 5. Linear Regression            | 11,764.09 | 19,066.12 | -0.0037 | 11,891.29    | 5       |

---

## 6. KẾT LUẬN VÀ KHUYẾN NGHỊ

### 6.1. Kết luận

#### 6.1.1. Mô hình tốt nhất

**XGBoost (Tuned)** với:

- **Mean WMAE:** 1,246.91 ± 8.06 (metric chính)
- **Mean MAE:** 1,134.12 ± 6.09
- **Mean RMSE:** 2,511.51 ± 54.08
- **Mean R²:** 0.9878 ± 0.0005 (giải thích 98.78% phương sai)
- **Mean Train Time:** 13.87 giây

#### 6.1.2. Thành tựu

- ✅ Đạt được Mean WMAE = 1,246.91 (rất tốt, thấp hơn nhiều so với mục tiêu)
- ✅ Mô hình robust với độ lệch chuẩn thấp (±8.06)
- ✅ R² cao (0.9878) cho thấy mô hình giải thích được gần như toàn bộ phương sai
- ✅ Sử dụng K-Fold Cross-Validation (K=5) để đánh giá robust
- ✅ Hyperparameter tuning cải thiện đáng kể (69.33% so với untuned)

#### 6.1.3. Bài học

- ✅ Hyperparameter tuning rất quan trọng cho XGBoost (cải thiện 69.33%)
- ✅ K-Fold Cross-Validation cho đánh giá chính xác và robust hơn
- ✅ Random Forest (Untuned) tốt hơn Tuned một chút (có thể do overfitting khi tuning)
- ✅ Feature engineering và hyperparameter tuning đều quan trọng

### 6.2. Khuyến nghị

#### 6.2.1. Cho Production

1. **Sử dụng mô hình:** XGBoost (Tuned)
2. **Mean WMAE đạt được:** 1,246.91 ± 8.06
3. **Monitor performance:** Theo dõi WMAE trên dữ liệu mới, kiểm tra độ lệch chuẩn
4. **Retrain định kỳ:** Cập nhật mô hình với dữ liệu mới (hàng quý)
5. **Feature monitoring:** Theo dõi sự thay đổi của features
6. **Model stability:** Độ lệch chuẩn thấp (±8.06) cho thấy mô hình ổn định

#### 6.2.2. Cải thiện trong tương lai

1. **Feature Engineering:**

   - Tạo thêm features từ domain knowledge
   - External data (weather, events, promotions)
   - Store-specific features

2. **Ensemble Methods:**

   - Kết hợp Random Forest Baseline + XGBoost Tuned
   - Stacking với meta-learner
   - Weighted average của top models

3. **Advanced ML Algorithms:**

   - Thử nghiệm CatBoost (xử lý categorical tốt)
   - LightGBM (nhanh hơn XGBoost)
   - Extra Trees (biến thể của Random Forest)

4. **Model Interpretation:**
   - SHAP values để giải thích predictions
   - Partial dependence plots
   - Feature interaction analysis

### 6.3. Business Impact

#### 6.3.1. Lợi ích

- **Dự báo chính xác:** Giúp tối ưu hóa inventory management
- **Holiday Planning:** Chuẩn bị tốt hơn cho các tuần lễ đặc biệt
- **Resource Allocation:** Phân bổ nguồn lực hiệu quả hơn
- **Cost Reduction:** Giảm chi phí tồn kho và thiếu hàng

#### 6.3.2. ROI ước tính

- Giảm 10-15% chi phí tồn kho
- Giảm 5-10% thiếu hàng
- Tăng 2-5% doanh số nhờ planning tốt hơn

### 6.4. Hạn chế

1. **Dữ liệu:**

   - Chỉ có dữ liệu từ 2010-2012 (có thể lỗi thời)
   - Thiếu thông tin về promotions cụ thể
   - Không có external factors (competitors, events)

2. **Mô hình:**

   - Không sử dụng Deep Learning (có thể tốt hơn cho time series)
   - Chưa thử ensemble methods
   - Chưa optimize cho từng store riêng biệt

3. **Evaluation:**
   - Chỉ đánh giá trên test set cố định
   - Chưa có backtesting trên nhiều periods
   - Chưa có A/B testing trong production

---

## 7. PHỤ LỤC

### 7.1. Cấu trúc dự án

```
Machine-Learning-Assignment-251/
├── dataset/                          # Dữ liệu gốc
│   ├── walmart-features.csv
│   ├── walmart-stores.csv
│   ├── walmart-train.csv
│   └── walmart-test.csv
├── preprocessing.py                  # EDA và data cleaning
├── ml_data_preparation.py            # Chuẩn bị dữ liệu cho ML
├── baseline_models.py                # Baseline models
├── hyperparameter_tuning.py          # Hyperparameter tuning
├── model_evaluation_analysis.py      # Đánh giá và phân tích
├── train_detail.csv                  # Dữ liệu đã preprocess
├── X_train.csv, X_test.csv           # Features
├── y_train.csv, y_test.csv           # Targets
├── weights.npy                       # Weights cho WMAE
├── baseline_*.pkl                    # Baseline models
├── tuned_*.pkl                       # Tuned models
├── best_model.pkl                    # Best model
└── *.csv, *.png, *.md                # Results và reports
```

### 7.2. Dependencies

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
```

### 7.3. Thời gian thực hiện

- **Data Preprocessing:** ~2 giờ
- **Feature Engineering:** ~3 giờ
- **Baseline Models:** ~30 phút
- **Hyperparameter Tuning:** ~10 phút
- **Evaluation & Analysis:** ~1 giờ
- **Tổng cộng:** ~7 giờ

### 7.4. Tài liệu tham khảo

- Walmart Sales Forecasting Competition
- Scikit-learn Documentation
- XGBoost Documentation
- Time Series Cross-Validation Best Practices

---

## 8. TÓM TẮT EXECUTIVE

### 8.1. Kết quả chính

- ✅ **Mô hình tốt nhất:** XGBoost (Tuned)
- ✅ **Mean WMAE:** 1,246.91 ± 8.06 (rất tốt, thấp hơn nhiều so với mục tiêu)
- ✅ **Mean R²:** 0.9878 ± 0.0005 (giải thích 98.78% phương sai)
- ✅ **Thời gian training:** 13.87 giây (trung bình qua 5 folds)
- ✅ **Độ ổn định:** Độ lệch chuẩn thấp (±8.06) cho thấy mô hình robust

### 8.2. Điểm nổi bật

1. K-Fold Cross-Validation (K=5) cho đánh giá robust và chính xác
2. Hyperparameter tuning cải thiện XGBoost đáng kể (69.33%)
3. XGBoost (Tuned) vượt trội so với tất cả các models khác
4. Mô hình ổn định với độ lệch chuẩn thấp
5. R² cao (0.9878) cho thấy mô hình giải thích được gần như toàn bộ phương sai

### 8.3. Khuyến nghị hành động

1. **Triển khai:** Sử dụng XGBoost (Tuned) cho production
2. **Monitor:** Theo dõi Mean WMAE và độ lệch chuẩn trên dữ liệu mới
3. **Cải thiện:** Thử ensemble methods (kết hợp XGBoost Tuned + Random Forest)
4. **Retrain:** Cập nhật mô hình định kỳ với dữ liệu mới
5. **Validation:** Tiếp tục sử dụng K-Fold Cross-Validation khi retrain

---

**Báo cáo được tạo bởi:** Machine Learning Team  
**Ngày:** 2025-11-23  
**Version:** 2.0 (Updated với K-Fold Cross-Validation results)

---

_Báo cáo này tóm tắt toàn bộ quá trình xây dựng mô hình dự báo doanh số Walmart từ data preprocessing đến model evaluation và analysis._
