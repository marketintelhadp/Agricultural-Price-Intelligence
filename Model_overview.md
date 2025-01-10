# Project Description: Variety and Grade-Specific Price Forecasting

## Overview
This project aims to predict the average price of fruits using datasets divided by variety and grade. Each dataset corresponds to a specific fruit variety and grade (e.g., Grade A or Grade B). To achieve robust and accurate predictions, six different machine learning and statistical models are employed for each dataset.

## Models Used

### 1. Tree-Based Models
#### (a) Random Forest
- **Description**: Random Forest is an ensemble model that combines multiple decision trees. It captures complex interactions and handles the `Mask` column and other engineered features effectively.
- **Usage**: Suitable for datasets with non-linear relationships and missing data patterns.

#### (b) XGBoost
- **Description**: XGBoost is a gradient-boosting algorithm that builds trees sequentially to minimize errors. It excels in handling datasets with diverse features like `Mask` and engineered time-based indicators.
- **Usage**: Preferred for datasets with high variance or where strong predictive performance is required.

### 2. Seasonal Models
#### (a) SARIMA
- **Description**: Seasonal AutoRegressive Integrated Moving Average (SARIMA) captures seasonal patterns and trends explicitly. It is effective for datasets with strong seasonality in price or availability.
- **Usage**: Models datasets where availability is cyclic or tied to specific times of the year.

#### (b) Prophet
- **Description**: Prophet is a flexible forecasting model developed by Facebook. It handles missing data and incorporates holidays or seasonal patterns seamlessly.
- **Usage**: Ideal for datasets with irregular seasonal patterns or additional time-based influences.

### 3. Deep Learning Models
#### (a) LSTM
- **Description**: Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) designed to learn temporal dependencies. They work well for datasets with sequential patterns in availability and price.
- **Usage**: Employed for datasets where complex temporal relationships dominate.

#### (b) Transformer
- **Description**: Transformer-based models capture long-term dependencies and temporal dynamics. These models are state-of-the-art for time-series forecasting with high flexibility.
- **Usage**: Suitable for datasets requiring nuanced understanding of extended time relationships.

## Implementation Strategy
1. **Dataset Division**:
   - Each dataset corresponds to a unique variety and grade.
   - Mask column (0 for unavailable data, 1 for available data) is included as a feature.

2. **Feature Engineering**:
   - Time-based features: Month, quarter, and year.
   - Lagged features and rolling averages.
   - Encoding the `Mask` column to influence predictions.

3. **Model Training and Evaluation**:
   - Each model is trained on the respective dataset.
   - Metrics like MSE, MAE, and R-squared are used for evaluation.
   - Predictions vs actuals are visualized to understand performance.

4. **Comparison**:
   - Results from all six models are compared to identify the best performer for each dataset.

## Conclusion
This approach allows tailored predictions for each fruit variety and grade while accounting for their unique characteristics and availability patterns. By using diverse models, the project ensures flexibility and robustness in price forecasting.

