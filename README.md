# Zillow Assignment Kaggle

A machine learning project for predicting real estate prices using the Zillow dataset. This repository contains multiple regression models, feature engineering, and comprehensive analysis for the Kaggle Zillow Prize competition.

## Project Overview

This project implements various machine learning algorithms to predict the log error of Zestimate (Zillow's automated home valuation) using property features and transaction data. The goal is to minimize the Mean Absolute Error (MAE) between predicted and actual log errors.

## Dataset

The project uses Zillow's dataset containing:
- **Properties 2016/2017**: Property features including square footage, number of rooms, location, etc.
- **Training Data 2016/2017**: Transaction records with actual sale prices and dates
- **Target Variable**: Log error between Zestimate and actual sale price

## Repository Structure

```
├── assign.py                    # Main assignment implementation
├── kaggle_submission.py         # Kaggle submission script
├── requirements.txt             # Python dependencies
├── test_gpu.py                 # GPU testing utilities
├── properties_2016.csv         # Property features for 2016
├── properties_2017.csv         # Property features for 2017
├── train_2016_v2.csv           # Training data for 2016
├── train_2017.csv              # Training data for 2017
├── models/                     # Trained model files
│   ├── elasticnet_model.joblib
│   ├── glm_model.pkl
│   ├── kaggle_rf_model.joblib
│   ├── lasso_model.joblib
│   ├── lgb_fallback_model.joblib
│   ├── lgb_model.txt
│   ├── ols_model.pkl
│   ├── rf_model.joblib
│   ├── ridge_model.joblib
│   ├── rlm_model.pkl
│   ├── scaler.joblib
│   ├── xgb_fallback_model.joblib
│   └── xgb_model.json
└── outputs/                    # Results and visualizations
    ├── metrics.json
    ├── submission.csv
    └── plots/
        ├── logerror_dist.png
        ├── pred_vs_actual_RandomForest.png
        ├── region_heatmap_regionidzip.png
        ├── residuals_RandomForest.png
        └── rf_feature_importance.png
```

## Models Implemented

### Traditional Machine Learning Models
- **Linear Regression (OLS)**: Baseline linear model
- **Ridge Regression**: L2 regularized linear regression
- **Lasso Regression**: L1 regularized linear regression
- **Elastic Net**: Combined L1/L2 regularization
- **Random Forest**: Ensemble of decision trees
- **Robust Linear Model (RLM)**: Outlier-resistant regression

### Advanced Models
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Microsoft's gradient boosting framework
- **Generalized Linear Model (GLM)**: Statistical modeling approach

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/spyseecoder/Zillow-Assignment-Kaggle.git
cd Zillow-Assignment-Kaggle
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Assignment
```bash
python assign.py
```
This will:
- Load and preprocess the data
- Train multiple models
- Generate evaluation metrics
- Create visualizations
- Save trained models

### Generating Kaggle Submission
```bash
python kaggle_submission.py
```
This creates a submission file for the Kaggle competition.

### Testing GPU Support
```bash
python test_gpu.py
```
Checks if GPU acceleration is available for XGBoost and LightGBM.

## Feature Engineering

The project includes comprehensive feature engineering:
- **Missing Value Handling**: Imputation strategies for different feature types
- **Categorical Encoding**: Label encoding and one-hot encoding
- **Feature Scaling**: StandardScaler for linear models
- **Date Features**: Extraction of temporal patterns from transaction dates
- **Geographical Features**: Location-based feature engineering

## Model Performance

All models are evaluated using:
- **Mean Absolute Error (MAE)**: Primary metric for Kaggle competition
- **Root Mean Square Error (RMSE)**: Secondary evaluation metric
- **R-squared**: Coefficient of determination
- **Cross-validation**: 5-fold cross-validation for robust evaluation

Results are saved in `outputs/metrics.json` and visualizations in `outputs/plots/`.

## Results and Outputs

- **Submission File**: `outputs/submission.csv` ready for Kaggle upload
- **Model Metrics**: Comprehensive evaluation metrics in JSON format
- **Visualizations**: 
  - Feature importance plots
  - Prediction vs actual scatter plots
  - Residual analysis
  - Geographic heatmaps
  - Error distribution plots

## Key Features

- **Ensemble Methods**: Multiple model approaches for robust predictions
- **Automated Model Selection**: Performance-based model ranking
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Production Ready**: Saved models for deployment
- **Reproducible Results**: Fixed random seeds for consistency

## Dependencies

Key libraries used:
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning models and preprocessing
- xgboost, lightgbm: Advanced gradient boosting
- matplotlib, seaborn: Data visualization
- statsmodels: Statistical modeling

See `requirements.txt` for complete list of dependencies.

## Contributing

This project is for taking part in a Kaggle competition. If there are any issues or improvements, please open an issue and I will assign them - contributions and help are welcome!

## License

This project is for participating in the Kaggle Zillow Prize competition.