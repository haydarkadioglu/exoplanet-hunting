# Exoplanet Detection Models

This directory contains trained machine learning models for exoplanet detection across different space missions.

## Directory Structure

### K2 Mission Models (`k2/`)
- `k2_model.onnx` - ONNX format model for web deployment
- `k2_scaler.onnx` - ONNX format feature scaler
- `k2_final_model_gradient_boosting.joblib` - Trained Gradient Boosting model
- `k2_scaler.joblib` - Feature scaler for preprocessing
- `k2_feature_selector.joblib` - Feature selection tool
- `k2_selected_features.joblib` - List of selected features
- `k2_model_results.joblib` - Model performance metrics

### Kepler Mission Models (`kepler/`)
- `kepler_model.onnx` - ONNX format model for web deployment
- `kepler_scaler.onnx` - ONNX format feature scaler
- `kepler_final_model_lightgbm.joblib` - Trained LightGBM model
- `kepler_scaler.joblib` - Feature scaler for preprocessing
- `kepler_feature_selector.joblib` - Feature selection tool
- `kepler_selected_features.joblib` - List of selected features
- `kepler_model_results.joblib` - Model performance metrics

### TESS Mission Models (`tess/`)
- `tess_model.onnx` - ONNX format model for web deployment
- `tess_scaler.onnx` - ONNX format feature scaler
- `scaler.joblib` - Feature scaler for preprocessing
- `imputer.joblib` - Missing value imputer
- `label_encoder.joblib` - Target label encoder
- `metadata.json` - Model metadata and configuration
- `class_mapping.json` - Class label mappings
- `column_mapping.json` - Feature column mappings
- `data_info.json` - Dataset information
- Training data splits:
  - `X_features.csv` - Feature names and information
  - `X_train_scaled.csv` - Scaled training features
  - `X_test_scaled.csv` - Scaled testing features
  - `y_train.csv` - Training targets
  - `y_test.csv` - Testing targets
  - `y_train_encoded.csv` - Encoded training targets
  - `y_test_encoded.csv` - Encoded testing targets
  - `y_target_3class.csv` - 3-class target mapping

## Model Performance

Each model is trained for 3-class exoplanet classification:
- **Class 0**: Planet Candidates
- **Class 1**: Confirmed Planets
- **Class 2**: False Positives

## Usage

### For Web Applications (ONNX Models)
Use the `.onnx` files for deployment in web browsers or JavaScript applications.

### For Python Applications (Joblib Models)
Use the `.joblib` files for deployment in Python environments.

```python
import joblib

# Load model and preprocessors
model = joblib.load('path/to/model.joblib')
scaler = joblib.load('path/to/scaler.joblib')
feature_selector = joblib.load('path/to/feature_selector.joblib')

# Make predictions
scaled_features = scaler.transform(raw_features)
predictions = model.predict(scaled_features)
```

## Model Training

Models were trained using comprehensive machine learning pipelines including:
- Data preprocessing and cleaning
- Feature engineering and selection
- Multiple algorithm comparison
- Hyperparameter optimization
- Cross-validation
- Performance evaluation

Training notebooks are available in the `main/` directory of the repository.
