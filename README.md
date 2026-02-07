# Diabetes-Prediction-Using-ML-with-explanability
A machine learning project for predicting diabetes using K-Nearest Neighbors and Logistic Regression, with integrated explainability features for medical interpretation.

## Overview

This project implements interpretable machine learning models to predict diabetes diagnosis based on patient health metrics. The models are evaluated using multiple metrics and incorporate SHAP analysis to explain individual predictions, making them suitable for clinical decision support.

## Models

- **K-Nearest Neighbors (KNN)** - Optimized using GridSearchCV
- **Logistic Regression** - With L1/L2 regularization

## Features

### Model Evaluation
- Accuracy Score
- ROC AUC Curve
- 10-Fold Cross Validation
- Confusion Matrix
- Precision, Recall, F1-Score

### Explainability
- SHAP values for feature importance
- Individual prediction explanations
- Risk factor identification
- Feature contribution analysis

### Data Preprocessing
- Zero value imputation using median by outcome group
- Feature scaling with StandardScaler
- Stratified train-test split

## Dataset

The project uses the Pima Indians Diabetes Dataset with the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

**Target Variable**: Outcome (0 = No Diabetes, 1 = Diabetes)

## Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## Usage

### Training Models

```python
# Load and preprocess data
df = pd.read_csv('diabetes.csv')

# Train models (see notebook for full implementation)
# Models are saved as pickle files
```

### Making Predictions

```python
import pickle

# Load saved models
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
    
with open('logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Predict for new patient
new_patient = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
new_patient_scaled = scaler.transform(new_patient)
prediction = lr_model.predict(new_patient_scaled)
```

### Getting Explanations

The notebook includes functions to explain predictions:

```python
predict_and_explain_new_patient(
    pregnancies=6,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=0,
    bmi=33.6,
    diabetes_pedigree=0.627,
    age=50
)
```

This provides:
- Model predictions with confidence scores
- Clinical risk assessment
- SHAP-based feature contributions
- Medical interpretation

## Key Results

The models demonstrate strong predictive performance:
- High accuracy scores
- Excellent ROC AUC values
- Consistent cross-validation results
- Clear feature importance rankings

Most influential features:
1. Glucose levels
2. BMI
3. Age
4. Diabetes Pedigree Function

## Project Structure

```
├── diabetes.csv                              # Dataset
├── diabetes-prediction-with-explainability.ipynb  # Main notebook
├── knn_model.pkl                            # Trained KNN model
├── logistic_regression_model.pkl            # Trained LR model
├── scaler.pkl                               # Feature scaler
└── README.md                                # Documentation
```

## Visualizations

The project generates comprehensive visualizations:
- Feature distributions by outcome
- Correlation heatmap
- Model accuracy comparisons
- Confusion matrices
- ROC curves
- SHAP summary plots
- Cross-validation results

## Clinical Application

This model is designed to assist healthcare professionals by:
- Providing probability scores for diabetes risk
- Explaining which factors contribute to each prediction
- Identifying key risk factors for individual patients
- Supporting clinical decision-making with interpretable results

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap

## Limitations

- Model trained on specific population (Pima Indians)
- Requires validation on diverse patient populations
- Not a substitute for professional medical diagnosis
- Zero values in dataset required imputation

## Future Work

- Ensemble methods for improved accuracy
- Integration with electronic health records
- Real-time risk monitoring
- Web application deployment
- Extended validation on diverse datasets

## License

This project is for educational and research purposes only. Always consult qualified healthcare professionals for medical decisions.

## Contributing

Contributions are welcome. Please ensure all changes include appropriate documentation and maintain code quality standards.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.
