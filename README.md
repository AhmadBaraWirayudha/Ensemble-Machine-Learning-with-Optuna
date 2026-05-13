# Ensemble Machine Learning with Optuna for Surface Roughness Prediction

An industrial machine learning project for predicting surface roughness (`Ra`) in CNC milling of S45C steel using ensemble learning, Gaussian Process Regression (GPR), Support Vector Regression (SVR), and Optuna-based hyperparameter optimization.

---

# Overview

Surface roughness is one of the most important quality indicators in machining processes. Traditional trial-and-error parameter tuning is expensive, time-consuming, and inconsistent.

This project develops a machine learning pipeline capable of predicting surface roughness based on machining parameters:

- Cutting speed (`Vc`)
- Feed per tooth (`Fz`)
- Axial depth of cut (`ap`)

The system combines:

- Support Vector Regression (SVR)
- Gaussian Process Regression (GPR)
- Weighted ensemble learning
- Stacking ensemble learning
- Automated hyperparameter tuning using Optuna

The project also includes:

- Feature engineering
- Cross-validation
- Visualization
- Feature importance analysis
- Interactive desktop GUI

---

# Features

- Industrial machining dataset processing
- Advanced feature engineering
- SVR with polynomial feature expansion
- Gaussian Process Regression with custom kernels
- Optuna hyperparameter optimization
- Weighted ensemble optimization
- Stacking meta-learner
- Automated metrics evaluation
- Feature importance analysis
- Automated plot generation
- Tkinter-based GUI dashboard

---

# Project Structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── features.py
│   ├── models.py
│   ├── optuna_tuning.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── train.py
│
├── data/
│   └── Sheet2.csv
│
├── models/
│
├── notebooks/
│
├── configs/
│
├── reports/
│   ├── metrics/
│   ├── figures/
│   └── feature_importance/
│
├── app/
│   └── gui.py
│
└── tests/
```

---

# Machine Learning Pipeline

## 1. Data Loading

The dataset is loaded and cleaned automatically:

- Column normalization
- Missing value removal
- Numeric conversion
- Feature selection

Input variables:

| Feature | Description |
|---|---|
| `Vc` | Cutting speed |
| `Fz` | Feed per tooth |
| `ap` | Axial depth of cut |

Target variable:

| Target | Description |
|---|---|
| `Ra` | Surface roughness |

---

# 2. Feature Engineering

Additional engineered features are generated:

- Squared terms
- Interaction terms
- Ratio features
- Logarithmic transformations

Examples:

- `Vc²`
- `Fz²`
- `Vc × Fz`
- `Vc / ap`

---

# 3. Model Development

## Support Vector Regression (SVR)

Pipeline:

- PolynomialFeatures
- RobustScaler
- PowerTransformer
- RBF Kernel SVR

## Gaussian Process Regression (GPR)

Custom kernel combination:

- RBF Kernel
- Rational Quadratic Kernel
- White Noise Kernel

---

# 4. Hyperparameter Optimization

Hyperparameters are optimized using Optuna with TPE sampler.

Optimized parameters include:

## SVR

- `C`
- `epsilon`
- `gamma`
- `poly_degree`

## GPR

- `amplitude`
- `length_scale`
- `alpha`
- `noise`

---

# 5. Ensemble Learning

Two ensemble strategies are implemented:

## Weighted Ensemble

Combines:

```python
Prediction = α(GPR) + (1 - α)(SVR)
```

The optimal ensemble weight is tuned using Optuna.

## Stacking Ensemble

A RidgeCV meta-learner combines:

- SVR predictions
- GPR predictions

---

# 6. Model Evaluation

Evaluation metrics:

- MSE
- RMSE
- MAE
- MAPE
- MBE
- R² Score

Cross-validation:

- Stratified 5-Fold Cross Validation

---

# 7. Feature Importance Analysis

Two methods are used:

## SVR

- Permutation Importance

## GPR

- Inverse Length-Scale Sensitivity

---

# Visualization Outputs

The pipeline automatically generates:

- Surface roughness distribution
- Scatter plots
- Actual vs Predicted plots
- Model comparison plots
- Feature importance plots
- Machining parameter sensitivity plots

---

# GUI Dashboard

The project includes a Tkinter-based GUI application with:

- Real-time logging
- Runtime monitoring
- Metrics viewer
- Plot viewer
- Optimization progress display

---

# Installation

## Clone Repository

```bash
git clone https://github.com/AhmadBaraWirayudha/Ensemble-Machine-Learning-with-Optuna.git

cd Ensemble-Machine-Learning-with-Optuna
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Project

## Run Training Pipeline

```bash
python -m src.train
```

## Run GUI Application

```bash
python app/gui.py
```

---

# Example Outputs

Generated outputs:

- Metrics CSV
- Feature importance CSV
- Prediction plots
- Sensitivity analysis plots

Saved automatically inside:

```text
reports/
```

---

# Industrial Relevance

This project is relevant for:

- CNC machining optimization
- Manufacturing quality prediction
- Smart manufacturing
- Predictive quality control
- Digital manufacturing systems
- AI-assisted machining optimization

---

# Future Improvements

Possible future extensions:

- Real-time CNC monitoring
- IoT integration
- Deep learning models
- Bayesian optimization
- Explainable AI (XAI)
- Web-based dashboard deployment
- CAD/CAM integration

---

# Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn |
| Optimization | Optuna |
| Visualization | Matplotlib, Seaborn |
| GUI | Tkinter |
| Data | Pandas, NumPy |

---

# License

This project is licensed under the Apache License 2.0.

See the `LICENSE` file for details.

---

# Author

Ahmad Bara Wirayudha

Mechanical Engineering × Machine Learning × Industrial AI
