# Diabetes Prediction Model

![CI](https://github.com/aayxshkumar/diabetes-prediction/actions/workflows/ci.yml/badge.svg)

A Jupyter notebook (`Diabetes.ipynb`) that compares several classifiers for predicting diabetes on the Pima Indians Diabetes data.

## Data

`diabetes.csv` is included: 768 rows with eight measurements (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and an `Outcome` label. It was taken from the Kaggle dataset `mathchi/diabetes-data-set`, whose page (https://www.kaggle.com/datasets/mathchi/diabetes-data-set) no longer resolves.

## What the notebook does

- Treats zeros in the measurement columns as missing values and fills Glucose, BloodPressure, SkinThickness, Insulin and BMI with each column's mode.
- Plots feature histograms by outcome and a correlation heatmap.
- Uses Glucose, BMI and Age as features.
- Compares cross-validated accuracy of a random forest, XGBoost, k-nearest neighbours, SVMs (RBF and linear kernels), Gaussian naive Bayes and a small Keras neural network, then plots the results.

The saved output of the final comparison cell is:

```
{'RF': 0.7708988380041013,
 'XGB': 0.75,
 'KNN': 0.7422544775485951,
 'SVM': 0.7656565656565657,
 'NB': 0.7708333333333334,
 'ANN': 0.7683133859604447}
```

These outputs come from the original run on Python 3.6. The execution counts show the cells were not run strictly top to bottom.

## Requirements

- Python 3.12 (the version CI uses) and Jupyter
- The libraries in `requirements.txt`, pinned to current releases
- On macOS, xgboost needs the OpenMP runtime (`brew install libomp`)

## Setup and usage

```bash
git clone https://github.com/aayxshkumar/diabetes-prediction.git
cd diabetes-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open `Diabetes.ipynb` in Jupyter. It reads `diabetes.csv` from the working directory.

## Running it today

The notebook has not been re-run with the pinned versions. These cells use APIs that current releases changed or removed:

- `from keras.wrappers.scikit_learn import KerasClassifier` fails on Keras 3. The `scikeras` package provides a replacement wrapper.
- `y.as_matrix()` was removed from pandas; use `y.values`.
- `train['BMI'].fillna(..., inplace=True)` and the similar lines no longer update `train` under pandas 3 copy-on-write; assign the result back instead (`train['BMI'] = train['BMI'].fillna(...)`).
- `plt.bar(..., color="rgbkymc")` raises `ValueError` in current matplotlib; pass a list of colours.

## CI

CI validates the notebook JSON with nbformat, checks the shape of `diabetes.csv` and checks that the notebook's libraries import (xgboost and the removed Keras wrapper excluded). It does not run the notebook.
