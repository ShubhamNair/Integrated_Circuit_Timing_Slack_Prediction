# Integrated_Circuit_Timing_Slack_Prediction

### Integrated Circuit Timing Slack Prediction

#### Overview
This repository contains code for training and evaluating regression models to predict timing slack values in integrated circuits. Timing slack is a critical metric in digital design, representing the amount of time a signal can be delayed without violating timing constraints.

#### Code Description
- **Data Loading and Preprocessing**: The code starts by loading data from CSV files, merging them, and preprocessing the features. It then filters the data based on specified slack ranges and conditions.
- **XGBoost Model Tuning with Optuna**: The first part of the code uses XGBoost to train a regression model with hyperparameters optimized using Optuna, a hyperparameter optimization framework. It defines a custom loss function to penalize positive slack values more heavily and evaluates the model's performance.
- **LightGBM Model Tuning with Optuna**: Similarly, it tunes a LightGBM model using Optuna, employing a custom asymmetric loss function. The performance of the tuned LightGBM model is evaluated.
- **Custom Neural Network Model**: Next, a custom neural network model is defined using PyTorch. The model incorporates a custom loss function that assigns different weights to negative slack values during training. The model is trained, evaluated, and its performance visualized.
- **Extra Credit: CatBoost Model**: Additionally, the code trains a CatBoost model, another gradient boosting algorithm, with a focus on optimizing performance for negative slack values. The model is evaluated, and its performance is compared with other models.

#### How to Use
1. Ensure Python and required packages are installed (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`, `optuna`, `matplotlib`, `catboost`).
2. Clone the repository and navigate to the directory.
3. Place your data files (`mod_labels.csv` and `mod_features.csv`) in the appropriate location.
4. Update the data file paths in the code if necessary.
5. Run the Python script.
6. Results including RMSE, R2 score, and correlation plots will be displayed for each model.

#### Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- torch
- optuna
- matplotlib
- catboost
