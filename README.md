# Loan Approval Prediction

## Overview

This project focuses on predicting loan approval status based on various applicant features using machine learning techniques and Bayesian inference. The analysis is performed in a Jupyter notebook (`Loan_Prediction.ipynb`) that covers data exploration, preprocessing, model training, evaluation, and probabilistic reasoning.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Bayesian Inference](#bayesian-inference)
- [Results](#results)
- [Usage](#usage)

## Installation

To run this notebook, ensure you have Python installed along with the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Additionally, you'll need Jupyter Notebook:

```bash
pip install jupyter
```

## Dataset

The dataset used is `loan_data.csv`, which contains information about loan applicants including:

- Personal details: age, gender, education, income, employment experience
- Loan details: amount, intent, interest rate, percentage of income
- Credit history: credit score, history length, previous defaults
- Home ownership status
- Target variable: loan_status (0 for rejected, 1 for approved)

## Exploratory Data Analysis (EDA)

The notebook performs comprehensive EDA including:

- Basic data inspection (head, tail, shape, columns)
- Missing value analysis
- Duplicate detection
- Unique value counts
- Feature type classification (numerical vs categorical)
- Distribution plots for numerical features
- Count plots for categorical features
- Correlation heatmap for numerical variables

## Feature Engineering

Preprocessing steps include:

- Encoding categorical variables:
  - Gender: mapped to binary (male=1, female=0)
  - Previous defaults: Yes=1, No=0
  - Education: ordinal encoding (High School=0 to Doctorate=4)
  - Home ownership and loan intent: one-hot encoding
- Feature scaling using StandardScaler for numerical features

## Modeling

Multiple machine learning models are trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors
- Gradient Boosting Classifier

All models are trained on an 80-20 train-test split.

## Evaluation

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

Results are displayed for both training and test sets.

## Bayesian Inference

The notebook implements a custom Bayesian inference system:

- Builds a knowledge base with prior probabilities and feature statistics
- Performs probabilistic reasoning for loan approval prediction
- Optimizes decision threshold for best F1-score
- Compares Bayesian predictions with traditional ML models

## Results

The notebook provides comparative analysis of all models. Key findings include performance metrics and insights from the correlation analysis and Bayesian reasoning.

## Usage

1. Clone or download the repository
2. Ensure `loan_data.csv` is in the same directory as the notebook
3. Open `Loan_Prediction.ipynb` in Jupyter Notebook
4. Run the cells sequentially to reproduce the analysis
