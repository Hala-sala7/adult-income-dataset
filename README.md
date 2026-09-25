# Adult Income Classification

**Author:** Hala Salah

## Overview

This project predicts whether a person earns more than $50K per year using U.S. census data, and compares machine learning and deep learning models.

**Business case:** A company wants to target high-income customers for a marketing campaign. Targeting people who are not actually high-income wastes resources, so **precision for the `>50K` class** is the most important metric: when the model predicts that someone earns more than $50K, it should be right as often as possible.

## Data

- **Source:** [Adult Income Dataset (UCI / Kaggle)](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)
- **Size:** 48,842 records and 15 columns
- **Target:** `income` (`<=50K` or `>50K`)
- **Class balance:** imbalanced, with about 76% earning `<=50K` and 24% earning `>50K`

## Data Cleaning

- Removed 52 duplicate rows
- Found 6,456 hidden missing values stored as `"?"` in `workclass`, `occupation`, and `native-country`, and converted them to real missing values
- Dropped the `education` column because it duplicates `educational-num`
- Split the data into training and test sets **before** imputation, and fitted all imputation and preprocessing on the training data only to prevent data leakage

## Exploratory Analysis

Key patterns found in the data:
- People with any capital gain are far more likely to earn more than $50K.
- The share of high earners rises steadily with education level.
- High earners work more hours on average (about 45 vs. 39 per week) and are older (about 44 vs. 37 years).
- Executive-managerial and professional-specialty occupations have the highest share of high earners.

The project also includes explanatory visualizations designed for a non-technical audience, each focused on one clear trend.

<img width="690" height="490" alt="download (3)" src="https://github.com/user-attachments/assets/3dc203f0-9fdf-44c7-bdba-112c244eb8cb" />

About 62% of people with capital gains earn more than $50K, compared to only about 20% of people without capital gains.

<img width="989" height="490" alt="download (4)" src="https://github.com/user-attachments/assets/a3a12011-7a2d-40a3-af1e-9eeb2774cf6b" />

## Modeling

### Part 1: Random Forest baseline
A Random Forest with `class_weight='balanced'`, built in a scikit-learn Pipeline with a ColumnTransformer (scaling and one-hot encoding).

### Part 2: Feature engineering and selection
- **PCA:** added 3 principal components (fitted on training data only), which explained about 43% of the variance. They did not improve performance, since they only summarized features the model already had.
- **Feature selection:** used `SelectFromModel` to reduce the feature space from **91 to 17 features**, which slightly improved accuracy and precision while making the model simpler.

### Part 3: Neural networks
- A baseline neural network in TensorFlow/Keras with early stopping
- A tuned neural network using **Keras Tuner**, optimized directly for validation precision

## Results (test set, `>50K` class)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (baseline) | 0.82 | 0.58 | 0.87 | 0.69 |
| Random Forest (feature selection) | 0.83 | 0.60 | 0.81 | 0.69 |
| Neural Network (baseline) | 0.85 | 0.77 | 0.55 | 0.64 |
| **Neural Network (tuned for precision)** | **0.85** | **0.81** | 0.47 | 0.59 |

**Selected model:** the tuned neural network, because it has the highest precision (0.81), which matches the business case. When it predicts that someone earns more than $50K, it is correct 81% of the time.

**Trade-off:** its recall is lower (0.47), meaning it misses more than half of the actual high earners. This is acceptable for this business case, where avoiding wrong targets matters more than reaching every high earner. If the business priority changed to reaching as many high earners as possible, the Random Forest would be the better choice.

## Feature Importance

Permutation importance (scored on `>50K` precision) showed that the strongest predictors of high income are:
1. **Capital gain**, the most important feature in every model
2. **Education level**
3. **Occupation**, especially executive-managerial roles

<img width="989" height="590" alt="download (5)" src="https://github.com/user-attachments/assets/24ea992b-063a-4799-b625-318b891624dd" />
## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow/Keras, Keras Tuner

## How to Run

Open the notebook in Google Colab using the badge at the top of the notebook. Note that neural network results can vary slightly between runs because of random weight initialization.
