# Predicting Severe Building Damage from Earthquake Data Using Machine Learning

## Project Overview
This project predicts whether a building is likely to experience severe earthquake damage using machine learning techniques and structural building information. The goal of the project is to identify important building characteristics associated with high earthquake damage and compare multiple classification models to determine the best-performing approach.

The project uses earthquake assessment data containing information about building structure, materials, location, and post-earthquake damage observations. A binary target variable called `severe_damage` was created, where buildings with Grade 4 or Grade 5 damage were classified as severe damage.

---

## Dataset
The project uses the following datasets:

- `building_damage_assessment.csv`
- `building_structure.csv`

The datasets include:
- Building structural characteristics
- Construction materials
- Geographic identifiers
- Repair information
- Earthquake damage assessments

To improve model quality, duplicate columns, leakage variables, and columns with excessive missing values were removed during the data wrangling process.

---

## Main Methods and Models

### Data Wrangling
A custom `wrangle()` function was created to:
- Import datasets
- Merge datasets using `building_id`
- Remove duplicate columns
- Create the binary target variable `severe_damage`
- Remove leakage variables that would not be available before prediction
- Remove columns with more than 50% missing values
- Reduce memory usage by sampling 100,000 observations

### Exploratory Data Analysis (EDA)
Exploratory analysis was performed to understand:
- Distribution of severe damage
- Missing values
- Building characteristics
- Feature relationships

Visualizations were created using Matplotlib and Seaborn.

### Machine Learning Models
Several classification models were trained and compared:

1. Baseline Majority Class Model
2. Logistic Regression
3. Decision Tree Classifier
4. XGBoost Classifier

### Preprocessing Pipeline
A machine learning pipeline was implemented using Scikit-learn:
- Numerical missing values were imputed using `SimpleImputer`
- Categorical variables were encoded using `OneHotEncoder`
- Models were integrated into a preprocessing pipeline

### Hyperparameter Tuning
The XGBoost classifier was optimized using `RandomizedSearchCV` to improve predictive performance and reduce overfitting. Hyperparameters tuned included:
- Number of estimators
- Maximum tree depth
- Learning rate
- Subsample ratio

---

## Key Findings and Results

The tuned XGBoost classifier achieved the strongest performance among all models tested.

### Final Model Performance
- Accuracy: approximately **90.6%**
- ROC-AUC Score: approximately **0.97**

### Model Comparison
| Model | Accuracy |
|---|---|
| Baseline Majority Class | ~60% |
| Logistic Regression | ~60% |
| Decision Tree | ~86.7% |
| Tuned XGBoost | ~90.6% |

The results demonstrate that ensemble boosting methods significantly improved earthquake damage prediction performance compared to simpler baseline models.

---
