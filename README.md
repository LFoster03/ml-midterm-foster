# Mushroom Classification Project
**Author:** Lindsay Foster 
**Date:** November 2025  
**Course:** Applied Machine Learning – Midterm Project  

---

## Overview
This project applies **machine learning classification** techniques to predict whether a mushroom is **edible or poisonous** based on its physical characteristics.  
The analysis follows a structured workflow including data exploration, preprocessing, feature selection, model training, evaluation, and comparison of multiple classifiers.

## Imports Used
```from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
```
---

## Dataset
**Source:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)  
**Description:**  
The dataset contains 8124 mushroom samples described by 22 categorical features such as cap shape, odor, gill color, and habitat.  
The **target variable** (`class`) indicates whether a mushroom is **edible (e)** or **poisonous (p)**.

**Example Features:**
- `cap-shape`
- `cap-surface`
- `odor`
- `gill-color`
- `habitat`
- `class` (target)

---

## Project Goals
- Load, inspect, and clean the mushroom dataset.  
- Explore distributions and feature importance.  
- Encode categorical variables for modeling.  
- Train and evaluate multiple classification algorithms:
  - Decision Tree  
  - Random Forest  
  - Logistic Regression  
- Compare performance using metrics such as:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  

---

## Setup & Run Instructions

### Clone the Repository
```bash
git clone https://github.com/yourusername/mushroom-classification.git
cd mushroom-classification
```
### Create and Activate Virtual Environment
```python -m venv .venv
.venv\Scripts\activate
```
### Install Dependencies
```pip install -r requirements.txt
```
### Launch Notebook: classification_foster

### Load and Inspect Data
- Loaded dataset into pandas DataFrames for features (X) and target (y).
- Displayed the first 10 rows of the dataset and verified feature names and target variable.
- Check for missing values and duplicates.
- ***Findings*** All columns have no missing values except stalk-root, which has 2,480 missing entries (~30% of rows). There are no duplicate rows.
- Fill missing 'stalk-root' values with 'unknown'.
- ***Summary Statistics*** Most features are categorical with a small number of unique values. Target variable class is balanced between edible and poisonous mushrooms.

### Data Exploration and Preparation

#### Patterns and Anomalies Observed
- Most features in the Mushroom dataset are categorical, with a small set of unique values per feature.  
- The target variable (poisonous) is roughly balanced, which is ideal for classification tasks.  
- Some categories are rare (e.g., certain odor or cap-color values), which could be highly predictive.  
- The stalk-root column had 30% missing values, which needed attention to avoid biasing the model.

#### Features That Stand Out
- odor: Certain odors strongly correlate with poisonous mushrooms and are likely key predictors.  
- cap_look : A new feature created by combining cap-shape and cap-color, capturing patterns not evident from individual features.  
- stalk-root : After imputing missing values, this feature adds predictive information.

#### Prep Steps
1. **Handling Missing Values**
   - Imputed missing values in stalk-root with "unknown" to preserve all rows.  
2. **Feature Engineering**
   - Created cap_look by combining cap-shape and cap-color to summarize the mushroom’s overall appearance.  
3. **Encoding Categorical Variables**
   - Applied one-hot encoding using pd.get_dummies() to convert categorical features into numeric columns.  
4. **Scaling / Normalization (Optional)**
   - Not necessary for tree-based models (Decision Tree, Random Forest). Required only for models sensitive to feature scale, such as Logistic Regression or MLP.

#### Feature Selection and Modification
- Selected key features likely to contribute to predictive power: cap_look, odor, stalk-root.  
- One-hot encoding expanded categorical features into binary columns for better interpretability and modeling.  
- Combined features like cap_look were created to capture interactions and patterns that improve model performance.

#### Reflection
> During data exploration, I identified key predictive features (odor and cap_look) and addressed missing values in stalk-root by imputing "unknown".  
> One-hot encoding transformed categorical variables into numeric form suitable for modeling. Creating combined features, like cap_look, allowed the model to detect patterns that individual features alone might not capture.

### Feature Selection and Justification
- Features: cap_look, odor, stalk-root

- Target: poisonous (encoded as 0/1)

- Rationale: These features are meaningful, interpretable, and likely to improve model performance.

- Preprocessing: Combined features, imputed missing values, and one-hot encoding applied.

* I selected cap_look, odor, and stalk-root as input features because they are directly related to mushroom characteristics that indicate edibility or toxicity.

- I created a heatmap to discern a relationship between cap_look and odor to see if there were any crossovers. All mushrooms with odor n of all cap looks were *not* poisonous.

- Selected features: 'cap_look', 'odor', 'stalk-root' for predictive power.
- 'odor' is highly indicative of poisonous mushrooms.
- 'cap_look' combines cap shape and color to capture visual patterns.
- 'stalk-root' provides structural information and missing values were imputed as 'unknown'.
- One-hot encoding converts categorical data into numeric form suitable for modeling.
- These features are expected to improve prediction accuracy and model interpretability.

### Train a Model

Split data into 70% training and 30% test sets, maintaining class proportions (stratify=y).

Trains a Decision Tree classifier using fit().

Predicts on the test set.

Evaluates performance with:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Full classification report