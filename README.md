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