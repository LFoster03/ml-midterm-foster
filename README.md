# Mushroom Classification Project
**Author:** Lindsay Foster 
**Date:** November 2025  
**Course:** Applied Machine Learning – Midterm Project  

---

## Overview
This project applies **machine learning classification** techniques to predict whether a mushroom is **edible or poisonous** based on its physical characteristics.  
The analysis follows a structured workflow including data exploration, preprocessing, feature selection, model training, evaluation, and comparison of multiple classifiers.

## Project Files
- [Jupyter Notebook: C:\Repos\ml-midterm-foster\classification_foster.ipynb](https://github.com/LFoster03/ml-midterm-foster/blob/main/classification_foster.ipynb)
- [Peer Review: C:\Repos\ml-midterm-foster\peer_review.md](https://github.com/LFoster03/ml-midterm-foster/blob/main/peer_review.md)

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
```
pip install -r requirements.txt
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
- During data exploration, I identified key predictive features (odor and cap_look) and addressed missing values in stalk-root by imputing "unknown".  
- One-hot encoding transformed categorical variables into numeric form suitable for modeling. Creating combined features, like cap_look, allowed the model to detect patterns that individual features alone might not capture.

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

- I split data into a 70% training and a 30% test set, maintaining class proportions (stratify=y).
- I trained a Decision Tree classifier to predict whether a mushroom is poisonous or edible.

#### Split the Data
- The dataset was split into training (70%)** and test (30%) sets using train_test_split.  
- The split was stratified by the target variable poisonous to maintain class balance between edible and poisonous mushrooms.  
- A Decision Tree classifier was trained using Scikit-Learn's fit() method on the training data (X_train, y_train).  
- Input features used for training:  
  - cap_look (combined cap shape + color)  
  - odor  
  - stalk-root 

#### Evaluate Performance
- The model was evaluated using common classification metrics:
  - Accuracy – proportion of correctly classified mushrooms.  
  - Precision – proportion of predicted poisonous mushrooms that are truly poisonous.  
  - Recall – proportion of actual poisonous mushrooms that were correctly identified.  
  - F1-score – harmonic mean of precision and recall.  
- Additionally, I generated a confusion matrix and a full classification report to inspect detailed performance.

#### Results
> The Decision Tree model achieved high accuracy due to the strong predictive power of features.
> The confusion matrix highlights how well the model distinguishes between edible and poisonous mushrooms.  
> Overall, the model demonstrates the effectiveness of our feature selection and preprocessing steps.

#### Reflection: 
This model performed very well. The model performance metrics are all close to 99%.
- Accuracy: 0.9979 → about 99.8% of mushrooms were correctly classified.
- Precision: 0.9991 → of all mushrooms predicted as poisonous, 99.9% were actually poisonous. High precision means few false positives (few edible mushrooms were wrongly labeled as poisonous).
- Recall: 0.9966 → of all actual poisonous mushrooms, 99.66% were correctly identified. High recall means few false negatives (few poisonous mushrooms were missed).
- F1-score: 0.9979 → combines precision and recall. Close to 1.
- Confusion matrix shows:
  - TP (1262): edible mushrooms correctly classified
  - FP (1): edible mushrooms incorrectly classified as poisonous
  - FN (4): poisonous mushrooms incorrectly classified as edible
  - TN (1171): poisonous mushrooms correctly classified
- Only 5 misclassifications out of 2438 test samples.


### Improve the Model or Try Alternates (Implement a Second Option)
To evaluate whether a different modeling approach could improve performance, we trained a Random Forest classifier as an alternative to the Decision Tree.

#### Train an Alternative Model
- Random Forest was trained using 100 estimators (n_estimators=100) with default hyperparameters.  
- Same input features were used as in the Decision Tree:
  - cap_look (cap shape + color)  
  - odor  
  - stalk-root  

#### Compare Performance
| Metric      | Decision Tree | Random Forest |
|------------|---------------|---------------|
| Accuracy   | 0.9979        | 0.9984        |
| Precision  | 0.9991        | 0.9992        |
| Recall     | 0.9966        | 0.9974        |
| F1-score   | 0.9979        | 0.9983        |

- Confusion Matrices show very few misclassifications in both models, indicating that the dataset is highly predictable with these features.  
- Random Forest slightly improved metrics over the Decision Tree, demonstrating the benefit of ensemble methods for stability and minor performance gains.

#### Insights
> Both models perform exceptionally well due to the highly predictive nature of features like odor and cap_look.  
> Random Forest offers slightly better recall, reducing the chance of missing poisonous mushrooms.  
> The comparison supports that ensemble methods can provide small but meaningful improvements on top of simple tree-based models.  

### Summary of Findings
- The classification models were able to predict whether a mushroom is poisonous or edible with extremely high accuracy (>99%).  
- Key features driving predictions:
  - Odor – most predictive of poisonous mushrooms  
  - Cap_look – combination of cap shape and color  
  - Stalk-root – structural characteristic  
- Both Decision Tree and Random Forest performed almost perfectly due to the strong predictive features.  

#### Challenges Faced
- Handling the missing values in stalk-root required careful preprocessing; changed the missing values to "unknown".  
- Encoding categorical features into a numerical format with one-hot encoding increased the number of features, which could become cumbersome for larger datasets.  
- Ensuring proper stratified splits to maintain class balance required attention.  

#### Learning from the Project
- Learned how to preprocess categorical data, handle missing values, and perform one-hot encoding.  
- Gained experience in training, evaluating, and comparing multiple classifiers for a real-world dataset.  
- Developed skills in interpreting performance metrics like accuracy, precision, recall, F1-score, and confusion matrices.  
- Reinforced the importance of feature selection and thoughtful preprocessing to achieve high-performing machine learning models.
