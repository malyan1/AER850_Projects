"""
@author: Muhammad Alyan 501096627
AER850 Project 1
Section 1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib

# %%
# 2.1: Data Processing
data = pd.read_csv("project1_data.csv") # Read in data
data = data.dropna().reset_index(drop=True) # Drop empty data rows

# %%
# 2.2: Data Visualization

fig1 = plt.figure(figsize=(10,6))
plt.plot(data['Step'], data['X'], label="X")
plt.plot(data['Step'], data['Y'], label="Y")
plt.plot(data['Step'], data['Z'], label="Z")
plt.title("Figure 1: Relationship between Coordinates and Steps")
plt.xlabel('Step')
plt.ylabel('Coordinate')
plt.legend()
plt.grid(True)
# plt.close(fig1)

# Create 3D scatter plot of X, Y, Z coordinates for each step
fig2 = plt.figure(figsize=(10,6))
ax = fig2.add_subplot(projection='3d')
scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=data['Step'], cmap='turbo')
ax.set_title("Figure 2: 3D Scatter of Coordinates by Step")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig2.colorbar(scatter, ax=ax, label='Step')
# plt.close(fig2)

# %% 
# 2.3: Correlation Analysis

# Create a heatmap of the absolute Pearson Correlation Matrix of the data
corr_matrix = data.corr(method='pearson')
plt.figure(figsize=(10,6))
sns.heatmap(np.abs(corr_matrix), annot=True)
plt.title("Figure 3: Pearson Correlation Matrix")

# %%
# 2.4: Classification Model Development/Engineering

# Split data into test and training datasets
X = data.drop(columns=['Step'])
y = data['Step']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Train variouse ML Models

# Logistic Regression
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])

param_grid_lr = {
    'clf__C': [0.01, 0.1, 1, 10],
}

grid_lr = GridSearchCV(pipe_lr,
                       param_grid_lr,
                       cv=5,
                       n_jobs=-1)
grid_lr.fit(X_train, y_train)


# Random Forest
pipe_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

param_grid_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Decision Tree
pipe_dt = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', DecisionTreeClassifier(random_state=42))
])
param_grid_dt = {
    'clf__criterion': ['gini', 'entropy'],
    'clf__max_depth': [5, 10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}
grid_dt = GridSearchCV(pipe_dt,
                       param_grid_dt,
                       cv=5,
                       n_jobs=-1)
grid_dt.fit(X_train, y_train)


# SVM with Randomized Search
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
])

param_dist_svm = {
    'clf__C': np.logspace(-2, 2, 20),
    'clf__gamma': np.logspace(-3, 1, 20),
    'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

rand_svm = RandomizedSearchCV(pipe_svm,
                              param_distributions=param_dist_svm,
                              n_iter=15,
                              cv=5,
                              n_jobs=-1,
                              random_state=42)
rand_svm.fit(X_train, y_train)

# %%
# 2.5: Model Performance Analysis

models = {
    'Logistic Regression': grid_lr.best_estimator_,
    'Random Forest': grid_rf.best_estimator_,
    'Decision Tree': grid_dt.best_estimator_,
    'SVM (Randomized Search)': rand_svm.best_estimator_
}

performance = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    performance[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'F1 Score': f1_score(y_test, y_pred, average='macro')
    }
    
performance_df = pd.DataFrame(performance).T
best_model_name = performance_df['F1 Score'].idxmax()
best_model = models[best_model_name]

print(performance_df)

print(f"Best Model: {best_model_name}")

# Get predictions for best model
y_pred_best = best_model.predict(X_test)

# Compute and plot confusion matrix for best model
cm = confusion_matrix(y_test, y_pred_best, labels=np.unique(y_train))
plt.figure(figsize=(10,6))
sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title("Figure 4: Confusion Matrix - Logistic Regression")


# %%
# 2.6: Stacked Model Performance Analysis

estimators = [
    ('lr', grid_lr.best_estimator_),     # Logistic Regression
    ('svm', rand_svm.best_estimator_)    # SVM from RandomizedSearchCV
]

# Create stacked model
stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

# Train and predict using stacked model
stacked_model.fit(X_train, y_train)
y_pred_stacked = stacked_model.predict(X_test)

# Get performance metrics
stacked_performance = {
    'Accuracy': accuracy_score(y_test, y_pred_stacked),
    'Precision': precision_score(y_test, y_pred_stacked, average='macro'),
    'F1 Score': f1_score(y_test, y_pred_stacked, average='macro')
}

for metric, score in stacked_performance.items():
    print(f"{metric}: {score}")

# Create confusion matrix
cm1 = confusion_matrix(y_test, y_pred_best, labels=np.unique(y_train))
plt.figure(figsize=(10,6))
sns.heatmap(cm1, annot=True, cmap='Blues', xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title("Figure 5: Confusion Matrix - Logistic Regression + Random SVM")

# %%
# 2.7: Model Evaluation

joblib.dump(stacked_model, "maintenance_step_model.joblib") # Save model
loaded_model = joblib.load("maintenance_step_model.joblib") # Load in model

coords = pd.DataFrame([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3.0, 1.8],
    [9.4, 3.0, 1.3]],
    columns=['X', 'Y', 'Z'])

predicted_steps = loaded_model.predict(coords) # Predict steps using loaded model
    
results = coords.assign(**{'Predicted Step': predicted_steps})

print(results)
