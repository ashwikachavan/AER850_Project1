# Step 1: Read data from CSV file
import pandas as pd

file_path = "Project 1 Data.csv"  
df = pd.read_csv(file_path)

# Few rows from the file
print(df)

# Step 2: Data Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

# Histogram (X)
plt.subplot(3, 1, 1)
counts, bins = np.histogram(df['X'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, color='skyblue', edgecolor='black')
plt.title('Histogram for X')
plt.xlabel('X')
plt.ylabel('Frequency')

# Histogram (Y)
plt.subplot(3, 1, 2)
counts, bins = np.histogram(df['Y'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, color='orange', edgecolor='black')  
plt.title('Histogram for Y')
plt.xlabel('Y')
plt.ylabel('Frequency')

# Histogram (Z)
plt.subplot(3, 1, 3)
counts, bins = np.histogram(df['Z'], bins=30)
plt.hist(bins[:-1], bins, weights=counts, color='teal', edgecolor='black')  
plt.title('Histogram for Z')
plt.xlabel('Z')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 3 – Correlation Analysis 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Pearson correlation among features
corr_features = df[['X', 'Y', 'Z']].corr(method='pearson')
sns.heatmap(corr_features, annot=True, cmap='YlGnBu', vmin=-1, vmax=1)
plt.title('Feature Correlation (Pearson)')
plt.show()

# Pearson correlation between features and target classes
step_ohe = pd.get_dummies(df['Step'], prefix='Step')
corr_all = pd.concat([df[['X', 'Y', 'Z']], step_ohe], axis=1).corr(method='pearson')
step_cols = [c for c in corr_all.columns if c.startswith('Step_')]

plt.figure(figsize=(16, 4))
sns.heatmap(corr_all.loc[['X', 'Y', 'Z'], step_cols],
            annot=True, fmt='.2f', cmap='YlGnBu', vmin=-1, vmax=1)
plt.title('Feature ↔ Step Correlations (Pearson)')
plt.ylabel('Features')
plt.xlabel('Step classes')
plt.tight_layout()
plt.show()

# Step 4: Classification 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data split
X = df[['X','Y','Z']]
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scoring = "f1_macro"   
cv = 5                 

# Logistic Regression 
logreg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=3000, multi_class="multinomial", solver="saga", random_state=42))
])
logreg_grid = {"clf__C": [0.1, 1, 10]}
logreg_cv = GridSearchCV(logreg, logreg_grid, cv=cv, scoring=scoring).fit(X_train, y_train)

# SVM 
svm = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC())
])
svm_grid = [
    {"clf__kernel": ["linear"], "clf__C": [0.5, 1, 5]},
    {"clf__kernel": ["rbf"],    "clf__C": [1, 5], "clf__gamma": ["scale", 0.1]}
]
svm_cv = GridSearchCV(svm, svm_grid, cv=cv, scoring=scoring).fit(X_train, y_train)

# Decision Tree 
dt = DecisionTreeClassifier(random_state=42)
dt_grid = {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 4]}
dt_cv = GridSearchCV(dt, dt_grid, cv=cv, scoring=scoring).fit(X_train, y_train)

# Random Search CV
rf = RandomForestClassifier(random_state=42)
rf_dist = {
    "n_estimators": np.arange(150, 401, 50),
    "max_depth": [None, 6, 10, 14],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 4]
}
rf_cv = RandomizedSearchCV(rf, rf_dist, n_iter=20, cv=cv, scoring=scoring, random_state=42).fit(X_train, y_train)

# Test-set comparison 
models = {
    "LogReg": logreg_cv.best_estimator_,
    "SVM": svm_cv.best_estimator_,
    "DecisionTree": dt_cv.best_estimator_,
    "RandomForest": rf_cv.best_estimator_,
}

print("Best params:")
print("  LogReg      ->", logreg_cv.best_params_)
print("  SVM         ->", svm_cv.best_params_)
print("  DecisionTree->", dt_cv.best_params_)
print("  RandomForest->", rf_cv.best_params_)

print("\nTest performance (Acc | F1_macro | Prec_macro):")
for name, m in models.items():
    yp = m.predict(X_test)
    acc = accuracy_score(y_test, yp)
    f1m = f1_score(y_test, yp, average="macro")
    prec = precision_score(y_test, yp, average="macro", zero_division=0)
    print(f"{name:12s}: {acc:.3f} | {f1m:.3f} | {prec:.3f}")
    
# Step 5: Model Performance Analysis 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix

# Trained models from Step 4
models = {
    "Logistic Regression": logreg_cv.best_estimator_,
    "SVM": svm_cv.best_estimator_,
    "Decision Tree": dt_cv.best_estimator_,
    "Random Forest": rf_cv.best_estimator_,
}

# Evaluate each model
results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    results.append([name, acc, f1m, prec])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1 Score", "Precision"])
print("\n Model Comparison (Test Set)")
print(results_df.to_string(index=False))

# Confusion matrix (based on F1)
best_model_name = results_df.loc[results_df["F1 Score"].idxmax(), "Model"]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

# Plot 
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Step")
plt.ylabel("True Step")
plt.tight_layout()
plt.show()

#Step 6: Stacked Model Performance Analysis 
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

stack = StackingClassifier(
    estimators=[("svm", svm_cv.best_estimator_), ("rf", rf_cv.best_estimator_)],
    final_estimator=LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000, random_state=42),
    stack_method="auto",
    passthrough=False,   
    cv=5,
    n_jobs=-1
)

stack.fit(X_train, y_train)

y_pred = stack.predict(X_test)
print("Stacked Model")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.6f}")
print(f"F1 (macro): {f1_score(y_test, y_pred, average='macro'):.6f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, average='macro', zero_division=0):.6f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix — Stacked Model (SVM + RF)")
plt.xlabel("Predicted Step"); plt.ylabel("True Step")
plt.tight_layout(); plt.show()

# Step 7: Model Evaluation 
import joblib
import numpy as np

# Select the best-performing model 
best_model = logreg_cv.best_estimator_  

# Save the trained model for future use
joblib.dump(best_model, "best_model.joblib")
print("Model successfully saved as 'best_model.joblib'")

# Reload the saved model to verify reusability
loaded_model = joblib.load("best_model.joblib")

# Predict maintenance steps 
coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.000, 3.0625, 1.93],
    [9.400, 3.000, 1.80],
    [9.400, 3.000, 1.30]
])

predictions = loaded_model.predict(coords)

print("\nPredicted Maintenance Steps:")
for coord, step in zip(coords, predictions):
    print(f"Coordinates {coord} → Step {step}")

