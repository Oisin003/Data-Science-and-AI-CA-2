#L00172671 - Oisin Gibson - CA2
#Structure
# 1. Import libraries
# 2. Load dataset
# 3. Data cleaning
# 4. Exploratory analysis (optional graphs)
# 5. Feature selection
# 6. Machine learning model
# 7. Evaluation

import pandas as pd
import numpy as np
from pathlib import Path

# Load dataset
dataset_xlsx = Path("ai_medical_triage_dataset.xlsx")
dataset_csv = Path("ai_medical_triage_dataset.csv")

if dataset_xlsx.exists():
    df = pd.read_excel(dataset_xlsx)
elif dataset_csv.exists():
    df = pd.read_csv(dataset_csv)
else:
    raise FileNotFoundError("Dataset file not found. Expected ai_medical_triage_dataset.xlsx or ai_medical_triage_dataset.csv")

# View first rows
print(df.head())

#Inspect the dataset
print(df.info())        # Data types & missing values
print(df.describe())    # Summary statistics
print(df.isnull().sum())  # Count missing values

#Clean the Gender column (Inconsistent entries)
#Standardize gender values
df['Gender'] = df['Gender'].astype(str).str.strip().str.lower()

#Do not need to treat M and Male as different + vice versa for Female and F
df['Gender'] = df['Gender'].replace({
    'male': 'M',
    'm': 'M',
    'female': 'F',
    'f': 'F'
})

#Clean the Temperature column (Inconsistent entries)
# Remove "C" and convert to float
df['Temperature_C'] = df['Temperature_C'].astype(str).str.replace(r'(?i)c', '', regex=True).str.strip()

df['Temperature_C'] = pd.to_numeric(df['Temperature_C'], errors='coerce')

#Handle the Missing Values
# Fill missing numerical values with mean
df['Temperature_C'] = df['Temperature_C'].fillna(df['Temperature_C'].mean())
df['Heart_Rate'] = df['Heart_Rate'].fillna(df['Heart_Rate'].mean())

#Remove the Erroneous Data
# Keep only realistic ages
df = df[(df['Age'] > 0) & (df['Age'] < 100)]

#Hanlde the Outliner like Heart_Rate
# Remove extreme heart rate values
df = df[(df['Heart_Rate'] >= 50) & (df['Heart_Rate'] <= 180)]

#Convert to yes or no to binary
binary_cols = ['Cough', 'Fatigue', 'Headache', 'Chronic_Disease']

for col in binary_cols:
    df[col] = df[col].astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
    
#Encode the target variable
df['Risk_Level'] = df['Risk_Level'].astype(str).str.strip().str.lower().map({
    'low': 0,
    'medium': 1,
    'high': 2
})

# Drop rows that still have invalid or unmapped values in key columns
required_cols = ['Gender', 'Temperature_C', 'Heart_Rate', 'Cough', 'Fatigue', 'Headache', 'Chronic_Disease', 'Risk_Level']
df = df.dropna(subset=required_cols)

#Final Check
print(df.info())
print(df.head())

#Save the cleaned dataset
df.to_excel("cleaned_medical_dataset.xlsx", index=False)

# =====================================================
# MACHINE LEARNING: Predicting Patient Risk Level
# =====================================================

# -------------------------------
# 5. Feature Selection
# -------------------------------

# Features (X) and target (y)
X = df.drop(['Patient_ID', 'Risk_Level'], axis=1)
y = df['Risk_Level']

# Convert Gender to numeric
X['Gender'] = X['Gender'].map({'M': 0, 'F': 1})

# Remove any remaining rows with unmapped gender values
X = X.dropna(subset=['Gender'])
y = y.loc[X.index]

print("\nFeatures used:\n", X.columns)


# -------------------------------
# 6. Machine Learning Model
# -------------------------------

try:
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError as exc:
    raise SystemExit("Missing dependency: scikit-learn. Install it with: pip install scikit-learn") from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("Missing dependency: matplotlib. Install it with: pip install matplotlib") from exc

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(max_depth=4, random_state=42)

# Train model
model.fit(X_train, y_train)

print("\nModel training complete.")


# -------------------------------
# 7. Evaluation
# -------------------------------

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Classification Report
print(
    "\nClassification Report:\n",
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        target_names=['Low', 'Medium', 'High'],
        zero_division=0
    )
)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
print("\nConfusion Matrix:\n", cm)


# -------------------------------
# Graph 1: Confusion Matrix
# -------------------------------

plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1, 2], ['Low', 'Medium', 'High'])
plt.yticks([0, 1, 2], ['Low', 'Medium', 'High'])

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()


# -------------------------------
# Graph 2: Feature Importance
# -------------------------------

importances = model.feature_importances_

plt.figure()
plt.barh(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()


# -------------------------------
# Graph 3: Decision Tree Diagram
# -------------------------------

from sklearn.tree import plot_tree

plt.figure(figsize=(15,10))
plot_tree(model, feature_names=X.columns, class_names=['Low','Medium','High'], filled=True)
plt.title("Decision Tree")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.close()

print("\nSaved plots: confusion_matrix.png, feature_importance.png, decision_tree.png")