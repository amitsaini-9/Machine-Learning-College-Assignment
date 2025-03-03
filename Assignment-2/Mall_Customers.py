# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score, confusion_matrix

# Load the data
df = pd.read_csv('Mall_Customers.csv')

# Initial Data Exploration
print("Data Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Data Visualization
plt.figure(figsize=(15, 5))

# 1. Spending Score Distribution
plt.subplot(1, 3, 1)
sns.histplot(data=df, x='Spending Score (1-100)', kde=True)
plt.title('Spending Score Distribution')

# 2. Age vs Spending Score
plt.subplot(1, 3, 2)
sns.scatterplot(data=df, x='Age', y='Spending Score (1-100)', hue='Genre')
plt.title('Age vs Spending Score by Gender')

# 3. Income vs Spending Score
plt.subplot(1, 3, 3)
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Genre')
plt.title('Income vs Spending Score by Gender')

plt.tight_layout()
plt.show()

# Data Preprocessing
# Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Annual Income (k$)'] = df['Annual Income (k$)'].fillna(df['Annual Income (k$)'].mean())
df['Genre'] = df['Genre'].fillna(df['Genre'].mode()[0])
df = df.dropna(subset=['Spending Score (1-100)'])

# Encode Gender for Classification
le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])

# REGRESSION MODELS

# Prepare data for regression
X_reg = df[['Age', 'Annual Income (k$)']]
y_reg = df['Spending Score (1-100)']

# Split data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Create regression models dictionary
regression_models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'SVR (Linear)': SVR(kernel='linear'),
    'SVR (RBF)': SVR(kernel='rbf')
}

# Train and evaluate regression models
print("\nRegression Models Performance:")
regression_results = {}
for name, model in regression_models.items():
    model.fit(X_reg_train_scaled, y_reg_train)
    y_pred = model.predict(X_reg_test_scaled)
    mse = mean_squared_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    regression_results[name] = {'mse': mse, 'r2': r2, 'predictions': y_pred}
    print(f"\n{name}:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")

# CLASSIFICATION MODELS

# Prepare data for classification
X_class = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
y_class = df['Genre_encoded']

# Split data
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

# Scale features
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

# Create classification models dictionary
classification_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVC (Linear)': SVC(kernel='linear', random_state=42),
    'SVC (RBF)': SVC(kernel='rbf', random_state=42)
}

# Train and evaluate classification models
print("\nClassification Models Performance:")
classification_results = {}
for name, model in classification_models.items():
    model.fit(X_class_train_scaled, y_class_train)
    y_pred = model.predict(X_class_test_scaled)
    accuracy = accuracy_score(y_class_test, y_pred)
    classification_results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred,
        'report': classification_report(y_class_test, y_pred)
    }
    print(f"\n{name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_class_test, y_pred))

# VISUALIZATIONS

# 1. Regression Models Comparison
plt.figure(figsize=(12, 5))

# MSE Comparison
plt.subplot(1, 2, 1)
mse_scores = [results['mse'] for results in regression_results.values()]
sns.barplot(x=list(regression_models.keys()), y=mse_scores)
plt.title('Regression Models - MSE Comparison')
plt.xticks(rotation=45)

# R² Comparison
plt.subplot(1, 2, 2)
r2_scores = [results['r2'] for results in regression_results.values()]
sns.barplot(x=list(regression_models.keys()), y=r2_scores)
plt.title('Regression Models - R² Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 2. Classification Models Comparison
plt.figure(figsize=(10, 5))
accuracies = [results['accuracy'] for results in classification_results.values()]
sns.barplot(x=list(classification_models.keys()), y=accuracies)
plt.title('Classification Models - Accuracy Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Best Model Analysis

# Best Regression Model
best_reg_model_name = max(regression_results.items(), 
                         key=lambda x: x[1]['r2'])[0]
best_reg_predictions = regression_results[best_reg_model_name]['predictions']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(y_reg_test, best_reg_predictions)
plt.plot([y_reg_test.min(), y_reg_test.max()], 
         [y_reg_test.min(), y_reg_test.max()], 'r--', lw=2)
plt.xlabel('Actual Spending Score')
plt.ylabel('Predicted Spending Score')
plt.title(f'Best Regression Model: {best_reg_model_name}\nActual vs Predicted')

# Best Classification Model
best_class_model_name = max(classification_results.items(), 
                          key=lambda x: x[1]['accuracy'])[0]
best_class_predictions = classification_results[best_class_model_name]['predictions']

plt.subplot(1, 2, 2)
cm = confusion_matrix(y_class_test, best_class_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Best Classification Model: {best_class_model_name}\nConfusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
plt.show()

# 4. Feature Importance Analysis
plt.figure(figsize=(12, 4))

# Random Forest Regression Feature Importance
plt.subplot(1, 2, 1)
rf_reg = regression_models['Random Forest']
importance_reg = pd.DataFrame({
    'feature': X_reg.columns,
    'importance': rf_reg.feature_importances_
})
sns.barplot(data=importance_reg, x='feature', y='importance')
plt.title('Feature Importance - Regression')
plt.xticks(rotation=45)

# Random Forest Classification Feature Importance
plt.subplot(1, 2, 2)
rf_class = classification_models['Random Forest']
importance_class = pd.DataFrame({
    'feature': X_class.columns,
    'importance': rf_class.feature_importances_
})
sns.barplot(data=importance_class, x='feature', y='importance')
plt.title('Feature Importance - Classification')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Print Final Summary
print("\nFinal Model Summary:")
print("===================")

print("\nBest Regression Model:")
print(f"Model: {best_reg_model_name}")
print(f"R² Score: {regression_results[best_reg_model_name]['r2']:.4f}")
print(f"MSE: {regression_results[best_reg_model_name]['mse']:.4f}")

print("\nBest Classification Model:")
print(f"Model: {best_class_model_name}")
print(f"Accuracy: {classification_results[best_class_model_name]['accuracy']:.4f}")
print("\nClassification Report:")
print(classification_results[best_class_model_name]['report'])