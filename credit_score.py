import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Generate realistic synthetic data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(60000, 15000, n_samples).clip(20000, 150000),
    'age': np.random.randint(21, 65, n_samples),
    'credit_history_years': np.random.randint(1, 30, n_samples),
    'loan_amount': np.random.normal(20000, 10000, n_samples).clip(5000, 100000),
    'open_credit_lines': np.random.randint(1, 10, n_samples),
    'debt_to_income_ratio': np.random.normal(0.3, 0.1, n_samples).clip(0.1, 0.9),
    'missed_payments': np.random.poisson(1, n_samples),
})

# Simulate credit score based on weighted sum + noise
data['credit_score'] = (
    300
    + (data['income'] / 1000)
    - (data['loan_amount'] / 2000)
    + (data['credit_history_years'] * 2)
    - (data['debt_to_income_ratio'] * 100)
    - (data['missed_payments'] * 10)
    + (data['open_credit_lines'] * 5)
    + np.random.normal(0, 25, n_samples)
).clip(300, 850)

# Define features and target
X = data.drop('credit_score', axis=1)
y = data['credit_score']

# Split the data BEFORE model fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), X.columns.tolist())
])

# Pipeline with RandomForest
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Predictions
y_pred = grid_search.predict(X_test)

# Evaluation
print("Best parameters found:", grid_search.best_params_)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Visualization: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='skyblue', alpha=0.6)
plt.plot([300, 850], [300, 850], color='red', linestyle='--')  # Ideal line
plt.xlabel('Actual Credit Score')
plt.ylabel('Predicted Credit Score')
plt.title('Actual vs Predicted Credit Score')
plt.grid(True)
plt.show()

