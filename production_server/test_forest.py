import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# Load the dataset
df = pd.read_csv("/home/appuser/de2-final-project/data.csv")

# Select only the required columns
df = df[['Star Count', 'Fork Count', 'PR Count', 'Issue Count', 'Watcher Count']]

# Split the data into input features (X) and target variable (y)
X = df.drop(['Star Count'], axis=1)
y = df["Star Count"]

# Impute missing values with the mean value of each column
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Save the imputer and scaler
joblib.dump(imputer, 'imputer.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Create a Random Forest Regression model
rf_model = RandomForestRegressor()

# Grid search for hyperparameter tuning
params = {'n_estimators': [50, 100, 200]}  # Define the hyperparameters to tune
grid_search = GridSearchCV(rf_model, params, cv=5)
grid_search.fit(X_scaled, y)

# Save the trained model
joblib.dump(grid_search.best_estimator_, 'randomForest.joblib')
