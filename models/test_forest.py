import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("/home/appuser/de2-final-project/data.csv")

# Convert categorical columns to strings
categorical_cols = ['Primary Language', 'License Info']
df[categorical_cols] = df[categorical_cols].astype(str)

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Save the encoded column names
joblib.dump(df_encoded.columns, 'columns.joblib')

# Split the data into input features (X) and target variable (y)
X = df_encoded.drop(['Star Count', 'Owner', 'Repository Name', 'Owner', 'Created at', 'Updated at', 'Topics'], axis=1).values
y = df_encoded["Star Count"].values

# Impute missing values with the mean value of each column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Save the imputer
joblib.dump(imputer, 'imputer.joblib')

# Normalize numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Generate polynomial features
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

# Split the data into train and test sets
np.random.seed(0)  # Set a random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2)

# Create a Random Forest Regression model
rf_model = RandomForestRegressor()
rf_params = {'n_estimators': [50, 100, 200]}  # Define the hyperparameters to tune
grid_search = GridSearchCV(rf_model, rf_params, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)

print('Best score: ', grid_search.best_score_)

# Saving the model for possible deployment
joblib.dump(grid_search.best_estimator_, 'randomForest.joblib')
