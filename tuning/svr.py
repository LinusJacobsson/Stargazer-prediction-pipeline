import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.ensemble as skl_ensemble
import sklearn.model_selection as skl_ms
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingRegressor
import xgboost as xgb
from sklearn.svm import SVR

df= pd.read_csv('../data.csv')
# Convert categorical columns to strings
categorical_cols = ['Primary Language', 'License Info']
df[categorical_cols] = df[categorical_cols].astype(str)

# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split the data into input features (X) and target variable (y)
X = df_encoded.drop(['Star Count','Owner', 'Repository Name', 'Owner', 'Created at', 'Updated at',  'Topics'], axis=1).values
y = df_encoded["Star Count"].values

# Impute missing values with the mean value of each column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Normalize numerical features
scaler = skl_pre.StandardScaler()
X = scaler.fit_transform(X)

np.random.seed(42)  # Set a random seed for reproducibility

# Split the data into train and test sets
X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y, test_size=0.2)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Perform GridSearchCV for SVR
svr_model = SVR()
svr_grid = GridSearchCV(estimator=svr_model, param_grid=param_grid, scoring='r2', cv=5)
svr_grid.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_svr_model = svr_grid.best_estimator_

# Predict on the test set using the best SVR model
y_pred_svr = best_svr_model.predict(X_test)

# Calculate the R-squared score for SVR
r2_svr = r2_score(y_test, y_pred_svr)
print('R-squared (SVR): ', r2_svr)


# Ensemble Methods - Bagging
bagging_model = BaggingRegressor(base_estimator=best_svr_model, n_estimators=10)
bagging_model.fit(X_train, y_train)

# Predict on the test set using the Bagging model
y_pred_bagging = bagging_model.predict(X_test)

# Calculate the R-squared score for Bagging model
r2_bagging = r2_score(y_test, y_pred_bagging)
print('R-squared (Bagging): ', r2_bagging)


