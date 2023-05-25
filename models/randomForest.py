import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.model_selection as skl_ms
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import joblib
####
df = pd.read_csv("/home/appuser/de2-final-project/data.csv")
# Convert categorical columns to strings
categorical_cols = ['Primary Language', 'License Info']
df[categorical_cols] = df[categorical_cols].astype(str)

# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split the data into input features (X) and target variable (y)
X = df_encoded.drop(['Star Count', 'Owner', 'Repository Name', 'Owner', 'Created at', 'Updated at', 'Topics'], axis=1).values
y = df_encoded["Star Count"].values

# Impute missing values with the mean value of each column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Normalize numerical features
scaler = skl_pre.StandardScaler()
X = scaler.fit_transform(X)

# Generate polynomial features
poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

np.random.seed(0)  # Set a random seed for reproducibility

# Split the data into train and test sets
X_train, X_test, y_train, y_test = skl_ms.train_test_split(X_poly, y, test_size=0.2)

# Create a Random Forest Regression model
rf_model = RandomForestRegressor()
rf_params = {'n_estimators': [50, 100, 200]}  # Define the hyperparameters to tune
rf_grid = GridSearchCV(rf_model, rf_params, scoring='r2', cv=5)
rf_grid.fit(X_train, y_train)
rf_best_model = rf_grid.best_estimator_

# Predict on the test set using the best Random Forest Regression model
y_pred_rf = rf_best_model.predict(X_test)

# Calculate the R-squared score for Random Forest Regression
#r2_rf = r2_score(y_test, y_pred_rf)
#print(r2_rf)


# Perform cross-validation using KFold for Random Forest Regression
kf = KFold(n_splits=5, shuffle=True, random_state=0)
rf_cv_scores = cross_val_score(rf_best_model, X_poly, y, cv=kf, scoring='r2')
mean_r2_rf_cv = np.mean(rf_cv_scores)
print(mean_r2_rf_cv)

joblib.dump(rf_best_model, 'randomForest.joblib')
