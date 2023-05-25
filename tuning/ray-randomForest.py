import time
import ray
from ray.tune import run_experiments
from ray import tune
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

# Load the dataset
df = pd.read_csv('../data.csv')

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

# Define the training function
def rf_training(config):

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
    rf_model = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        ccp_alpha=config["ccp_alpha"],
    )

    # Fit the model
    rf_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_rf = rf_model.predict(X_test)

    # Calculate the R-squared score for Random Forest Regression
    r2_rf = r2_score(y_test, y_pred_rf)

    # Perform cross-validation using KFold for Random Forest Regression
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    rf_cv_scores = cross_val_score(rf_model, X_poly, y, cv=kf, scoring='r2')
    mean_r2_rf_cv = np.mean(rf_cv_scores)

    return {"R-squared (Random Forest Regression)": r2_rf, "Mean R-squared (Random Forest Regression, Cross-Validation)": mean_r2_rf_cv}

# Define the search space using a configuration dictionary
search_config = {
    "max_depth": tune.grid_search([10, 20, 30, 40, 50]),    
    "n_estimators": tune.grid_search([50, 100, 200]),
    "ccp_alpha": tune.grid_search([0, 0.001, 0.01, 0.1]),
}

# Set up the Ray cluster
ray.init()

# Start the hyperparameter search
start_timestamp = time.time()

run_experiments({
    "random_forest_results": {
        "run": rf_training,
        "config": search_config,
        "num_samples": 1,
        "resources_per_trial": {"cpu": 1},
    }
})

end_timestamp = time.time()

print("Time taken to complete the tuning using 2 VM of 'small' flavor: {:.2f} seconds".format(end_timestamp - start_timestamp))

