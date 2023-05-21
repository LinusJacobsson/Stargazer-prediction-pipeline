import json
import sys
import pandas as pd
import numpy as np
import sklearn.preprocessing as skl_pre
import sklearn.ensemble as skl_ensemble
import sklearn.model_selection as skl_ms
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

filename = sys.argv[1]

# Load the data from the file
with open(filename, 'r') as f:
    data = json.load(f)

# Extract the repository data
repos = data['data']['search']['edges']

# Initialize a list to hold dictionaries
repo_dicts = []

# Iterate over repos and extract relevant info
for i, repo in enumerate(repos, start=1):
    repo_data = repo['node']

    # Create a dictionary for each repository
    repo_dict = {}

    # Basic info
    repo_dict['Repository Name'] = repo_data['name']
    repo_dict['Owner'] = repo_data['owner']['login']
    repo_dict['Star Count'] = repo_data['stargazers']['totalCount']

    # Additional features
    repo_dict['Fork Count'] = repo_data['forkCount']
    repo_dict['Created at'] = pd.to_datetime(repo_data['createdAt']).timestamp()
    repo_dict['Updated at'] = pd.to_datetime(repo_data['updatedAt']).timestamp()
    repo_dict['Primary Language'] = repo_data['primaryLanguage']['name'] if repo_data['primaryLanguage'] else 'None'
    repo_dict['PR Count'] = repo_data['pullRequests']['totalCount']
    repo_dict['Issue Count'] = repo_data['issues']['totalCount']
    repo_dict['Watcher Count'] = repo_data['watchers']['totalCount']
    repo_dict['Disk Usage'] = repo_data['diskUsage']
    repo_dict['Is Fork'] = repo_data['isFork']
    repo_dict['Is Archived'] = repo_data['isArchived']
    repo_dict['License Info'] = repo_data['licenseInfo']['name'] if repo_data['licenseInfo'] else 'None'

    # Extract the topics
    topics = [edge['node']['topic']['name'] for edge in repo_data['repositoryTopics']['edges']]
    for topic in topics:
        repo_dict['Topic_' + topic] = 1

    # Add the dictionary to the list
    repo_dicts.append(repo_dict)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(repo_dicts)

# Convert categorical columns to strings
categorical_cols = ['Primary Language', 'License Info']
df[categorical_cols] = df[categorical_cols].astype(str)

# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Split the data into input features (X) and target variable (y)
X = df_encoded.drop(["Star Count", "Repository Name", "Owner"], axis=1).values
y = df_encoded["Star Count"].values

# Impute missing values with the mean value of each column
imputer = SimpleImputer()
X = imputer.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = skl_ms.train_test_split(X, y, test_size=0.2)

# Fit the Gradient Boosting Classifier model
gbm_model = GradientBoostingClassifier(n_estimators=100)  # Adjust the number of estimators as needed
gbm_model.fit(X_train, y_train)

# Predict on the test set using Gradient Boosting Classifier
y_pred_gbm = gbm_model.predict(X_test)

# Calculate the misclassification rate for Gradient Boosting Classifier
misclassification_gbm = np.mean(y_pred_gbm != y_test)
print('Misclassification (Gradient Boosting): ', misclassification_gbm)

# Convert the target variables to integers
y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Fit the XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100)
xgb_model.fit(X_train, y_train)

# Predict on the test set using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# Calculate the misclassification rate for XGBoost
misclassification_xgb = np.mean(y_pred_xgb != y_test)
print('Misclassification (XGBoost): ', misclassification_xgb)




