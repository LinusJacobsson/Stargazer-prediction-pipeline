import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data_utils
from tqdm import tqdm
import ray
from ray import tune
import joblib
import tempfile
import os
import time
from ray.tune.logger import CSVLogger
from ray.tune import run_experiments


def train_model(config):
    df = pd.read_csv('data.csv')

    # Add your pre-processing steps here. Note that the 'Star Count' and other 
    # columns need to be part of your CSV file.

    df2 = df*1
    columns_to_drop = ['Star Count', 'Repository Name', 'Owner', 'Created at', 'Updated at', 'License Info', 'Primary Language', 'Topics', 'Is Fork', 'Is Archived']
    X_train, X_test, y_train, y_test = train_test_split(df2.drop(labels=columns_to_drop, axis=1), df2['Star Count'], test_size=0.3, random_state=0)
    df_train = pd.concat([X_train, y_train], axis = 1).astype(np.float32)
    df_test = pd.concat([X_test, y_test], axis = 1).astype(np.float32)

    sc = MinMaxScaler()
    df_train[df_train.columns] = sc.fit_transform(df_train)
    df_train, df_val = train_test_split(df_train, test_size=0.2)
    df_test[df_test.columns] = sc.transform(df_test)

    X_train = df_train.drop(labels=['Star Count'], axis=1)
    y_train = df_train['Star Count']

    X_val = df_val.drop(labels=['Star Count'], axis=1)
    y_val = df_val['Star Count']

    X_test = df_test.drop(labels=['Star Count'], axis=1)
    y_test = df_test['Star Count']

    # Making tensor
    target = torch.tensor(y_train.values)
    features = torch.tensor(X_train.values)
    train = data_utils.TensorDataset(features, target)
    train_loader = data_utils.DataLoader(train, batch_size=32, shuffle=True)

    target = torch.tensor(y_val.values)
    features = torch.tensor(X_val.values)
    val = data_utils.TensorDataset(features, target)
    val_loader = data_utils.DataLoader(val, batch_size=32, shuffle=True)

    # Model
    learning_rate=0.001
    n_feature = X_train.shape[1]
    n_output = 1

    model = torch.nn.Sequential(
        torch.nn.Linear(n_feature, int(config["n_hidden1"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(config["n_hidden1"]), int(config["n_hidden2"])),
        torch.nn.ReLU(),
        torch.nn.Linear(int(config["n_hidden2"]), n_output),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    model.train()

    for epoch in tqdm(range(int(config["epochs"]))):
        running_loss = 0
        for data, labels in train_loader:
            optimizer.zero_grad()
            target  = model(data)
            loss = criterion(target, labels.view(-1,1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        val_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in val_loader:
            target = model(data)
            loss = criterion(target,labels.view(-1,1))
            val_loss += loss.item()
        val_loss /= len(val_loader)
        
        tune.report(loss=val_loss) # report validation loss for hpt
        
    # Save the model
    joblib.dump (model,'CNN.joblib')

# Define the search space
config = {
    "n_hidden1": tune.choice([16, 32, 64]),
    "n_hidden2": tune.choice([8, 16, 32]),
    #"lr": tune.loguniform(1e-4, 1e-1),
    "epochs": tune.choice([1000, 2000, 3000])
}
# Create a temporary directory for results
temp_dir = tempfile.mkdtemp()

# Set up the Ray cluster
ray.init()

# Start the hyperparameter search
start_timestamp = time.time()

analysis = tune.run(
        train_model,
        config = config,
        num_samples = 1,
        resources_per_trial = {"cpu": 1},
        local_dir = temp_dir,  # Specify the directory to save the results
        trial_name_creator = lambda trial: f"trial_{trial.trial_id}",  # Specify trial names
)

#analysis_obj = ExperimentAnalysis("ray_results/random_forest_results")
#best_rf_trial = analysis_obj.get_best_trial(metric="mean_accuracy", mode="max")

# Get the best hyperparameters
best_trial = min(analysis, key=lambda trial: trial.last_result["Mean R-squared (Random Forest Regression, Cross-Validation)"])

end_timestamp = time.time()

#analysis = tune.run(
#    train_model, 
#    config=config,
#    metric="loss",
 #   mode="min"
#)

print("Best hyperparameters found were: ", analysis.best_config)

