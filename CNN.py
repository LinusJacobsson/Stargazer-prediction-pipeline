#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils
import torch.nn.functional as F
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from statistics import mean
import joblib



def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device


# # Loading Data

df = pd.read_csv('data.csv')


df.info()

df.head(10)


# # Change boolean to number

df2 = df*1
df2.head()


# # Drop Unnumeric column

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





target = torch.tensor(y_test.values)
features = torch.tensor(X_test.values)
test = data_utils.TensorDataset(features, target)
test_loader = data_utils.DataLoader(test, batch_size=32, shuffle=True)


# # Model


TRAIN_EPOCHS = 2000
Learning_rate = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_feature = X_train.shape[1]
n_hidden1 = 16
n_hidden2 = 8
n_output = 1




model = torch.nn.Sequential(
    torch.nn.Linear(n_feature, n_hidden1),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden1, n_hidden2),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden2, n_output),
)



optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
criterion = torch.nn.MSELoss()
model.train()



log_val = []
log_train = []
for epoch in tqdm(range(TRAIN_EPOCHS)):
    running_loss = 0
    for data, labels in train_loader:
        #if torch.cuda.is_available():
        #    data, labels = data.cuda(), labels.cuda()
        optimizer.zero_grad()
        target  = model(data)
        loss = criterion(target, labels.view(-1,1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    log_train.append(running_loss / len(train_loader))
    
    
    val_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for data, labels in val_loader:
        #if torch.cuda.is_available():
        #    data, labels = data.cuda(), labels.cuda()
        target = model(data)
        loss = criterion(target,labels.view(-1,1))
        val_loss += loss.item()
    log_val.append(val_loss / len(val_loader))


# # Find loss function

test_loss = 0.0
for data, labels in test_loader:
    target = model(data)
    loss = criterion(target,labels.view(-1,1))
    test_loss += loss.item()
print(test_loss / len(test_loader))


# # Find R-Square

 # Convert PyTorch tensors to numpy arrays for computation
target_numpy = target.detach().numpy()
labels_numpy = labels.view(-1,1).detach().numpy()

r_squared = r2_score(labels_numpy, target_numpy)
print(r_squared)


# Saving model


joblib.dump (model,'CNN.joblib')





