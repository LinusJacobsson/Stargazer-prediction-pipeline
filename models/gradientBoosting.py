import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
######
df = pd.read_csv("/home/appuser/de2-final-project/data.csv")
cols = ['Fork Count', 'PR Count', 'Issue Count', 'Watcher Count']
X = df[cols]
y = df["Star Count"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
reg = GradientBoostingRegressor(learning_rate = 0.2, n_estimators=100, max_depth=None)
#reg = RandomForestRegressor(max_features="log2", ccp_alpha=0.02, n_estimators=200, max_depth=100)
#reg = Ridge(alpha=0.1)
#cv_scores = cross_val_score(reg, X_train, y_train, cv=5, scoring='r2')

#print("Cross-Validation R-squared Scores:", cv_scores)
#print("Average R-squared:", cv_scores.mean())

reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
res = r2_score(y_test, predictions)

print(res)
