from celery import Celery
import numpy as np
import pandas as pd
import joblib
import sklearn.preprocessing as skl_pre
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

model_file = '/home/ubuntu/model_serving/single_server_with_docker/production_server/de2-final-project/models/randomForest.joblib'
data_file = '/home/ubuntu/model_serving/single_server_with_docker/production_server/de2-final-project/data.csv'

def preprocess_data(df):
    # Convert categorical columns to strings
    categorical_cols = ['Primary Language', 'License Info']
    df[categorical_cols] = df[categorical_cols].astype(str)

    # Convert categorical columns to numerical using one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Create input features (X)
    X = df_encoded.drop(['Star Count', 'Repository Name', 'Owner', 'Created at', 'Updated at', 'Topics'], axis=1).values

    # Impute missing values with the mean value of each column
    imputer = SimpleImputer()
    X = imputer.fit_transform(X)

    # Normalize numerical features
    scaler = skl_pre.StandardScaler()
    X = scaler.fit_transform(X)

    # Generate polynomial features
    poly = PolynomialFeatures(degree=1)
    X_poly = poly.fit_transform(X)

    return X_poly

def load_model():
    loaded_model = joblib.load(model_file)
    return loaded_model

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'

# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery.task
def get_predictions():
    # Load the data
    df = pd.read_csv(data_file)

    # Sample 5 random rows
    df_sample = df.sample(5)

    # Preprocess the data
    X_sample = preprocess_data(df_sample)

    # Load the trained model
    loaded_model = load_model()

    # Make predictions
    predictions = loaded_model.predict(X_sample).flatten()

    # Get the indices of the predictions sorted in descending order
    sorted_indices = np.argsort(predictions)[::-1]

    # Return the sorted repository names and their corresponding predicted star counts
    results = [(df_sample.iloc[i]['Repository Name'], predictions[i]) for i in sorted_indices]

    return results
