from celery import Celery
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score

model_file = './randomForest.joblib'
data_file = './data.csv'

def preprocess_data(df):
    # Keep only the required columns
    df = df[['Fork Count', 'PR Count', 'Issue Count', 'Watcher Count']]

    # Load and apply saved Imputer
    imputer = joblib.load('imputer.joblib')
    df = imputer.transform(df)

    # Load and apply saved StandardScaler
    scaler = joblib.load('scaler.joblib')
    df = scaler.transform(df)

    return df

def load_model():
    loaded_model = joblib.load(model_file)
    return loaded_model

# Celery configuration
CELERY_BROKER_URL = 'amqp://rabbitmq:rabbitmq@rabbit:5672/'
CELERY_RESULT_BACKEND = 'rpc://'

# Initialize Celery
celery = Celery('workerA', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

@celery.task()
def add_nums(a, b):
   return a + b

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

    # Create a dictionary with 'y' and 'predicted' keys
    final_results = {
        'y': df_sample['Star Count'].tolist(),
        'predicted': predictions.tolist()
    }

    return final_results

@celery.task
def get_accuracy():
    # Load the data
    df = pd.read_csv(data_file)

    # Sample 5 random rows
    df_sample = df.sample(5)

    # Preprocess the data
    X_sample = preprocess_data(df_sample)

    # Get the true labels
    y_true = df_sample['Star Count']

    # Load the trained model
    loaded_model = load_model()

    # Make predictions
    y_pred = loaded_model.predict(X_sample)

    # Compute R-squared score
    r2 = r2_score(y_true, y_pred)

    return r2

print(get_predictions())
