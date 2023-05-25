import pandas as pd

# Read the CSV file
data_file = 'data.csv'
df = pd.read_csv(data_file)

# Get the number of features
num_features = df.shape[1] - 1  # Exclude the target variable column if present

# Print the number of features
print(f"Number of features: {num_features}")
