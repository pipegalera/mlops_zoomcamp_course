import pandas as pd
import numpy as np
import pickle

# Read data
def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

# Read model
with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Make prediction
categorical = ['PULocationID', 'DOLocationID']

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print("Q6: ", np.mean(y_pred))
