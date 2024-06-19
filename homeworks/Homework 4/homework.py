import pickle
import os
import sys
import pandas as pd
import numpy as np

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

std_preds = np.std(y_pred).round(2)

print("Question 1:", std_preds)

year = 2023
month = 3

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
#df_result['PULocationID'] = df['PULocationID']
#df_result['DOLocationID'] = df['DOLocationID']
#df_result['actual_duration'] = df['duration']
df_result['predicted_duration'] = y_pred
#df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

output_file = f'output/{year:04d}_{month:02d}.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print("Question 2:", os.stat(output_file).st_size)

print("Question 3: `jupyter nbconvert notebook.ipynb --to script`")

def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    output_file = f'output/{year:04d}_{month:02d}.parquet'

    print("Question 4: ", np.mean(df_result['predicted_duration']))


if __name__ == "__main__":
    run()
