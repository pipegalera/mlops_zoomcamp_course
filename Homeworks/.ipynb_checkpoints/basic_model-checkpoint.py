import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = '../week_1/data/'

# 1. READ THE DATA
def read_taxis_data(filepath,
    taxi_type = 'green',
    year = '2022',
    train_month = '01',
    valid_month = '02'):

  df = pd.DataFrame()
  data_months = [train_month, valid_month]

  for i in data_months:
    df_part = pd.read_parquet(filepath + f'{year}/{taxi_type}_tripdata_{year}-{i}.parquet')
    df = pd.concat([df, df_part], ignore_index=True)

  if taxi_type == "green":
    df['duration'] = (df.lpep_dropoff_datetime - df.lpep_pickup_datetime)
    df['duration'] = df['duration'].dt.total_seconds().div(60)
    df['valid'] = 0
    df['valid'] = np.where(df.tpep_pickup_datetime >= (year + '-' + valid_month), 1 , df.valid)
  if taxi_type == "yellow":
      df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime)
      df['duration'] = df['duration'].dt.total_seconds().div(60)
      df['valid'] = 0
      df['valid'] = np.where(df.tpep_pickup_datetime >= (year + '-' + valid_month), 1 , df.valid)
  return df

data = read_taxis_data(DATA_PATH,
    taxi_type = 'yellow',
    year='2022',
    train_month =  '01' ,
    valid_month = '03')

# 2. PREP THE DATA
def transform_data(data):
  df = data.copy()

  categorical = ['PU_DO']
  numerical = ['trip_distance']

  # Filter data from tail outcome distribution (not interested in predicting 5 hour rides, 98% rides are under 1 hour)
  df = df[(df.duration >= 1) & (df.duration <= 60)]

  # Feature extraction
  df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)

  # Variable section
  categorical = ['PU_DO']
  numerical = ['trip_distance']

  # DictVectorizer
  dv = DictVectorizer()

  dict_train = df[df.valid == 0][categorical + numerical].to_dict(orient='records')
  dict_valid = df[df.valid == 1][categorical + numerical].to_dict(orient='records')

  X_train = dv.fit_transform(dict_train)
  X_valid = dv.transform(dict_valid)

  y_train = df[df.valid == 0].duration.values
  y_valid = df[df.valid == 1].duration.values

  return X_train, y_train, X_valid, y_valid, dv

X_train, y_train, X_valid, y_valid, dv = transform_data(data)

# 3. BASIC LINEAR MODEL
def run_linear_regressor(print_mse=True):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred, squared=False)
    if print_mse:
        print("MSE of a basic linear regression model:", mse)

    return lr

lr = run_linear_regressor()

# 4. SAVE MODEL
def save_model():
    with open('models/bechmark_lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv,lr), f_out)

save_model()
