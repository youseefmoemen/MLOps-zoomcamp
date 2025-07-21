#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import mlflow
import uuid



# In[2]:


def read_dataframe(filename):
    if filename.endswith('.csv'):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df['ride_id'] = generate_ids(len(df))
    
    return df

def prepare_dict(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    df[categorical] = df[categorical].astype(str)
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts

def generate_ids(N):
    ride_ids = []
    for _ in range(N):
        ride_ids.append(str(uuid.uuid4()))
    return ride_ids


# In[3]:


def load_model(model_id):
    return mlflow.pyfunc.load_model(model_id)


def apply_model(input_file, output_file, model_id):
    df = read_dataframe(input_file)
    dicts = prepare_dict(df)
    model = load_model(model_id)
    y_pred = model.predict(dicts)
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['lpep_pickup_datetime'] = df['lpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['predicted_duration'] = y_pred
    df_result['actual_duration'] = df['duration']
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration'] 
    df_result['model_version'] = model_id
    df_result.to_parquet(output_file, index=False)
    print('Saved')



# In[4]:

def run():
    MLFLOW_TACKING_URI = "sqlite:////home/youseef/mlflow.db"
    mlflow.set_tracking_uri(MLFLOW_TACKING_URI)
    mlflow.set_experiment('nyc-experiment')
    MODEL_ID = "models:/taxi-best/19"
    input_path = '/home/youseef/MLOps-zoomcamp/data/green_tripdata_2021-01.parquet'
    output_path = 'output/green_tripdata_2021-01_predictions.parquet'

    apply_model(input_path, output_path, MODEL_ID)


# In[5]
if __name__== '__main__':
    print('Running...')
    run()
    print('Finished')