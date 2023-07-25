import streamlit as st
import boto3

import datetime
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import XGBModel

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import pickle

#Caching the models for faster loading

s3_bucket = 'cabi-model-artefacts'

# Function to read the pickled model from S3
@st.cache_data
def load_model_from_s3(bucket, file_name):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, file_name)

    if file_name.endswith('gz'):
        with gzip.GzipFile(fileobj=response['Body']) as f:
            model_bytes = f.read()
    else:
        model_bytes = obj.get()['Body'].read()

    return pickle.loads(model_bytes)



# Load the model from S3
#model = load_model_from_s3(s3_bucket, "darts_xgb_clus_alldata_final.pkl")
model = load_model_from_s3(s3_bucket, "darts_xgb_gzip.gz")
#station_locs = load_model_from_s3(s3_bucket, "ride_locations.pkl")
cluster_model = load_model_from_s3(s3_bucket, "km_clusters.pkl")
st.write('Models loaded')

#station_locs.rename(columns={'lng':'lon'}, inplace=True)
#station_locs = station_locs[["lat", "lon"]]
# station_locs['color'] = '#000000'
# station_locs['size'] = 1

cluster_df = pd.DataFrame(cluster_model.cluster_centers_, columns=["start_lat", "start_lng"])
cluster_df.rename(columns={'start_lng':'lon', 'start_lat':'lat'}, inplace=True)
cluster_labels = range(0, len(cluster_df))
cluster_df['color'] = '#ff0000'
cluster_df['size'] = 12
# cluster_df = pd.concat([cluster_df, station_locs])


def add_time_features(dt_series):
    """create exogenous time-related features"""
    
    season_month = {12:'Winter', 1:'Winter', 2:'Winter',
                    3:'Spring', 4:'Spring', 5:'Spring',
                    6:'Summer', 7:'Summer', 8:'Summer',
                    9:'Autumn', 10:'Autumn', 11:'Autumn'}

    dt_series = dt_series.to_frame()
    dt_series['date'] = dt_series.index.date
    dt_series['year'] = dt_series.index.year
    dt_series['month'] = dt_series.index.month
    dt_series['hour'] = dt_series.index.hour
    dt_series['day_of_week'] = dt_series.index.dayofweek
    dt_series['weekend'] = dt_series.index.dayofweek>5
    
    dt_series['season'] = dt_series['month'].map(season_month)

    holidays = calendar().holidays(start=dt_series.index.min(), end=dt_series.index.max())
    dt_series['is_holiday'] = pd.to_datetime(dt_series.loc[:, 'date']).isin(holidays)
    dt_series['is_workday'] = ~(dt_series['weekend'] | dt_series['is_holiday'])
    dt_series['rush_hour'] = ((dt_series['hour'] > 5) & (dt_series['hour'] < 9)) | ((dt_series['hour'] > 16) & (dt_series['hour'] < 20))
    
    dt_series[['year', 'day_of_week', 'weekend']] = dt_series[['year', 'day_of_week', 'weekend']].astype("category")
    dt_series['rush_hour'] = ((dt_series['hour'] >= 6) & (dt_series['hour'] < 10)) | ((dt_series['hour'] >= 16) & (dt_series['hour'] < 20))
    #dt_series['condition'].replace({'rainy':0, 'cloudy':1, 'clear':2}, inplace=True)
    dt_series['season'].replace({'Winter':0, 'Autumn':1, 'Spring':1, 'Summer':2}, inplace=True)

    return dt_series


def predict(steps, exog):
    return model.predict(
        series=model.training_series,
        n=steps,
        future_covariates=exog,
    )

st.title('Bikeshare Demand Prediction')
st.map(cluster_df,
        latitude='lat',
        longitude='lon',
        size='size',
        color='color',
       )

date_range = st.date_input("Select date range for prediction:",
                           value= (datetime.date(2023, 7, 1), datetime.date(2023,7, 8)),
                           min_value=datetime.date(2023, 7, 1),
                           max_value=datetime.date(2023, 7, 31),
                           )

datetime_range = pd.date_range(start=date_range[0], end=date_range[1], freq='H')
dt_series = pd.Series(index=datetime_range)

train_end = model.training_series.time_index[-1].date()

start_pred = ((date_range[0] - train_end).days - 1) * 24 
end_pred = ((date_range[1] - train_end).days - 1)  * 24

dt_df = add_time_features(dt_series)
dt_df = dt_df[['month', 'season', 'hour', 'is_workday', 'is_holiday', 'rush_hour']]

clus_str = [f'clus{c}' for c in cluster_labels]

clus_for_pred = st.multiselect(
                    "Select clusters for prediction:",
                    options=clus_str,
                    default='clus0'
                )

if st.button('Predict'):
    prediction = predict(steps=end_pred, exog=dt_df)

    #st.write("selected clusters for prediction: ", clus_for_pred)
    st.write('Made prediction')
    pred_df = prediction.pd_dataframe()
    st.line_chart(pred_df, y=clus_for_pred)