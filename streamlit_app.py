import streamlit as st
import boto3

import datetime
import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.models import XGBModel

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import joblib
import pickle


s3_bucket = 'cabi-model-artefacts'

# Function to read the pickled model from S3
def load_model_from_s3(bucket, file_name):
    s3 = boto3.resource('s3')
    obj = s3.Object(bucket, file_name)
    model_bytes = obj.get()['Body'].read()
    return pickle.loads(model_bytes)

# Load the model from S3
model = load_model_from_s3(s3_bucket, "darts_xgb_clus_alldata_final.pkl")
station_locs = load_model_from_s3(s3_bucket, "ride_locations.pkl")

s3 = boto3.resource('s3')
obj = s3.Object(bucket, "km_clusters_busystations.joblib")
model_bytes = obj.get()['Body'].read()
cluster_model = joblib.loads(model_bytes)

station_locs.rename(columns={'lng':'lon'}, inplace=True)
station_locs = station_locs[["lat", "lon"]]
station_locs['color'] = 'k'
station_locs['size'] = 1

cluster_df = pd.DataFrame(cluster_model.cluster_centers_, columns=["lat", "lon"])
cluster_labels =  cluster_df.index()
cluster_df['color'] = 'r'
cluster_df['size'] = 2
cluster_df = pd.concat([cluster_df, station_locs])

#Caching the model for faster loading
@st.cache

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
    
    dt_series[['year', 'day_of_week', 'weekend']] = exog[['year', 'day_of_week', 'weekend']].astype("category")
    exog['rush_hour'] = ((exog['hour'] >= 5) & (exog['hour'] < 10)) | ((exog['hour'] >= 15) & (exog['hour'] < 20))
    #exog['condition'].replace({'rainy':0, 'cloudy':1, 'clear':2}, inplace=True)
    exog['season'].replace({'Winter':0, 'Autumn':1, 'Spring':1, 'Summer':2}, inplace=True)

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

train_end = model.training_range[1].date()

start_pred = ((date_range[0] - train_end).days - 1) * 24 
end_pred = ((date_range[1] - train_end).days - 1)  * 24

dt_df = add_time_features(dt_series)
dt_df = dt_df[['month', 'season', 'hour', 'is_workday', 'is_holiday', 'rush_hour']]

clus_for_pred = st.multiselect(
                    "Select clusters for prediction:",
                    options=cluster_labels,
                )

if st.button('Predict'):
    prediction = predict(steps=end_pred, exog=dt_df)
    pred_df = prediction.pd_dataframe()

    st.line_chart(pred_df['clus0'])
    st.line_chart(pred_df['clus1'])
