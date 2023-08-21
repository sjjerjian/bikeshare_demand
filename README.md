# bikeshare_demand
## Predicting demand in D.C. Capital Bikeshare network

### Overview
This was a project I undertook for the Johns Hopkins Professional Development and Careers Office Data Science Fellowship Program.
Over 2 months, on a part-time basis, my goal was to build a forecasting model for the Capital Bikeshare rental bike scheme in Washington D.C.

A record of individual rides is [publicly released](https://capitalbikeshare.com/system-data) each month. 
I conducted my exploratory data analysis and model training using Sagemaker Studio Lab notebooks, Amazon's version of Google colab.


### Notebooks

`tripdata_preproc.ipynb` - reading data from AWS S3, and some basic preprocessing
`rides_eda.ipynb` -  exploratory data analysis of temporal patterns in rides
`rides_clustering.ipynb`  - grouping of stations by k-means of geographic locations
`station_connectivity.ipynb` - graphing of start-end connections between stations
`ts_forecast_allrides.ipynb` - forecasting model for ride counts across entire network (univariate time series)
`ts_forecast_clusters.ipynb` - forecasting model for ride counts across each cluster
`weather_eda.ipynb`
