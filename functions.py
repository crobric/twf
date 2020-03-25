

import json
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


def query_pioupiou_API(month_query, year_query, station_id):
    data = pd.DataFrame()
    data_int = pd.DataFrame()

    # Run loop beacause the API can only be queried for 1 month
    for j in year_query:
        for i in month_query:
            response = requests.get(
                "http://api.pioupiou.fr/v1/archive/{}?start={}-{}&stop={}-{}".format(station_id, j, i, j, i + 1))
            apidata = response.json()['data']
            apicolumns = response.json()['legend']

            data_int = pd.DataFrame(apidata, columns=apicolumns)
            data = data.append(data_int)

    # Make all date time items in usable format
    data['datetime'] = pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    data['date'] = data['datetime'].dt.date
    data['time'] = data['datetime'].dt.time

    # Calculate the time difference before the time when we want to assess the thermal wind
    timenow = datetime.datetime.now()
    data['time_before_detection'] = timenow - data['datetime']
    filepath = "cache/pioupiou_sauveterre.csv"
    data.to_csv(filepath)
    print('File written succesfully in {}'.format(filepath))

    return filepath


def query_open_weather_api(latitude='46,547764', longitude='-1,831054'):
    ###Calling weather forecast API from open weather, 5 days forecast looking forward with data every 3 hour

    api_key = '51fa8091162304b5aebae18745810d61'

    response = requests.get(
        "http://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}&units=metric".format(latitude,
                                                                                                     longitude,
                                                                                                     api_key))
    wind_speed = []
    date = []
    wind_direction = []
    humidity = []
    temperature = []
    df = pd.json_normalize(response.json())

    for i in range(len(df.list[0])):
        wind_speed.append(df.list[0][i]['wind']['speed'])
        wind_direction.append(df.list[0][i]['wind']['deg'])
        temperature.append(df.list[0][i]['main']['temp'])
        humidity.append(df.list[0][i]['main']['humidity'])
        date.append(datetime.datetime.fromtimestamp(df.list[0][i]['dt']))

    data = [wind_speed, date, wind_direction, humidity, temperature]
    names = ['wind_speed', 'date', 'wind_direction', 'humidity', 'temperature']
    forecast_df = pd.DataFrame(dict(zip(names, data)), columns=names)

    filepath = "cache/forecast_latest.csv"
    forecast_df.to_csv(filepath, index=False)
    print('File written succesfully in {}'.format(filepath))

    return filepath


def collect_observation_and_forecast(observation_path="cache/pioupiou_sauveterre.csv",
                                     forecast_path="cache/atlas_sauveterre.csv"):
    # Collecting data
    pioupiou = pd.read_csv(observation_path)
    pioupiou['datetime'] = pd.to_datetime(pioupiou['datetime'])

    atlas = pd.read_csv(forecast_path)

    # Formating datetime values in order to create a similar dateindex in both dataframes
    atlas['datetime'] = pd.to_datetime(atlas['time'], format='%Y-%m-%dT%H:%M:%SZ')
    atlas['date'] = (atlas['datetime'].dt.date)
    atlas['hour'] = (atlas['datetime'].dt.hour)
    atlas['time_compare'] = atlas['date'].astype(str) + '/' + atlas['hour'].astype(str)

    pioupiou['date'] = (pioupiou['datetime'].dt.date)
    pioupiou['hour'] = (pioupiou['datetime'].dt.hour)
    pioupiou['time_compare'] = pioupiou['date'].astype(str) + '/' + pioupiou['hour'].astype(str)
    pioupiou['wind_speed_avg'] = pioupiou['wind_speed_avg'].apply(lambda x: float(x))

    print('Shape of pioupiou database is : {}'.format(pioupiou.shape))
    print('Shape of forecast database is : {}'.format(atlas.shape))

    return pioupiou, atlas


def identify_good_wind_conditions(observed_data,old_forecast_data):
    pioupiou = observed_data.copy()
    atlas=old_forecast_data.copy()
    ##Create dataframe with real and forecast wind in order to assess if there is thermal wind or not
    pioupiou_data = pioupiou.groupby('time_compare', as_index=False, sort=False).agg(
        'mean')  # Grouping at same level as the atlas data

    thermal_detection = pd.DataFrame()
    forecast_data = atlas[['direction.wind.true', 'speed.wind.true', 'time_compare', 'datetime', 'date', 'hour']]

    thermal_detection['time_compare'] = pioupiou_data['time_compare']
    thermal_detection['wind_direction_real'] = pioupiou_data['wind_heading']
    thermal_detection['wind_speed_real'] = pioupiou_data['wind_speed_avg']

    thermal_detection = thermal_detection.set_index('time_compare').join(forecast_data.set_index('time_compare'))
    thermal_detection.rename(
        columns={'direction.wind.true': 'wind_direction_forecast', 'speed.wind.true': 'wind_speed_forecast'},
        inplace=True)
    thermal_detection

    ###Searching for winds above 14 knots and within acceptable incoming wind direction and during daylight hours
    # We create our database for valid condition

    wind_min = 14
    min_duration = 2  # Minimum number of consecutive hours with requested wind
    heading_limits = [180, 360]
    acceptable_hour = [6, 21]
    thermal = thermal_detection.copy().reset_index()

    # pioupiou_day_mask=(pioupiou['date']==datetime.date(2019,7,23)) & (pioupiou['hour']>=12)
    pioupiou_wind_min_mask = (thermal['wind_speed_real'] >= wind_min) & (
                thermal['wind_direction_real'] >= heading_limits[0]) & (
                                         thermal['wind_direction_real'] <= heading_limits[1]) \
                             & (thermal['hour'] >= acceptable_hour[0]) & (thermal['hour'] <= acceptable_hour[1])

    # Apply the filtering mask to the df
    thermal_wind_min = thermal[pioupiou_wind_min_mask]

    ##Make sure that the days when there is wind, there is wind in several hours in a row
    thermal_wind_days = thermal_wind_min.copy()

    # For each day in the database, check if the hours identified are following each others
    for day in thermal_wind_min['date'].unique():
        t = 1
        mask = thermal_wind_min['date'] == day  ##Create mask to check each day independently
        dataint = thermal_wind_min[mask]

        t1 = -100
        for hour in dataint[
            'hour']:  ## For each day, check if the hours are following each other and increment t if yes
            t2 = hour
            if t2 - t1 == 1:
                t = t + 1
            t1 = t2

        if t < min_duration:  ## if t has not been incremented enough, remove the day from the database
            thermal_wind_days = thermal_wind_days[thermal_wind_days.date != day]

    # Final list of days with conditions to be used
    days_with_wind = thermal_wind_days.copy()

    print('We identify {} hours with conditions that can be used for kitesurfing between {} and {}'.format(
        days_with_wind['wind_speed_real'].size, thermal_wind_days['date'].min(), thermal_wind_days['date'].max()))

    return days_with_wind


def prepare_feature_and_labels_for_classification(days_with_wind, past_forecast):
    # Prepare the features for ML algorithm
    atlas = past_forecast.copy()
    atlas['wind_condition'] = 'na'
    atlas['wind_condition'] = atlas['time_compare'].apply(
        lambda x: 1 if x in days_with_wind['time_compare'].values else 0)
    features = atlas.dropna()

    features = features.drop(columns=['Unnamed: 0', 'date', 'hour', 'datetime', 'time', 'time_compare', 'latitude',
                                      'longitude', 'speed.wind.true.longitudinal', 'speed.wind.true.transverse',
                                      'ratio.humidity.air',
                                      'speed.wind.true.u', 'speed.wind.true.v']).reset_index()

    # Labels are the values we want to predict
    labels = np.array(features['wind_condition'])

    # Remove the labels from the features

    features = features.drop(columns=['wind_condition', 'index'])
    # Saving feature names for later use
    feature_list = list(features.columns)
    # Convert to numpy array
    features = np.array(features)
    feature_list

    return features, labels, feature_list


def predict_conditions_5_days(model, forecast_path="cache/forecast_latest.csv"):
    ###Opening the latest forecast path in cache
    forecast_df = pd.read_csv(forecast_path)

    future_forecast = forecast_df.drop(columns=['date'])
    future_kite = model.predict(future_forecast)
    forecast_df = forecast_df.join(pd.DataFrame(future_kite, columns=['kite_forecast']))

    return forecast_df