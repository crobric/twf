import json
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


def forecast_conditions_5_days(model ,latitude='46,547764', longitude='-1,831054'):

    ###Calling weather forecast API from open weather, 5 days forecast looking forward with data every 3 hour

    api_key ='51fa8091162304b5aebae18745810d61'
    latitude ='46,547764'
    longitude ='-1,831054'

    response = requests.get \
        ("http://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}&units=metric".format(latitude
                                                                                                     ,longitude
                                                                                                     ,api_key))

    wind_speed =[]
    date =[]
    wind_direction =[]
    humidity =[]
    temperature =[]
    df =pd.json_normalize(response.json())

    for i in range(len(df.list[0])):

        wind_speed.append(df.list[0][i]['wind']['speed'])
        wind_direction.append(df.list[0][i]['wind']['deg'])
        temperature.append(df.list[0][i]['main']['temp'])
        humidity.append(df.list[0][i]['main']['humidity'])
        date.append(datetime.datetime.fromtimestamp(df.list[0][i]['dt']))

    data =[wind_speed ,date ,wind_direction ,humidity ,temperature]
    names =['wind_speed' ,'date' ,'wind_direction' ,'humidity' ,'temperature']
    forecast_df =pd.DataFrame(dict(zip(names, data)), columns=names)



    future_forecast =forecast_df.drop(columns=['date'])
    future_kite =model.predict(future_forecast)
    forecast_df =forecast_df.join(pd.DataFrame(future_kite ,columns=['kite_forecast']))

    return forecast_df


def query_pioupiou_API(month_query=[5, 6, 7, 8, 9, 10], year_query=[2016, 2017, 2018, 2019], station_id=308):
    data = pd.DataFrame()
    data_int = pd.DataFrame()

    # Identify the months and year that will be queried
    month_query = [5, 6, 7, 8, 9, 10]
    year_query = [2016, 2017, 2018, 2019]
    station_id = 308

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
    data.to_csv("cache/pioupiou_sauveterre.csv")
    print('File written succesfully in {}'.format(filepath))

    return