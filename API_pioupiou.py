import json
import pandas as pd
import numpy as np
import requests
import datetime
data=pd.DataFrame()


######Pïoupiou API:http://developers.pioupiou.fr/api/archive/
data=pd.DataFrame()
data_int=pd.DataFrame()

#Identify the months and year that will be queried
month_query=[5,6,7,8,9]
year_query=[2019]

#Run loop beacause the API can only be queried for 1 month
for j in year_query:
    for i in month_query:
        response = requests.get("http://api.pioupiou.fr/v1/archive/296?start={}-{}&stop={}-{}".format(j,i,j,i+1))
        apidata=response.json()['data']
        apicolumns=response.json()['legend']

        data_int=pd.DataFrame(apidata,columns=apicolumns)
        data=data.append(data_int)


#Make all date time items in usable format
data['datetime'] =pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
data['date']=data['datetime'].dt.date
data['time']=data['datetime'].dt.time

#Calculate the time difference before the time when we want to assess the thermal wind
timenow=datetime.datetime.now()
data['time_before_detection']=timenow-data['datetime']


