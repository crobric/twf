import json
import pandas as pd
import numpy as np
import requests
import datetime
data=pd.DataFrame()

response = requests.get("http://api.pioupiou.fr/v1/archive/296?start=last-hour&stop=now")
apidata=response.json()['data']
apicolumns=response.json()['legend']

data=pd.DataFrame(apidata,columns=apicolumns)


#data['time'] = data['time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ')
data['datetime'] =pd.to_datetime(data['time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
data['date']=data['datetime'].dt.date
data['time']=data['datetime'].dt.time

timenow=datetime.datetime.now()
data['time_before_detection']=timenow-data['datetime']

data
print(response.json())
