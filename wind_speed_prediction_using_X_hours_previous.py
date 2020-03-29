#!/usr/bin/env python
# coding: utf-8

# In[131]:


import json
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import scikitplot as skplt
# Set the style
plt.style.use('seaborn')


# from tf.keras.models import Sequential  # This does not work!
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[2]:




import json
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression


def query_pioupiou_API(month_query=[5,6,7,8,9,10],year_query=[2016,2017,2018,2019],station_id=308):

    data=pd.DataFrame()
    data_int=pd.DataFrame()

    #Run loop beacause the API can only be queried for 1 month
    for j in year_query:
        for i in month_query:
            response = requests.get("http://api.pioupiou.fr/v1/archive/{}?start={}-{}&stop={}-{}".format(station_id,j,i,j,i+1))
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
    filepath="cache/pioupiou_sauveterre.csv"
    data.to_csv(filepath)
    print('File written succesfully in {}'.format(filepath))
    
    return filepath
    
    
def query_open_weather_api(latitude='46,547764',longitude='-1,831054'):

    ###Calling weather forecast API from open weather, 5 days forecast looking forward with data every 3 hour

    api_key='51fa8091162304b5aebae18745810d61'

    response = requests.get("http://api.openweathermap.org/data/2.5/forecast?lat={}&lon={}&appid={}&units=metric".format(latitude,longitude,api_key))
    wind_speed=[]
    date=[]
    wind_direction=[]
    humidity=[]
    temperature=[]
    df=pd.json_normalize(response.json())

    for i in range(len(df.list[0])):

        wind_speed.append(df.list[0][i]['wind']['speed'])
        wind_direction.append(df.list[0][i]['wind']['deg'])
        temperature.append(df.list[0][i]['main']['temp'])
        humidity.append(df.list[0][i]['main']['humidity'])
        date.append(datetime.datetime.fromtimestamp(df.list[0][i]['dt']))

    data=[wind_speed,date,wind_direction,humidity,temperature]
    names=['wind_speed','date','wind_direction','humidity','temperature']
    forecast_df=pd.DataFrame(dict(zip(names, data)), columns=names)
    forecast_df['wind_speed']=forecast_df['wind_speed']*1.94384 #Convert to knot
    
    filepath="cache/forecast_latest.csv"
    forecast_df.to_csv(filepath,index=False)
    print('File written succesfully in {}'.format(filepath))
    
    return filepath
    
    
    

def collect_observation_and_forecast(observation_path="cache/pioupiou_sauveterre.csv",forecast_path="cache/atlas_sauveterre.csv"):
    
    #Collecting data
    pioupiou=pd.read_csv(observation_path)
    pioupiou['datetime']=pd.to_datetime(pioupiou['datetime'])
    
    atlas=pd.read_csv(forecast_path)

    #Formating datetime values in order to create a similar dateindex in both dataframes
    atlas['datetime']=pd.to_datetime(atlas['time'], format='%Y-%m-%dT%H:%M:%SZ')
    atlas['date']=(atlas['datetime'].dt.date) 
    atlas['hour']=(atlas['datetime'].dt.hour) 
    atlas['time_compare']=atlas['date'].astype(str)+'/'+atlas['hour'].astype(str)


    pioupiou['date']=(pioupiou['datetime'].dt.date) 
    pioupiou['hour']=(pioupiou['datetime'].dt.hour) 
    pioupiou['time_compare']=pioupiou['date'].astype(str)+'/'+pioupiou['hour'].astype(str)
    pioupiou['wind_speed_avg']=pioupiou['wind_speed_avg'].apply(lambda x : float(x))

    print('Shape of pioupiou database is : {}' .format(pioupiou.shape))
    print('Shape of forecast database is : {}' .format(atlas.shape))
    
    return pioupiou,atlas


def identify_good_wind_conditions(observed_data):
   ###Observed data : Dataframe
    
    pioupiou=observed_data.copy()
    
    ##Create dataframe with real and forecast wind in order to assess if there is thermal wind or not
    pioupiou_data=pioupiou.groupby('time_compare',as_index=False, sort=False).agg('mean') #Grouping at same level as the atlas data

    thermal_detection=pd.DataFrame()
    forecast_data=atlas[['direction.wind.true','speed.wind.true','time_compare','datetime','date','hour']]

    thermal_detection['time_compare']=pioupiou_data['time_compare']
    thermal_detection['wind_direction_real']=pioupiou_data['wind_heading']
    thermal_detection['wind_speed_real']=pioupiou_data['wind_speed_avg']

    thermal_detection=thermal_detection.set_index('time_compare').join(forecast_data.set_index('time_compare'))
    thermal_detection.rename(columns={'direction.wind.true':'wind_direction_forecast','speed.wind.true':'wind_speed_forecast'}, inplace=True)
    thermal_detection
    
    ###Searching for winds above 14 knots and within acceptable incoming wind direction and during daylight hours
    #We create our database for valid condition

    wind_min=14
    min_duration=2  #Minimum number of consecutive hours with requested wind
    heading_limits=[180,360]
    acceptable_hour=[6,21]
    thermal=thermal_detection.copy().reset_index()

    #pioupiou_day_mask=(pioupiou['date']==datetime.date(2019,7,23)) & (pioupiou['hour']>=12)
    pioupiou_wind_min_mask=(thermal['wind_speed_real']>=wind_min) & (thermal['wind_direction_real']>=heading_limits[0]) & (thermal['wind_direction_real']<=heading_limits[1])     & (thermal['hour']>=acceptable_hour[0]) & (thermal['hour']<=acceptable_hour[1])

    #Apply the filtering mask to the df
    thermal_wind_min=thermal[pioupiou_wind_min_mask]

    ##Make sure that the days when there is wind, there is wind in several hours in a row
    thermal_wind_days=thermal_wind_min.copy()

    # For each day in the database, check if the hours identified are following each others
    for day in thermal_wind_min['date'].unique():
        t=1
        mask=thermal_wind_min['date']==day ##Create mask to check each day independently
        dataint=thermal_wind_min[mask]

        t1=-100
        for hour in dataint['hour']: ## For each day, check if the hours are following each other and increment t if yes
            t2=hour
            if t2-t1==1:
                t=t+1
            t1=t2

        if t<min_duration: ## if t has not been incremented enough, remove the day from the database
            thermal_wind_days = thermal_wind_days[thermal_wind_days.date != day]

    #Final list of days with conditions to be used
    days_with_wind=thermal_wind_days.copy()

    print('We identify {} hours with conditions that can be used for kitesurfing between {} and {}'.format(days_with_wind['wind_speed_real'].size,thermal_wind_days['date'].min(),thermal_wind_days['date'].max()))
    
    return days_with_wind

def prepare_feature_and_labels_for_prediction_speed(observation, forecast):
    #Dataframes with resampled index by hour
    #The output variable is only wind speed
    
    #Prepare the features for ML algorithm
    data=forecast.join(observation)
    data=data.dropna()


    # Labels are the values we want to predict, convert them to array
    labels = data[['observed_wind']]
    label_list = list(labels.columns)
    labels = labels.values[:]
    print(type(labels))
    print("Labels shape:", labels.shape)
    print("Label list :", label_list)
    
    # Remove the labels from the features
    features= data.drop(columns=['observed_wind','wind_heading'])
    # Saving feature names for later use
    feature_list = list(features.columns)

    #convert to array
    features=features.values[:]

    print(type(features))
    print("Features shape:", features.shape)
    print("Feature list :", feature_list)
    
    return features, feature_list, labels, label_list


def prepare_feature_and_labels_for_prediction_speed_and_heading(observation, forecast):
    #Dataframes with resampled index by hour
    #The output variable is speed and wind direction
    
    #Prepare the features for ML algorithm
    data=forecast.join(observation)
    data=data.dropna()


    # Labels are the values we want to predict, convert them to array
    labels = data[['observed_wind','wind_heading']]
    label_list = list(labels.columns)
    labels = labels.values[:]
    print(type(labels))
    print("Labels shape:", labels.shape)
    print("Label list :", label_list)
    
    # Remove the labels from the features
    features= data.drop(columns=['observed_wind','wind_heading'])
    # Saving feature names for later use
    feature_list = list(features.columns)

    #convert to array
    features=features.values[:]

    print(type(features))
    print("Features shape:", features.shape)
    print("Feature list :", feature_list)
    
    return features, feature_list, labels, label_list


def predict_conditions_5_days(model,latitude='46,547764', longitude='-1,831054',forecast_path="cache/forecast_latest.csv"):

    ###Opening the latest forecast path in cache
    forecast_df=pd.read_csv(forecast_path)

    future_forecast=forecast_df.drop(columns=['date'])
    future_kite=model.predict(future_forecast)
    forecast_df=forecast_df.join(pd.DataFrame(future_kite,columns=['kite_forecast']))

    return forecast_df


# In[3]:


##Collect data
pioupiou,atlas=collect_observation_and_forecast()
observation=pioupiou.copy()
forecast=atlas.copy()


# In[4]:


pioupiou.columns


# In[5]:


#Resample the data and remove the NaN values when there is no wind observation

observation=pioupiou.copy()
observation=observation[['datetime','wind_speed_avg','wind_heading']]
observation=observation.set_index('datetime').resample('60T').mean()
observation=observation.interpolate(limit=3) #fill in missing values if max 3 hours of data are missing, otherwise will be dropped
#observation=observation.dropna()
observation=observation.rename(columns={'wind_speed_avg':'observed_wind'})
observation


# # Add the N previous hours of data in the input data

# In[109]:


def derive_nth_day_feature(df, features, N):
    for feature in features:
        if feature != 'hour' and feature != 'day_of_year':
            for i in range(1, N):
                col_name = "{}_{}".format(feature, i)
                df[col_name] = df[feature].shift(-i)
    return df

N=2
forecast=atlas.copy()

forecast=forecast.set_index('datetime')
forecast=forecast.drop(columns=['Unnamed: 0', 'time','time_compare','latitude','date','hour',
           'longitude','speed.wind.true.longitudinal', 'speed.wind.true.transverse','enthalpy.air',
           'speed.wind.true.u', 'speed.wind.true.v'])

#Resample by hour
forecast=forecast.resample('60T').mean()

#Add missing values
forecast=forecast.interpolate(limit=3).reset_index()
forecast['day_of_year']=forecast['datetime'].dt.dayofyear
forecast['hour']=forecast['datetime'].dt.hour
forecast=forecast.set_index('datetime')

#forecast=forecast.dropna()
forecast.plot(y='speed.wind.true', use_index=True)
 
#add the N previous hours of forecast   
columns_name=forecast.columns
forecast=derive_nth_day_feature(forecast, np.array(columns_name), N)


# In[110]:


forecast


# In[111]:


data=forecast.join(observation).reset_index()
data=data.dropna()
data=data[(data['day_of_year'] ==(135 or 161 or 162) )&(data['datetime'].dt.year==2019)]
ax = plt.gca()
data.plot(kind='line',x='datetime',y='speed.wind.true', ax=ax)
data.plot(kind='line',x='datetime',y='observed_wind', color='red', ax=ax)
plt.show


# # Prepare feature and labels for algorithm

# In[112]:


features, feature_list, labels, label_list=prepare_feature_and_labels_for_prediction_speed(observation, forecast)


# In[113]:


#Prepare data for the ML algorithms
features, feature_list, labels, label_list=prepare_feature_and_labels_for_prediction_speed(observation, forecast)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# # Scale data

# In[114]:


#Scale all data
x_scaler = MinMaxScaler(feature_range = (0, 1))
y_scaler = MinMaxScaler(feature_range = (0, 1))


X_train = x_scaler.fit_transform(train_features)
X_test = x_scaler.transform(test_features)
y_train = y_scaler.fit_transform(train_labels)
y_test = y_scaler.transform(test_labels)

print(X_train.shape)
print(y_train.shape)


# # Run models

# In[115]:


# #############################################################################
# Lasso
print('\nLASSO')

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
alpha = 0.00001

lasso = Lasso(alpha=alpha)

lasso_model = lasso.fit(X_train, y_train)
y_pred_lasso=lasso_model.predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso)
print(lasso)
print("r^2 on test data : %f" % r2_score_lasso)
##Accuracy test on non scaled data


y_pred_rescaled = y_scaler.inverse_transform(y_pred_lasso.reshape(1,-1))
y_test_rescaled=y_scaler.inverse_transform(y_test.reshape(1,-1))

# Calculate the errors
errors_square = metrics.mean_squared_error(y_test_rescaled,y_pred_rescaled,squared=False)
errors=(y_pred_rescaled-y_test_rescaled)


# Print out the mean absolute error (mae)
print("The Explained Variance: %.2f" % lasso.score(X_test, y_test))
print('Root Mean Squared Error:', errors_square)
print('Mean Error:', np.mean(errors))
skplt.estimators.plot_learning_curve(lasso, X_train, y_train.ravel())
plt.show()

# #############################################################################
# ElasticNet
print('\nELASTICNET')

from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7)

enet_model = enet.fit(X_train, y_train)
y_pred_enet=enet_model.predict(X_test)

r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r^2 on test data : %f" % r2_score_enet)

y_pred_rescaled = y_scaler.inverse_transform(y_pred_enet.reshape(1,-1))
y_test_rescaled=y_scaler.inverse_transform(y_test.reshape(1,-1))

# Calculate the errors
errors_square = metrics.mean_squared_error(y_test_rescaled,y_pred_rescaled,squared=False)
errors=(y_pred_rescaled-y_test_rescaled)


# Print out the mean absolute error (mae)
print("The Explained Variance: %.2f" % enet.score(X_test, y_test))
print('Root Mean Squared Error:', errors_square)
print('Mean Error:', np.mean(errors))
skplt.estimators.plot_learning_curve(enet, X_train, y_train.ravel())
plt.show()


# #############################################################################
# SVM
print('\nSVM')

from sklearn import svm

reg = svm.SVR()

svm_model = reg.fit(X_train, y_train.ravel())
y_pred_SVM=svm_model.predict(X_test)
r2_score_SVM = r2_score(y_test, y_pred_SVM.ravel())
print(reg)
print("r^2 on test data : %f" % r2_score_SVM)
##Accuracy test on non scaled data


y_pred_rescaled = y_scaler.inverse_transform(y_pred_SVM.reshape(1,-1))
y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(1,-1))

# Calculate the errors
errors_square = metrics.mean_squared_error(y_test_rescaled,y_pred_rescaled,squared=False)
errors=(y_pred_rescaled-y_test_rescaled)


# Print out the mean absolute error (mae)
print("The Explained Variance: %.2f" % reg.score(X_test, y_test))
print('Root Mean Squared Error:', errors_square)
print('Mean Error:', np.mean(errors))
skplt.estimators.plot_learning_curve(reg, X_train, y_train.ravel())

# #############################################################################
# RANDOM FOREST REGRESSOR
print('\nRANDOM FOREST REGRESSOR')
# Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf_model = rf.fit(X_train, y_train.ravel())
y_pred_RF=rf_model.predict(X_test)
r2_score_RF = r2_score(y_test, y_pred_RF.ravel())
print(rf)
print("r^2 on test data : %f" % r2_score_RF)
##Accuracy test on non scaled data


y_pred_rescaled = y_scaler.inverse_transform(y_pred_RF.reshape(1,-1))
y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(1,-1))

# Calculate the errors
errors_square = metrics.mean_squared_error(y_test_rescaled,y_pred_rescaled,squared=False)
errors=(y_pred_rescaled-y_test_rescaled)


# Print out the mean absolute error (mae)
print("The Explained Variance: %.2f" % rf.score(X_test, y_test))
print('Root Mean Squared Error:', errors_square)
print('Mean Error:', np.mean(errors))
skplt.estimators.plot_learning_curve(reg, X_train, y_train.ravel())


# # Using the Randon Forest output

# ## Check the importance of each feature
# 

# In[140]:


model=rf_model

# Get numerical feature importances
importances = list(rf_model.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

plt.figure()

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');


# # Visualize the full test set values

# In[138]:


y_pred_rescaled = y_scaler.inverse_transform(y_pred_RF.reshape(1,-1))
y_test_rescaled = y_scaler.inverse_transform(y_test.reshape(1,-1))
plt.figure()
signal_true=y_test_rescaled.ravel()
signal_pred=y_pred_rescaled.ravel()
plt.plot(signal_true, label='true')
plt.plot(signal_pred, label='pred')
plt.legend()
plt.show()

errors = (signal_pred - signal_true)
# Display the performance metrics
print('Root Mean Absolute Error:', round(np.mean(errors), 2), 'knots.')


# # Predict the whole training set of Atlas data

# In[118]:


##Predict the whole training set of Atlas data
X = x_scaler.fit_transform(features)
wind_predicted_scaled=model.predict(X)
wind_predicted=y_scaler.inverse_transform(wind_predicted_scaled.reshape(1,-1))

#Join the historical, observation and prediction sets
predict_df=forecast.copy()
predict_df=predict_df.join(observation)
predict_df=predict_df.dropna()

model_output=pd.DataFrame(wind_predicted.ravel(),columns=['predicted_wind'])
predict_df=predict_df.reset_index().join(model_output)

predict_df.head(30)


# In[125]:


predict_df.corr()[['observed_wind']].sort_values('observed_wind')


# # Plot the forecast, observed and predicted wind for several days in a row

# In[136]:


data=predict_df.copy()

for i in np.arange(130,140,1):
    data_plot=data[(data['day_of_year'] == i )&(data['datetime'].dt.year==2019)]
    plt.figure()
    ax = plt.gca()
    data_plot.plot(kind='line',x='datetime',y='speed.wind.true', ax=ax)
    data_plot.plot(kind='line',x='datetime',y='observed_wind', color='red', ax=ax)
    data_plot.plot(kind='line',x='datetime',y='predicted_wind', color='green', ax=ax)
plt.show()


# # Visualize the decision tree

# In[107]:


# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# # Finally predict windy days in the future

# In[133]:


## Finally predict windy days in the future

#Sauveterre
latitude='46,5'
longitude='-1,78'
#forecast_path = query_open_weather_api(latitude,longitude)


forecast_df=pd.read_csv(forecast_path)
prediction_openweather=forecast_df.copy()

#Transform date to datetime
forecast_df['date']=pd.to_datetime(forecast_df['date'], format='%Y-%m-%d %H:%M:%S')

#Downsample to per hour samples and interpolate
forecast_df=forecast_df.set_index('date').resample('60T').mean()
forecast_df=forecast_df.interpolate(limit=3).reset_index()

#Add the required features for the model
forecast_df['day_of_year']=forecast_df['date'].dt.dayofyear
forecast_df['hour']=forecast_df['date'].dt.hour
future_forecast=forecast_df.drop(columns=['date'])
future_forecast=future_forecast.rename(columns={'wind_speed':'speed.wind.true',
                                'wind_direction':'direction.wind.true',
                                'humidity': 'ratio.humidity.air',
                                'temperature': 'temperature.air',
                                'day_of_year':'day_of_year', 
                                'hour':'hour'})

#Add the features that come from the previous hours of forecast
future_forecast=derive_nth_day_feature(future_forecast, np.array(future_forecast.columns), N)

future_forecast=future_forecast[feature_list].dropna()


#Scale inputs
future_forecast_scaled=x_scaler.fit_transform(future_forecast)

#Predict with model
future_kite_scaled=model.predict(future_forecast_scaled)

#De-Scale outputs
future_kite=y_scaler.inverse_transform(future_kite_scaled.reshape(1,-1))

#Merge the forecast and prediction
forecast_df=forecast_df.join(pd.DataFrame(future_kite.ravel(),columns=['predicted_wind']))

#Plot the result
plt.figure()
ax = plt.gca()
forecast_df.plot(kind='line',x='date',y='predicted_wind',color='green',ax=ax)
forecast_df.plot(kind='line',x='date',y='wind_speed', color='red', ax=ax)


# # Plot forecast for each day in the future

# In[137]:


for day in forecast_df['day_of_year'].unique():
    forecast_plt=forecast_df[forecast_df['day_of_year']==day]
    #Plot the result
    plt.figure()
    ax = plt.gca()
    forecast_plt.plot(kind='line',x='date',y='predicted_wind',color='green',ax=ax)
    forecast_plt.plot(kind='line',x='date',y='wind_speed', color='red', ax=ax)


# In[66]:


forecast_df.head(10)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Assess how many days can be used for kitesurf in this spot

# In[ ]:


#Assess when there is more observed wind than forecasted
threshold=5

observation_kite_ok=predict_df[predict_df['observed_wind']>14]
observation_kite_thermal=observation_kite_ok[observation_kite_ok['observed_wind']>(observation_kite_ok['speed.wind.true']+threshold)]
observation_kite_thermal['date']=observation_kite_thermal['datetime'].dt.date
observation_kite_thermal['month']=observation_kite_thermal['datetime'].dt.month
observation_kite_thermal['year']=observation_kite_thermal['datetime'].dt.year
observation_kite_thermal


# In[ ]:


###Searching for winds above 14 knots and within acceptable incoming wind direction and during daylight hours
#We create our database for valid condition

wind_min=14
min_duration=3  #Minimum number of consecutive hours with requested wind
heading_limits=[180,360]
acceptable_hour=[6,21]
thermal=observation_kite_thermal

#pioupiou_day_mask=(pioupiou['date']==datetime.date(2019,7,23)) & (pioupiou['hour']>=12)
pioupiou_wind_min_mask=(thermal['observed_wind']>=wind_min) & (thermal['wind_heading']>=heading_limits[0]) & (thermal['wind_heading']<=heading_limits[1]) & (thermal['hour']>=acceptable_hour[0]) & (thermal['hour']<=acceptable_hour[1])

#Apply the filtering mask to the df
thermal_wind_min=thermal[pioupiou_wind_min_mask]

##Make sure that the days when there is wind, there is wind in several hours in a row
thermal_wind_days=thermal_wind_min.copy()

# For each day in the database, check if the hours identified are following each others
for day in thermal_wind_min['date'].unique():
    t=1
    mask=thermal_wind_min['date']==day ##Create mask to check each day independently
    dataint=thermal_wind_min[mask]

    t1=-100
    for hour in dataint['hour']: ## For each day, check if the hours are following each other and increment t if yes
        t2=hour
        if t2-t1==1:
            t=t+1
        t1=t2

    if t<min_duration: ## if t has not been incremented enough, remove the day from the database
        thermal_wind_days = thermal_wind_days[thermal_wind_days.date != day]

#Final list of days with conditions to be used
days_with_wind=thermal_wind_days.copy()

print('We identify {} hours with conditions that can be used for kitesurfing between {} and {}'.format(days_with_wind['datetime'].size,thermal_wind_days['date'].min(),thermal_wind_days['date'].max()))
 


# In[ ]:


days_with_wind


# In[ ]:



##Check if we have at least
days_thermal=days_with_wind.groupby('date').agg(['count', 'mean'])
days_thermal=days_thermal[days_thermal[('hour','count')]>2]
days_thermal['month']=days_thermal.reset_index()['date'].dt.month
days_thermal['year']=days_thermal.reset_index()['date'].dt.year


#month_thermal=days_thermal.groupby(['year','month']).agg(['count', 'mean'])
days_thermal


# In[ ]:







# In[ ]:


days_thermal


# In[ ]:




