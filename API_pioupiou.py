import json
import pandas as pd
import numpy as np
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression
from functions import *


######Pïoupiou API:http://developers.pioupiou.fr/api/archive/
data=pd.DataFrame()
data_int=pd.DataFrame()

#Identify the months and year that will be queried
month_query=[5,6,7,8,9,10]
year_query=[2016,2017,2018,2019]
station_id=308

latitude='46,547764'
longitude='-1,831054'

##Collect data
observation_path="cache/pioupiou_sauveterre.csv"
#observation_path = query_pioupiou_API(month_query,year_query,station_id)  #Remove comment in order to update
historical_data_path="cache/atlas_sauveterre.csv"

pioupiou,atlas=collect_observation_and_forecast(observation_path=observation_path,forecast_path=historical_data_path)

##identify windy days situations
windy_days=identify_good_wind_conditions(pioupiou,atlas)

#Create list of features and labels for the ML classification algorithm
features, labels, feature_list = prepare_feature_and_labels_for_classification(days_with_wind=windy_days, past_forecast=atlas)

###Run ML algorithm and train a model. Remove comment symbol in front of selected model

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Random Forest regressor
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

#Support vector Machine
from sklearn import svm
#rf = svm.SVC()

#DecisionTree
from sklearn.tree import DecisionTreeClassifier
rf = DecisionTreeClassifier()

# Train the model on training data
rf.fit(train_features, train_labels);


# Use the forest's predict method on the test data
print("Selected model:{}".format(type(rf)))
predictions = rf.predict(test_features)


# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# calculate accuracy
from sklearn import metrics
acc=metrics.accuracy_score(test_labels, predictions)
print('Accuracy:', round(acc*100, 2), '%.')


## Finally predict windy days in the future : collect next days data and predict using the model
forecast_path = query_open_weather_api(latitude,longitude)
prediction=predict_conditions_5_days(rf, forecast_path)
print(prediction)
