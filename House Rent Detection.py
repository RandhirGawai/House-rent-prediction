# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 20:16:11 2022

@author: Randhir
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("F:/LnB PROGRAM/new project/Project 1/House_Rent_Dataset.csv")
print(data.head())
#Before moving forward, let’s check if the data contains null values or not:
print(data.isnull().sum())
#Let’s have a look at the descriptive statistics of the data:
print(data.describe())

#Now let’s have a look at the mean, median, highest, and lowest rent of the houses:
    
print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")

#Now let’s have a look at the rent of the houses in different cities according to the number of bedrooms, halls, and kitchens:

figure = px.bar(data, x=data["City"], 
                y = data["Rent"], 
                color = data["BHK"],
            title="Rent in Different Cities According to BHK")
figure.show()
#Now let’s have a look at the rent of the houses in different cities according to the area type:

figure = px.bar(data, x=data["City"], 
                y = data["Rent"], 
                color = data["Area Type"],
            title="Rent in Different Cities According to Area Type")
figure.show()

#Now let’s have a look at the rent of the houses in different cities according to the furnishing status of the house:

figure = px.bar(data, x=data["City"], 
                y = data["Rent"], 
                color = data["Furnishing Status"],
            title="Rent in Different Cities According to Furnishing Status")
figure.show()

#Now let’s have a look at the rent of the houses in different cities according to the size of the house:
    
figure = px.bar(data, x=data["City"], 
                y = data["Rent"], 
                color = data["Size"],
            title="Rent in Different Cities According to Size")
figure.show()

#Now let’s have a look at the number of houses available for rent in different cities according to the dataset:
    
cities = data["City"].value_counts()
label = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Number of Houses Available for Rent')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()


#Now let’s have a look at the number of houses available for different types of tenants:
    
# Preference of Tenant
tenant = data["Tenant Preferred"].value_counts()
label = tenant.index
counts = tenant.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Preference of Tenant in India')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()

#Now I will convert all the categorical features into numerical features that we need to train a house rent prediction model:
    
data["Area Type"] = data["Area Type"].map({"Super Area": 1, 
                                           "Carpet Area": 2, 
                                           "Built Area": 3})
data["City"] = data["City"].map({"Mumbai": 4000, "Chennai": 6000, 
                                 "Bangalore": 5600, "Hyderabad": 5000, 
                                 "Delhi": 1100, "Kolkata": 7000})
data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0, 
                                                           "Semi-Furnished": 1, 
                                                           "Furnished": 2})
data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors/Family": 2, 
                                                         "Bachelors": 1, 
                                                         "Family": 3})
print(data.head())

#Now I will split the data into training and test sets:
    
#splitting data
from sklearn.model_selection import train_test_split
x = np.array(data[["BHK", "Size", "Area Type", "City", 
                   "Furnishing Status", "Tenant Preferred", 
                   "Bathroom"]])
y = np.array(data[["Rent"]])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.10, 
                                                random_state=42)

#Now let’s train a house rent prediction model using an LSTM neural network model: Long Short Term Memory Networks. It is a type of recurrent neural network that is commonly used for regression and time series forecasting in machine learning.

from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, 
               input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=21)

#Now here’s how to predict the rent of a housing property using the trained model:
    
print("Enter House Details to Predict Rent")
a = int(input("Number of BHK: "))
b = int(input("Size of the House: "))
c = int(input("Area Type (Super Area = 1, Carpet Area = 2, Built Area = 3): "))
d = int(input("Pin Code of the City: "))
e = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Number of bathrooms: "))
features = np.array([[a, b, c, d, e, f, g]])
print("Predicted House Price = ", model.predict(features))