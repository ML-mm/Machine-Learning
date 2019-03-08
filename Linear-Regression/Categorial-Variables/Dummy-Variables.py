import pandas as pd
import numpy as np
from sklearn import linear_model

#Load the csv file into pandas dataframe.
df = pd.read_csv("carprices.csv")

#Rename the columns how I like so it makes it more simple for me to access.
df.rename(columns={
    'Car Model': 'Car_Model',
    'Sell Price($)': 'Sell_Price_in_USD',
}, inplace=True)

#Returns dummy variable columns
dummies = pd.get_dummies(df.Car_Model)

#Concatenate dummies dataframe with my original dataframe. This also brings the columns together using axis=1.
df_dummies = pd.concat([df, dummies], axis=1)

#Drops the Car_Model column
df_dummies.drop('Car_Model', axis=1, inplace=True)

#Dropping a single dummy variable column in order to avoid the dummy variable trap.
df_dummies.drop('BMW X5', axis=1, inplace=True)

#X is all the columns except the sell price, because the sell price is the dependant variable (y).
X = df_dummies.drop('Sell_Price_in_USD', axis=1)

#This is the dependant variable.
y = df_dummies.Sell_Price_in_USD

#Create an object for linear regression.
reg = linear_model.LinearRegression()

#Train the linear regression model, using the data points.
reg.fit(X, y)

#Predicts the Selling Price of a Mercedes Benz, which is 4 years old and has 45000 mileage.
print(reg.predict([[45000, 4, 0, 1]]))

#Predicts the Selling Price of a BMW X5, which is 7 years old and has 86000 mileage.
print(reg.predict([[86000, 7, 0, 0]]))


