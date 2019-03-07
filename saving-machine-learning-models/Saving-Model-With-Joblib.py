import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.externals import joblib

#Pickle will allows us to save our linear regression training model for reuse.

#Load the csv file into pandas dataframe
df = pd.read_csv("canada_per_capita_income.csv")

#Rename the columns from the csv file.
df.columns = ['year', 'Per_capita_income_in_USD']

#Create an object for linear regression
reg = linear_model.LinearRegression()

#Train the linear regression model, using the data points.
reg.fit(df[['year']], df.Per_capita_income_in_USD)

#This prints out a prediction for the income in 2020.
print(reg.predict([[2020]]))

#It's recommended to use joblib instead of pickle if you are dealing with large numpy arrays.

#This saves the training model into a file called reg_joblib.
joblib.dump(reg, 'reg_joblib')

#This loads the model from the file into new_reg.
new_reg = joblib.load('reg_joblib')

#Using the trained model from the joblib file to predict the income for 2022.
print(new_reg.predict([[2022]]))




