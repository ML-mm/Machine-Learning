import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle

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

#1. Opens a new file, wb for writing binary data. 2. Saves the training model into this file.
#After running this, it creates the pickle file. The model is saved into that file, you can now reuse it.
with open('reg_pickle', 'wb') as f:
    pickle.dump(reg, f)

#1. Opens the pickle file in read mode ('rb'). 2. Model is now loaded from the file into new_reg.
with open('reg_pickle', 'rb') as r:
    new_reg = pickle.load(r)

#Using the trained model from the pickle file to predict the income for 2022.
print(new_reg.predict([[2022]]))
