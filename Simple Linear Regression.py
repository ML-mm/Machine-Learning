import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

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

#Using the matplotlib library, I created a scatter graph to show the relationship between years and per capita income.
#I also made an additional point in the graph to present the predicted income for 2020.
plt.xlabel('Year')
plt.ylabel('Per capita income (US$)')
plt.scatter(df.year, df.Per_capita_income_in_USD, label="Actual Data", color='r', s=20, marker="x")
plt.scatter(2020, reg.predict([[2020]]), label="Predicted Data", color='b', marker="x")
plt.legend()
plt.show()


