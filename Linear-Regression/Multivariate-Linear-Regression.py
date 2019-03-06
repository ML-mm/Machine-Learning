import pandas as pd
import numpy as np
from sklearn import linear_model
import math

#Load the csv file into pandas dataframe.
df = pd.read_csv("hiring.csv")

#Rename the columns how I like so it makes it more simple for me to access.
df.rename(columns={
    'test_score(out of 10)': 'test_score_out_10',
    'interview_score(out of 10)': 'interview_score_out_10',
    'salary($)': 'salary_in_USD'
}, inplace=True)

#Create a variable, g, in order for it to contain the median of the test scores.
g = df.test_score_out_10.median()

#Replace NaN (not available) values for specific columns with whatever value I want the dataframe to show instead.
df = df.fillna({
    'experience': "zero",
    'test_score_out_10': g
})

#Replace all of the number strings in the column 'experience', with actual integers.
df.replace({"zero": 0, "five": 5, "two": 2, "seven": 7, "three": 3, "ten": 10, "eleven": 11}, inplace=True)

#Create an object for linear regression.
reg = linear_model.LinearRegression()

#Train the linear regression model, using the data points.
reg.fit(df[['experience', 'test_score_out_10', 'interview_score_out_10']], df.salary_in_USD)

#This prints out a prediction for someone's salary, given that they have 2 yrs of experience, scored 9 on the test and scored 6 for the interview.
print(reg.predict([[2, 9, 6]]))