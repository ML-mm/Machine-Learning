import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Load the csv file into pandas dataframe.
df = pd.read_csv("HR_comma_sep.csv")

#Defining a function that allows us to view all of the columns that are present in a pandas dataframe.
#Now if we print a dataframe, it will look much more complete.
def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None

#Now calling the function that is above.
set_pandas_options()

#Including only the variables that have a significant impact on employee retention.
df2 = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

#Since salary has text data, it needs to be converted into variables. So this will get dummy variables for salary.
dummies = pd.get_dummies(df2.salary)

#This adds both the 2nd dataframe and the dummy variables together.
new_df = pd.concat([df2, dummies], axis=1)

#This then drops the salary column for the new dataframe
new_df.drop('salary', axis=1, inplace=True)

#The new dataframe is stored here.
X = new_df

#The column 'salary' from the old dataframe is stored here.
y = df.left

#Preparing to train the Logistic Regression model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Storing the Logistic Regression model into a variable and then training it with the split data sets.
#solver=lbfgs is used to solve an error message.
reg = LogisticRegression(solver='lbfgs')
reg.fit(X_train, y_train)

#Printing the accuracy of the model.
print(reg.score(X_test, y_test))



