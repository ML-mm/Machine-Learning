import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from sklearn import linear_model
import math
from sklearn.model_selection import train_test_split

#Load the csv file into pandas dataframe.
df = pd.read_csv("titanic.csv")

#Defining a function that allows us to view all of the columns that are present in a pandas dataframe.
#Now if we print a dataframe, it will look much more complete.
def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None

#Now calling the function that is above.
set_pandas_options()

#Dropping many columns from the dataframe and storing it in a variable.
inputs = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'],  axis='columns')

#Returns dummy variable columns.
dummies = pd.get_dummies(df["Sex"])

#Concatenate dummies dataframe with my inputs dataframe. This also brings the columns together using axis=1.
df_dummies = pd.concat([inputs, dummies], axis=1)

#Drops the Survived column
df_dummies.drop('Survived', axis=1, inplace=True)

#Drops the Sex column.
df_dummies.drop('Sex', axis=1, inplace=True)

#Dropping a single dummy variable column in order to avoid the dummy variable trap.
df_dummies.drop('male', axis=1, inplace=True)

#Finding the average age and rounding it up with the Math module.
g = math.ceil(df['Age'].mean())

#Replace NaN (not available) values for specific columns with whatever value I want the dataframe to show instead.
df_dummies = df_dummies.fillna({'Age': g})

#X has all the relevant columns except 'Survived', because we want to predict whether someone Survived or not based on the X input.
X = df_dummies

#This is what we want to predict: The survived column stored in y.
y = df.Survived

#Storing the logistic regression model into a variable.
#solver=lbfgs is used to solve an error message.
reg = linear_model.LogisticRegression(solver='lbfgs')

#Preparing to train the Logistic Regression model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Training the logistic regression model with the split data sets.
reg.fit(X_train, y_train)

#Prints the accuracy of the model.
print(reg.score(X_test, y_test))



