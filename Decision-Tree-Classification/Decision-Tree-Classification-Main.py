import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
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
inputs = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Survived'],  axis='columns')

#This stores the column 'Survived' (the values we want to predict) into a variable.
target_var = df['Survived']

#This sets up the LabelEncoder we will use for encoding the values within the column "Sex".
le_Sex = LabelEncoder()

#This creates a new column, 'Sex_New' which has all of the string data encoded into integers.
#This is done through using LabelEncoder to transform the data.
inputs['Sex_New'] = le_Sex.fit_transform(inputs['Sex'])

#This drops the Sex column as we already have this column encoded and added onto the inputs dataframe.
inputs.drop(['Sex'], axis='columns', inplace=True)

#Finding the average age and rounding it up with the Math module.
g = math.ceil(df['Age'].mean())

#Replace NaN (not available) values for specific columns with whatever value I want the dataframe to show instead.
New_Inputs = inputs.fillna({'Age': g})

#X has all the relevant columns except 'Survived', because we want to predict whether someone Survived or not based on the X input.
X = New_Inputs

#This is what we want to predict. The survived column will get stored in y.
y = target_var

#Stores the Decision Tree Classifier model into a variable.
Model = tree.DecisionTreeClassifier()

#Preparing to train the Decision Tree Classifier model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Training the Decision Tree Classifier model with the split data sets.
Model.fit(X_train, y_train)

#Prints the accuracy of the model.
print(Model.score(X_test, y_test))



