import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#import warnings filter
from warnings import simplefilter

#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#Storing the digits dataset into a variable.
digits = datasets.load_digits()

#Loading the digits dataset into a dataframe.
df = pd.DataFrame(digits.data)

#Adding a new column with targets taken from the dataset.
df['target'] = digits.target

#df['actual_digits'] = df.target.apply(lambda x: digits.target_names[x])

#Dropping the target column and keeping the data only.
X = df.drop(['target'], axis='columns')

#Storing the target column into a variable.
y = df.target

#Preparing to train the Support Vector Machine model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Import Support Vector Machine, create an SVM classifier and store into a variable.
#Gamme defined as auto to avoid errors. The default is auto also.
model = SVC(gamma='auto')

#Train the SVC model, using the data points.
model.fit(X_train, y_train)

#Prints the accuracy of the model.
print(model.score(X_test, y_test))

#Now trying to modify and use different parameters to get the most accurate model.

#Modifying changing the regularization parameter.
model_C = SVC(C=5)
model_C.fit(X_train, y_train)
print(model_C.score(X_test, y_test))

#Modifying the gamma parameter.
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print(model_g.score(X_test, y_test))

#Modifying the kernel parameter
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)
print(model_linear_kernal.score(X_test, y_test))

#Changing the kernel parameter to 'linear' gives the most accurate result out of the 4 models.

