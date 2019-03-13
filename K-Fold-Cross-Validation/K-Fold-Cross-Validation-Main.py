from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#import warnings filter
from warnings import simplefilter

#ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

#Loads the iris dataset.
iris = load_iris()

#Loads the iris dataset into a pandas dataframe.
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#Makes a new column with the values from iris.target
df['target'] = iris.target

#Makes a new column with target_names, taken from target.
df['iris'] = df.target.apply(lambda x: iris.target_names[x])

#Drops the target and iris columns and stores the rest of the dataframe in the variable X.
X = df.drop(['target', 'iris'], axis='columns')

#Stores the target column in the variable y. This is what we want to predict.
y = df.target

#Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Training 3 different machine learning models with the data.
model1 = cross_val_score(LogisticRegression(), X, y)
model2 = cross_val_score(SVC(), X, y)
model3 = cross_val_score(RandomForestClassifier(n_estimators=40), X, y)

#After finding out Support Vector Machine is the most accurate, I tune some parameters to try and make it more accurate.
scores1 = cross_val_score(SVC(C=2), X, y, cv=10)
scores2 = cross_val_score(SVC(C=5), X, y, cv=10)
scores3 = cross_val_score(SVC(gamma=10), X, y, cv=10)
scores4 = cross_val_score(SVC(kernel='linear'), X, y, cv=10)

print(np.average(scores1))
print(np.average(scores2))
print(np.average(scores3))
print(np.average(scores4))

#By tuning the parameter C to C=2 in SVC, we get the highest accuracy.






