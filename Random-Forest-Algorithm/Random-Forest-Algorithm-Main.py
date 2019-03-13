import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier

#Loading the iris dataset into a variable
iris = load_iris()

#Loading the iris dataset into a pandas dataframe.
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#Makes a new column with the values from iris.target
df['target'] = iris.target

#Makes a new column with target_names, taken from target.
df['iris'] = df.target.apply(lambda x: iris.target_names[x])

#Drops the target and iris columns and stores the rest of the dataframe in the variable X.
X = df.drop(['target', 'iris'], axis='columns')

#Stores the target column in the variable y. This is what we want to predict.
y = df.target

#Preparing to train the RandomForestClassifier model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creating a RandomForestClassifier model. The default n estimators = , but I have changed it.
model = RandomForestClassifier(n_estimators=12)

#Training the model with the split data and printing an accuracy score for the model.
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#Storing the predictions for X (the predictions) in y_predicted.
y_predicted = model.predict(X_test)

#Now setting up a confusion matrix so we can better understand the models predictions.
cm = confusion_matrix(y_test, y_predicted)

#Setting up a heatmap using seaborn and matplotlib so that we can visualise the matrix.
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
