import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

#Loads the dataset iris from sklearn and stores it in a variable.
iris = datasets.load_iris()

#Preparing to train the Logistic Regression model. Splitting the data set: 20% for testing and 80% for training.
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

#Storing the Logistic Regression model into a variable and then training it with the split data sets.
#solver=lbfgs is used to solve an error message.
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

#Printing the accuracy of the model.
print(model.score(X_test, y_test))

#Stores predictions for y in y_predicted.
y_predicted = model.predict(X_test)

#Now setting up a confusion matrix so we can better understand the models predictions.
cm = confusion_matrix(y_test, y_predicted)

#Setting up a heatmap using seaborn and matplotlib so that we can visualise the matrix.
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()