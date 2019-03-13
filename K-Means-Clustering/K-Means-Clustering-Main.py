from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

#This loads the iris dataset
iris = load_iris()

#This loads the iris dataset into a pandas dataframe.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#Drops the unnecessary columns
df.drop(['sepal length (cm)', 'sepal width (cm)'], axis='columns', inplace=True)

#This is the K Means Classification model with the number of clusters set to 3.
km = KMeans(n_clusters=3)
#Fits and predicts the dataframe. We get back y predicted. Algorithm identifies 3 clusters.
y_predicted = km.fit_predict(df)

#Adding the y predicted values to a new column: cluster.
df['cluster'] = y_predicted

#You can use this to double check the cluster numbers.
#df.cluster.unique()

#Seperating 3 clusters into 3 seperate dataframes. Returning all rows where cluster = n.
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

#We could use the scaler if the clusters do not look correctly grouped. But for this dataset it's not needed.
#There are 3 clusters which will be confirmed with the elbow technique.
#Just as an additional note, this makes the scale 0 - 1.

#scaler = MinMaxScaler()

#scaler.fit(df[['petal width (cm)']])
#df['petal width (cm)'] = scaler.transform(df[['petal width (cm)']])

#scaler.fit(df[['petal length (cm)']])
#df['petal length (cm)'] = scaler.transform(df[['petal length (cm)']])

#Setting up two subplots.
fig_2, ax = plt.subplots(1, 2, figsize=(20, 5))

#The first subplot will contain our scatter graph for the 3 different cluster columns as specified above.
ax[0].scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='blue')
ax[0].scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='green')
ax[0].scatter(df3['petal length (cm)'], df3['petal width (cm)'], color='yellow')
ax[0].scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='centroid')
ax[0].set_xlabel('petal length (cm)')
ax[0].set_ylabel('petal width (cm)')
ax[0].legend()

#Using the elbow technique to find the optimal value of k.
#sse = sum of squared error
#km.inertia_ = parameter which gives us the sum of squared error.
#KMeans = K Means Classification model.
sse = []
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

#This creates an elbow chart showing the sse for k (from range 1, 10) in the second subplot.
#And we can see that the optimal value for k is 3.
ax[1].set_xlabel('K')
ax[1].set_ylabel('Sum of squared error')
ax[1].plot(k_rng, sse, label='elbow plot')
ax[1].legend()

plt.show()
