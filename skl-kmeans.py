#a class demonstrating kmeans in scikit learn
#@author Jim Nolan

#matrix math
import numpy as np
#graphing
import matplotlib.pyplot as plt
#dataframe and analysis
import pandas as pd
#Machine learning library scikit learn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


#variable to store our file name
filepath="P:\python\\bitcoin\\btc.csv"
#read the CSV file into a dataframe
df=pd.read_csv(filepath)

#extract the columns of interest from the dataframe
x=df.priceUSD
y=df.txCount

#plot the raw data
plt.ticklabel_format(axis='both', style='plain')
plt.scatter(x,y)
plt.xlabel('Price')
plt.ylabel('Transaction Count')
plt.show()

#put the two 1D arrays into a matrix
X=np.array(list(zip(x,y))).reshape(len(x),2)

#plot the elbow
distortions = []
K=range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    distortions.append(kmeanModel.inertia_)

fig=plt.figure(figsize=(1,20))
plt.plot(range(1,20), distortions)
plt.grid(True)
plt.title('Elbow Curve')
plt.xlabel("Number of Clusters")
plt.show()

#create the actual kmeans clustering output
kmeans = KMeans(n_clusters=7)
kmeansoutput=kmeans.fit(X)
#assign the data to it's cluster
y_means=kmeans.predict(X)

plt.figure('Cluster K-Means')
plt.scatter(X[:, 0], X[:, 1], c=y_means)
plt.xlabel('Price')
plt.ylabel('Transaction Count')
plt.title('Cluster K-Means')

#plot the centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.show()





