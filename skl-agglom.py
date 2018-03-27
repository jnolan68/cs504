#a class demonstrating kmeans and agglomerative clustering in scikit learn
#@author Jim Nolan

import time
#matrix math
import numpy as np
#graphing
import matplotlib.pyplot as plt
#dataframe and analysis
import pandas as pd
#Machine learning library scikit learn
import sklearn.cluster as cluster
import sys
import seaborn as sns
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.8, 's' : 20, 'linewidths':0}

#variable to store our file name
filepath="P:\python\\bitcoin\\btc.csv"
#read the CSV file into a dataframe
df=pd.read_csv(filepath)

#extract the columns of interest from the dataframe
x=df.priceUSD
y=df.txCount

#put the two 1D arrays into a matrix
data=np.array(list(zip(x,y))).reshape(len(x),2)
#X=np.concatenate(x,y)

#np.savetxt(sys.stdout, X)
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
plt.figure(1)

#routine to generically plot output clusters
def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=12)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=8)


plt.figure(2)
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'complete'})
plt.figure(3)
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'average'})
plt.figure(4)
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
plt.figure(5)
plot_clusters(data, cluster.KMeans, (), {'n_clusters':6})


plt.show()








