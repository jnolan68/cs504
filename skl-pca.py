#a class demonstrating Principal Component Analysis in scikit learn
#@author Jim Nolan

import time
#matrix math
import numpy as np
#graphing
import matplotlib.pyplot as plt
#dataframe and analysis
import pandas as pd
#Machine learning library scikit learn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



#test with Iris Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

#print(df)

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
#print(x)
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

#reduce the 4 dimensions down to 2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
print(pca.explained_variance_ratio_)
principalDf = pd.DataFrame(data = principalComponents,
                columns = ['principal component 1', 'principal component 2'])

#Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

#visualize the 2D projection
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
