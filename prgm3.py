import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
iris=load_iris()
iris_data=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_data.describe())
print(iris_data.tail())

X=iris.data
y=iris.target
plt.figure(figsize=(8,6))
plt.scatter(X[: ,0],X[: ,1],c=y,cmap='viridis')
pca=PCA(n_components=2)

X_pca=pca.fit_transform(X)
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='magma')
plt.show()

