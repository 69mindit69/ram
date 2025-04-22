import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
#print(df)
x=df['sepal length (cm)']
y=df['petal length (cm)']
plt.figure(figsize=(8,6))
plt.scatter(x,y,color='blue',alpha=0.6)
plt.title('Scatter plot')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

corr=np.corrcoef(x,y)[0,1]
print(f"person correlation coefficient is:{corr:.2f}")

cvm=df.cov()
print("\n covarience matrix:")
print(cvm)

crm=df.corr()
print("\n correlation matrix:")
print(crm)

plt.figure(figsize=(8,6))
sns.heatmap(crm,annot=True,cmap='coolwarm',fmt='.2f',cbar=True,linewidths=0.5)
plt.title('correlation matrix Heatmap')
plt.show()
