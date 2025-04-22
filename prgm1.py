import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
df=pd.read_csv("customers1.csv")
nc='Annual Income (k$)'
print(df)
print(df[nc])
mn=df[nc].mean()
md=df[nc].median()
mode=df[nc].mode()[0]
sd=df[nc].std()
var=df[nc].var()
rv=df[nc].max()-df[nc].min()

print(f"Mean:{mn}")
print(f"Median:{md}")
print(f"Mode:{mode}")
print(f"Standrad deviation:{sd}")
print(f"Varience:{var}")
print(f"Range_value:{rv}")
#histogram
plt.figure(figsize=(10,6))
sns.histplot(df[nc],kde=True,bins=30)
plt.title(f"Histogram of {nc}")
plt.xlabel(nc)
plt.ylabel('frequency')
plt.show()
#Box plot
plt.figure(figsize=(10,6))
sns.boxplot(x=df[nc])
plt.title(f"boxplot of {nc}")
plt.xlabel(nc)
#plt.ylabel('frequency')
plt.show()
#Identify outlier using IQR
Q1=df[nc].quantile(0.25)
Q3=df[nc].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outlier=df[(df[nc]<lower_bound)|(df[nc]>upper_bound)]
print(f"Outlearn based on IQR:\n{outlier}")
#select a categorial variable
cc='Annual Income (k$)'
# frequency of each category
category_counts=df[cc].value_counts()
#Barchart
plt.figure(figsize=(15,8))
category_counts.plot(kind='bar')
plt.title(f"Frequency of category in {cc}")
plt.xlabel('Annual Income (k$)')
plt.ylabel('frequency')
plt.show()
#piechart
plt.figure(figsize=(15,15))
category_counts.plot(kind='pie',autopct='%1.1f%%',startangle=90)
plt.title(f"category distribution in {cc}")
plt.ylabel('')
plt.show()
