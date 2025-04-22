import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier

iris=pd.read_csv('iris1.csv')
print(iris.head())
print(iris.columns)
x=iris.drop('class',axis=1).values
y=iris['class'].values
print(x[:5])
print(y[:5])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

def knn_model(x_train,x_test,y_train,y_test,k,weighted=False):
	if weighted:
		model=KNeighborsClassifier(n_neighbors=k,weights='distance')
	else:
		model=KNeighborsClassifier(n_neighbors=k,weights='uniform')
		model.fit(x_train,y_train)
		y_pred=model.predict(x_test)
		accuracy=accuracy_score(y_test,y_pred)
		f1=f1_score(y_test,y_pred,average='weighted')
	return accuracy,f1
	k_values=[1,3,5]
	results={'k':[],'Regular k-NN Accuracy':[],'Regular k-NN F1 Score':[],'Weighted k-NN Accuracy':[],'Weighted k-NN F1 Score':[]}
	for k in k_values:
		reg_accuracy,reg_f1=knn_model(x_train,x_test,y_train,y_test,k,weighted=False)
		weighted_accuracy,weighted_f1=knn_model(x_train,x_test,y_train,y_test,k,weighted=True)
		results['k'].append(k)
		results['Regular k-NN Accuracy'].append(reg_accuracy)
		results['Regular k-NN F1 Score'].append(reg_f1)
		results['Weighted k-NN Accuracy'].append(weighted_accuracy)
		results['Weighted k-NN F1 Score'].append(weighted_f1)
	results_df=pd.DataFrame(results)
	print(results_df)

	
		
 
