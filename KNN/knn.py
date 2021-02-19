#Load the necessary python libraries
import numpy as np
import pandas as pd


#Load the dataset
df = pd.read_csv('diabetes.csv')
#Let's create numpy arrays for features and target
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values

# split data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)


#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=7)

#Fit the model
knn.fit(X_train,y_train)


print("------------------")
#let us get the predictions using the classifier we had fit above
y_pred = knn.predict([[1,89,66,29,0,26.6,0.351,31]])
print("First person is : ",y_pred)
y_pred2 = knn.predict([[2,197,70,45,543,30.5,0.158,53]])
print("Second person is : ",y_pred2)
print("------------------")