
"""
    x1 x2  y 
    0  0   0
    0  1   0
    1  0   0
    1  1   1
"""
from sklearn.linear_model import Perceptron

X = [ [0,0], [1,1] ]

y = [0,1]

clf = Perceptron()

clf.fit(X,y)

pred = clf.predict([[1,0]])

print(pred)

print('Accuracy : ',clf.score(X,y))
