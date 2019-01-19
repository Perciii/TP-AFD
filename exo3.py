from math import floor

import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.load_breast_cancer(True)
# XTwoFeatures = [i[:2] for i in X]

lenX = len(X)
lenY = len(y)
tenPercentOfX = floor(lenX / 10)
tenPercentOfY = floor(lenY / 10)
XApp = X[:tenPercentOfX]
yApp = y[:tenPercentOfY]
XTest = X[tenPercentOfX:]
yTest = y[tenPercentOfY:]

print("XApp :", XApp)
print("XTest :", len(XTest))
print("yApp :", len(yApp))
print("yTest :", len(yTest))

plt.plot(XTest, yTest)
plt.show()
