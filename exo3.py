from math import floor

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier

'''
RBF : https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF
Random forest : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
'''


def displayData(X, y, figure=""):
    C0 = []
    C1 = []

    for i in range(len(X)):
        if y[i] == 0:
            C0.append(X[i])
        else:
            C1.append(X[i])

    x0, y0 = zip(*C0)
    plt.figure(figure)
    plt.scatter(x0, y0, c='red')

    x1, y1 = zip(*C1)
    plt.scatter(x1, y1, c='blue')

    plt.title('Blobs')

    plt.show()


def generateDataSets():
    datasets = dict()
    datasets["blobs"] = make_blobs(n_samples=1000, centers=2, n_features=2)
    datasets["class"] = make_classification(n_samples=1000, n_classes=2, n_features=2, n_redundant=0)
    datasets["circles"] = make_circles(n_samples=1000)
    datasets["moons"] = make_moons(n_samples=1000)
    return datasets


def createappandtestdata(dataset):
    lenDataset = len(dataset[0])
    tenPercentOfData = floor(lenDataset / 10)
    appData = [dataset[0][:tenPercentOfData], dataset[1][:tenPercentOfData]]
    testData = [dataset[0][tenPercentOfData:], dataset[1][tenPercentOfData:]]
    return appData, testData


datasets = generateDataSets()
# compare LDA with random forest for blobs appData
blobsApp, blobsTest = createappandtestdata(datasets["blobs"])
# print("blobsApp data :", blobsApp[0], "\nblobsApp target :", blobsApp[1])
classApp, classTest = createappandtestdata(datasets["class"])
circlesApp, circlesTest = createappandtestdata(datasets["circles"])
moonsApp, moonsTest = createappandtestdata(datasets["moons"])

randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(blobsApp[0], blobsApp[1])
print("Random Forest score for blobs :")
print("\tApp :", randomForestClassifier.score(blobsApp[0], blobsApp[1]))
print("\tTest :", randomForestClassifier.score(blobsTest[0], blobsTest[1]))

randomForestClassifier.fit(classApp[0], classApp[1])
print("Random Forest score for class :")
print("\tApp :", randomForestClassifier.score(classApp[0], classApp[1]))
print("\tTest :", randomForestClassifier.score(classTest[0], classTest[1]))

randomForestClassifier.fit(circlesApp[0], circlesApp[1])
print("Random Forest score for circles :")
print("\tApp :", randomForestClassifier.score(circlesApp[0], circlesApp[1]))
print("\tTest :", randomForestClassifier.score(circlesTest[0], circlesTest[1]))

randomForestClassifier.fit(moonsApp[0], moonsApp[1])
print("Random Forest score for moons :")
print("\tApp :", randomForestClassifier.score(moonsApp[0], moonsApp[1]))
print("\tTest :", randomForestClassifier.score(moonsTest[0], moonsTest[1]))

ldaClassifier = LDA()
ldaClassifier.fit(blobsApp[0], blobsApp[1])
print("LDA score for blobs :")
print("\tApp :", ldaClassifier.score(blobsApp[0], blobsApp[1]))
print("\tTest :", ldaClassifier.score(blobsTest[0], blobsTest[1]))

ldaClassifier.fit(classApp[0], classApp[1])
print("LDA score for class :")
print("\tApp :", ldaClassifier.score(classApp[0], classApp[1]))
print("\tTest :", ldaClassifier.score(classTest[0], classTest[1]))

ldaClassifier.fit(circlesApp[0], circlesApp[1])
print("LDA score for circles :")
print("\tApp :", ldaClassifier.score(circlesApp[0], circlesApp[1]))
print("\tTest :", ldaClassifier.score(circlesTest[0], circlesTest[1]))

ldaClassifier.fit(moonsApp[0], moonsApp[1])
print("LDA score for moons :")
print("\tApp :", ldaClassifier.score(moonsApp[0], moonsApp[1]))
print("\tTest :", ldaClassifier.score(moonsTest[0], moonsTest[1]))

# print("the rate of good answer with sklearn on app data")
# print(classifier.score(appData, classApp))

# compare our LDA with sklearn for testData

# classTest = np.concatenate(([0] * len(C0Test), [1] * len(C1Test)))

# print("the rate of good answer with sklearn on test data")
# print(classifier.score(testData, classTest))

# XTwoFeatures = [i[:2] for i in X]
'''
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

displayData(X, y, "Blobs")'''
