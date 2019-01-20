#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# Display data for question 1, take two data matrix (optional a name for the figure)
def display_data(c0, c1, figure=""):
    x0, y0 = zip(*c0)
    plt.figure(figure)
    plt.scatter(x0, y0, c='red')

    x1, y1 = zip(*c1)
    plt.scatter(x1, y1, c='blue')

    plt.title('Nuage de points')

    plt.show()


# this function takes two apprenticeship data matrixes and returns two parameters of an LDA
# (optional lamb between 0 and 1 to generalize, question 5)
def my_lda(c0, c1, lamb=None):
    n0 = len(c0)
    n1 = len(c1)

    if lamb and 0 <= lamb <= 1:
        x0, y0 = zip(*c0)
        sigma0 = np.cov(x0, y0)

        x1, y1 = zip(*c1)
        sigma1 = np.cov(x1, y1)

        sigma_hat = lamb * np.add(np.dot(n0 / (n0 + n1), sigma0), np.dot(n1 / (n0 + n1), sigma1)) + (
                1 - lamb) * np.identity(2)
    else:
        xApp, yApp = zip(*np.concatenate((c0, c1)))
        sigma_hat = np.cov(xApp, yApp)

    mu0_hat = np.mean(c0, axis=0)
    mu1_hat = np.mean(c1, axis=0)

    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)

    # mu0T = np.transpose(mu0_hat)
    # mu1T = np.transpose(mu1_hat)
    sigma_inv = np.linalg.inv(sigma_hat)

    w = np.dot(sigma_inv, np.subtract(mu0_hat, mu1_hat))
    b = np.dot(-1 / 2, np.dot(np.transpose(np.subtract(mu0_hat, mu1_hat)),
                              np.dot(sigma_inv, np.add(mu0_hat, mu1_hat)))) + np.log(
        pi0 / pi1)

    return w, b


# apply a LDA from w and b params and return the class of x

def apply_lda(w, b, x):
    x_t = np.transpose(x)
    decision = np.add(np.dot(x_t, w), b)

    if decision > 0:
        return 0
    else:
        return 1


# return the accuracy of good prediction (option lamb for question 5)
def test_my_lda(c0_app, c1_app, c0_test, c1_test, lamb=None):
    w, b = my_lda(c0_app, c1_app, lamb)
    good_prediction = 0
    for i in c0_test:
        if apply_lda(w, b, i) == 0:
            good_prediction += 1
    for i in c1_test:
        if apply_lda(w, b, i) == 1:
            good_prediction += 1

    return good_prediction / (len(c0_test) + len(c1_test))


# this function test myLDA with various lambda and return a an array of good success rate and an array of lambda
def test_my_lda_lamb(c0_app, c1_app, c0_test, c1_test):
    result = []
    lambs = []
    for i in range(11):
        lamb = i / 10
        lambs.append(lamb)
        result.append(test_my_lda(c0_app, c1_app, c0_test, c1_test, lamb))
    return lambs, result


# generate  two arrays of x data from two covariance and two mean
def generate_data(x, sigma0, sigma1, mu0, mu1):
    c0_app = np.random.multivariate_normal(mu0, sigma0, x)
    c1_app = np.random.multivariate_normal(mu1, sigma1, x)
    return c0_app, c1_app


# for question 4 test myLda with different variance matrix
def test_my_lda_with_varying_cov():
    result = []
    variances = []
    sigma = np.array([[1, 1 / 2], [1 / 2, 1]])
    mu0 = np.array([0, 0])
    mu1 = np.array([3, 2])

    c0_app = np.random.multivariate_normal(mu0, sigma, 10)
    c0_test = np.random.multivariate_normal(mu0, sigma, 1000)

    for i in range(100):
        variances.append(i / 10)
        sigma2 = np.array([[1 + i / 10, 1 / 2], [1 / 2, 1 + i / 10]])

        c1_app_bis = np.random.multivariate_normal(mu1, sigma2, 10)
        c1_test = np.random.multivariate_normal(mu1, sigma2, 1000)
        result.append(test_my_lda(c0_app, c1_app_bis, c0_test, c1_test))

    return result, variances


# take two data list of each class
def cross_validation(c0_app, c1_app):
    lambs = np.arange(0, 1.1, 0.1)

    good_prediction = [0] * (len(lambs))

    for i in range(len(lambs)):
        lamb = lambs[i]
        for j in range(len(c0_app)):
            c0_app_bis = list(c0_app)
            del c0_app_bis[j]
            w, b = my_lda(c0_app_bis, c1_app, lamb)
            if apply_lda(w, b, c0_app[j]) == 0:
                good_prediction[i] += 1

        for k in range(len(c1_app)):
            c1_app_bis = list(c1_app)
            del c1_app_bis[k]
            w, b = my_lda(c0_app, c1_app_bis, lamb)
            if apply_lda(w, b, c1_app[k]) == 1:
                good_prediction[i] += 1
        good_prediction[i] = good_prediction[i] / (len(c0_app) + len(c1_app))

    # search the max
    for m in range(len(good_prediction)):
        if max(good_prediction) == good_prediction[m]:
            return lambs[m]


#########################

#         main          #

#########################


# EXERCICE 1:

# initialize data

mu0 = np.array([0, 0])
mu1 = np.array([3, 2])
sigma = np.array([[1, 1 / 2], [1 / 2, 1]])

# generate data for apprenticeship
np.random.seed(1)
C0App = np.random.multivariate_normal(mu0, sigma, 10)
C1App = np.random.multivariate_normal(mu1, sigma, 10)

appData = np.concatenate((C0App, C1App))

# generate data for test
C0Test = np.random.multivariate_normal(mu0, sigma, 1000)
C1Test = np.random.multivariate_normal(mu1, sigma, 1000)

testData = np.concatenate((C0Test, C1Test))

# display appData
display_data(C0App, C1App, "App data")

# display testData
display_data(C0Test, C1Test, "Test data")

# EXERCICE 2:

# Question 1:

# print the good classification rate for apprenticeship data
print("The rate of good classification by my LDA on apprenticeship data is :")
print(test_my_lda(C0App, C1App, C0App, C1App))

# print the good classification rate for test data
print("The rate of good classification by my LDA on test data is :")
print(test_my_lda(C0App, C1App, C0Test, C1Test))

# compare our LDA with sklearn for appData
classApp = np.concatenate(([0] * len(C0App), [1] * len(C1App)))

classifier = LDA()
classifier.fit(appData, classApp)

print("The rate of good classification by sklearn on app data is :")
print(classifier.score(appData, classApp))

# compare our LDA with sklearn for testData

classTest = np.concatenate(([0] * len(C0Test), [1] * len(C1Test)))

print("The rate of good classification by sklearn on test data :")
print(classifier.score(testData, classTest))

# Question 2:

# test MyLDA with a new C0App (replace first value by [-10,-10])

C0AppBis = np.copy(C0App)
C0AppBis[0] = np.array([-10, -10])

print("The rate of good classification of data App (C0App modified) by myLDA:")
print(test_my_lda(C0AppBis, C1App, C0App, C1App))

appDataBis = np.concatenate((C0AppBis, C1App))
print("The rate of good classification of data Test (C0App modified) by myLDA:")
print(test_my_lda(C0AppBis, C1App, C0Test, C1Test))

# test sklearn with a new C0App

classifierBis = LDA()
classifierBis.fit(appDataBis, classApp)
print("The rate of good classification of data App (C0App modified) by sklearn:")
print(classifierBis.score(appDataBis, classApp))
print("The rate of good classification of data Test (C0App modified) by sklearn:")
print(classifierBis.score(testData, classTest))

# Question 3:
w, b = my_lda(C0App, C1App)

# computed obvious point to draw boundary of decision on app data and test data graphs
easyPointsX = np.transpose([0, -b / w[1]])
easyPointsY = np.transpose([-b / w[0], 0])
plt.figure("app data")
plt.plot(easyPointsX, easyPointsY, 'y-')
plt.figure("test data")
plt.plot(easyPointsX, easyPointsY, 'y-')
plt.show()

# Question 4:

# print(test_my_lda_with_varying_cov())
plt.figure("good classification by variance")
x, y = test_my_lda_with_varying_cov()
plt.plot(y, x)

plt.show()

# Question 5:

# test my generalized LDA
plt.figure("LDA with lambda app")
x, y = test_my_lda_lamb(C0App, C1App, C0App, C1App)
plt.plot(x, y)
plt.show()

# Question 6:
plt.figure("LDA with lambda test")
x, y = test_my_lda_lamb(C0App, C1App, C0Test, C1Test)
plt.plot(x, y)
plt.show()

print(cross_validation(C0App, C1App))
