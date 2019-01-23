#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# ==============================================================================
# Data functions
# ==============================================================================

# Generate data
def generate_app_and_test_data():
    data = dict()
    mu0 = np.array([0, 0])
    mu1 = np.array([3, 2])
    sigma = np.array([[1, 1 / 2], [1 / 2, 1]])

    # generate data for apprenticeship
    np.random.seed(1)
    c0_app = np.random.multivariate_normal(mu0, sigma, 10)
    c1_app = np.random.multivariate_normal(mu1, sigma, 10)
    data["app"] = [c0_app, c1_app]

    # generate data for test
    c0_test = np.random.multivariate_normal(mu0, sigma, 1000)
    c1_test = np.random.multivariate_normal(mu1, sigma, 1000)
    data["test"] = [c0_test, c1_test]

    return data


# Display data for question 1, take two data matrix (optional a name for the figure)
def display_data(c0, c1, figure=""):
    x0, y0 = zip(*c0)
    plt.figure(figure)
    plt.scatter(x0, y0, c='red')

    x1, y1 = zip(*c1)
    plt.scatter(x1, y1, c='blue')

    plt.title(figure)

    plt.show()


# ==============================================================================
# LDA functions
# ==============================================================================

# this function takes two apprenticeship data matrices and returns two parameters of an LDA
# (optional lamb between 0 and 1 to generalize, question 5)
def my_lda(c0, c1, lamb=None):
    n0 = len(c0)
    n1 = len(c1)

    if lamb is not None and 0 <= lamb <= 1:
        x0, y0 = zip(*c0)
        sigma0 = np.cov(x0, y0)

        x1, y1 = zip(*c1)
        sigma1 = np.cov(x1, y1)

        sigma_hat = lamb * np.add(np.dot(n0 / (n0 + n1), sigma0), np.dot(n1 / (n0 + n1), sigma1)) + (
                1 - lamb) * np.identity(2)
    else:
        x_app, y_app = zip(*np.concatenate((c0, c1)))
        sigma_hat = np.cov(x_app, y_app)

    mu0_hat = np.mean(c0, axis=0)
    mu1_hat = np.mean(c1, axis=0)
    pi0 = n0 / (n0 + n1)
    pi1 = n1 / (n0 + n1)
    sigma_inv = np.linalg.inv(sigma_hat)

    w = np.dot(sigma_inv, np.subtract(mu0_hat, mu1_hat))
    b = np.dot(-1 / 2, np.dot(np.transpose(np.subtract(mu0_hat, mu1_hat)),
                              np.dot(sigma_inv, np.add(mu0_hat, mu1_hat)))) + np.log(
        pi0 / pi1)

    return w, b


# applies a LDA from w and b parameters and returns the class of x
def apply_lda(w, b, x):
    x_t = np.transpose(x)
    decision = np.add(np.dot(x_t, w), b)

    if decision > 0:
        return 0
    else:
        return 1


# returns the accuracy of good prediction (optional lamb for question 5)
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


# this function test my_lda with various lambdas and returns an array of good classification rate and an array of lambdas
def test_my_lda_lamb(c0_app, c1_app, c0_test, c1_test):
    result = []
    lambs = []
    for i in range(11):
        lamb = i / 10
        lambs.append(lamb)
        result.append(test_my_lda(c0_app, c1_app, c0_test, c1_test, lamb))
    return lambs, result


# for question 4 test my_lda with different variance matrices
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


# takes two data list of each class
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


# ==============================================================================
# Main
# ==============================================================================
def main():
    # EXERCICE 1:

    # generate data
    data = generate_app_and_test_data()
    c0_app = data["app"][0]
    c1_app = data["app"][1]
    c0_test = data["test"][0]
    c1_test = data["test"][1]

    app_data = np.concatenate((c0_app, c1_app))
    test_data = np.concatenate((c0_test, c1_test))

    # display app_data
    display_data(c0_app, c1_app, "App data")

    # display test_data
    display_data(c0_test, c1_test, "Test data")

    # EXERCICE 2:
    # Question 1:
    # print the good classification rate for apprenticeship data
    print("The rate of good classification by my LDA on apprenticeship data is :")
    print(test_my_lda(c0_app, c1_app, c0_app, c1_app))

    # print the good classification rate for test data
    print("The rate of good classification by my LDA on test data is :")
    print(test_my_lda(c0_app, c1_app, c0_test, c1_test))

    # compare our LDA with sklearn for app_data
    class_app = np.concatenate(([0] * len(c0_app), [1] * len(c1_app)))

    classifier = LDA()
    classifier.fit(app_data, class_app)

    print("The rate of good classification by sklearn on app data is :")
    print(classifier.score(app_data, class_app))

    # compare our LDA with sklearn for test_data

    class_test = np.concatenate(([0] * len(c0_test), [1] * len(c1_test)))

    print("The rate of good classification by sklearn on test data :")
    print(classifier.score(test_data, class_test))

    # Question 2:
    # test MyLDA with a new c0_app (replace first value by [-10,-10])
    c0_app_bis = np.copy(c0_app)
    c0_app_bis[0] = np.array([-10, -10])

    print("The rate of good classification of data App (c0_app modified) by myLDA:")
    print(test_my_lda(c0_app_bis, c1_app, c0_app, c1_app))

    app_data_bis = np.concatenate((c0_app_bis, c1_app))
    print("The rate of good classification of data Test (c0_app modified) by myLDA:")
    print(test_my_lda(c0_app_bis, c1_app, c0_test, c1_test))

    # test sklearn with a new c0_app
    classifier_bis = LDA()
    classifier_bis.fit(app_data_bis, class_app)
    print("The rate of good classification of data App (c0_app modified) by sklearn:")
    print(classifier_bis.score(app_data_bis, class_app))
    print("The rate of good classification of data Test (c0_app modified) by sklearn:")
    print(classifier_bis.score(test_data, class_test))

    # Question 3:
    w, b = my_lda(c0_app, c1_app)

    # computed obvious point to draw boundary of decision on app data and test data graphs
    easy_points_x = np.transpose([0, -b / w[1]])
    easy_points_y = np.transpose([-b / w[0], 0])
    plt.figure("App data")
    plt.plot(easy_points_x, easy_points_y, 'y-')
    plt.figure("Test data")
    plt.plot(easy_points_x, easy_points_y, 'y-')
    
    w_bis,b_bis=my_lda(c0_app_bis,c1_app)
    easy_points_x2 = np.transpose([0, -b_bis / w_bis[1]])
    easy_points_y2 = np.transpose([-b_bis / w_bis[0], 0])
    plt.figure("App data")
    plt.plot(easy_points_x2, easy_points_y2, 'g-')
    plt.figure("Test data")
    plt.plot(easy_points_x2, easy_points_y2, 'g-')
    
    plt.show()


    # Question 4:
    # print(test_my_lda_with_varying_cov())
    plt.figure("good classification by variance")
    plt.title("good classification by variance")
    x, y = test_my_lda_with_varying_cov()
    plt.plot(y, x)

    plt.show()

    # Question 5:
    # test my generalized LDA
    plt.figure("LDA with lambda app")
    plt.title("LDA with lambda app")
    x, y = test_my_lda_lamb(c0_app, c1_app, c0_app, c1_app)
    plt.plot(x, y)
    plt.show()

    # Question 6:
    plt.figure("LDA with lambda test")
    plt.title("LDA with lambda test")
    x, y = test_my_lda_lamb(c0_app, c1_app, c0_test, c1_test)
    plt.plot(x, y)
    plt.show()

    print(cross_validation(c0_app, c1_app))


main()
