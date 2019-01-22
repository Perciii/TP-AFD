from math import floor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# ==============================================================================
# Data functions
# ==============================================================================
def display_data(x, y, figure=""):
    c0 = []
    c1 = []

    for i in range(len(x)):
        if y[i] == 0:
            c0.append(x[i])
        else:
            c1.append(x[i])

    x0, y0 = zip(*c0)
    plt.figure(figure)
    plt.scatter(x0, y0, c='red')

    x1, y1 = zip(*c1)
    plt.scatter(x1, y1, c='blue')

    plt.title(figure)

    plt.show()


def generate_data_sets():
    generated_data_sets = dict()
    generated_data_sets["Blobs"] = make_blobs(n_samples=1000, centers=2, n_features=2)
    generated_data_sets["Class"] = make_classification(n_samples=1000, n_classes=2, n_features=2, n_redundant=0)
    generated_data_sets["Circles"] = make_circles(n_samples=1000, noise=0.1)
    generated_data_sets["Moons"] = make_moons(n_samples=1000, noise=0.1)
    generated_data_sets["Breast cancer"] = load_breast_cancer(True)
    return generated_data_sets


# separate data_set in two data sets with different classes
def split_data_set(data_set):
    data_zero = [[], []]
    data_one = [[], []]
    for i in range(len(data_set[1])):
        if data_set[1][i] == 0:
            data_zero[0].append(data_set[0][i])
            data_zero[1].append(data_set[1][i])
        else:
            data_one[0].append(data_set[0][i])
            data_one[1].append(data_set[1][i])
    return data_zero, data_one


def create_app_and_test_data(dataset, percentage=1):
    len_dataset = len(dataset[0])
    one_percent_of_data = floor(len_dataset * percentage / 100)
    app_data = [dataset[0][:one_percent_of_data], dataset[1][:one_percent_of_data]]
    test_data = [dataset[0][one_percent_of_data:], dataset[1][one_percent_of_data:]]
    return app_data, test_data


# ==============================================================================
# Classifiers functions
# ==============================================================================
def compare_classifiers():
    # Classifiers
    lda_classifier = LinearDiscriminantAnalysis()
    random_forest_classifier = RandomForestClassifier()
    svc_classifier = SVC(gamma='auto')

    # Data
    data_sets = generate_data_sets()

    blobs_app, blobs_test = create_app_and_test_data(data_sets["Blobs"])
    class_app, class_test = create_app_and_test_data(data_sets["Class"])
    circles_app, circles_test = create_app_and_test_data(data_sets["Circles"])
    moons_app, moons_test = create_app_and_test_data(data_sets["Moons"])
    breast_app, breast_test = create_app_and_test_data(data_sets["Breast cancer"], 10)

    # Blobs
    print("Blobs data :")
    lda_classifier.fit(blobs_app[0], blobs_app[1])
    random_forest_classifier.fit(blobs_app[0], blobs_app[1])
    svc_classifier.fit(blobs_app[0], blobs_app[1])

    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(blobs_app[0], blobs_app[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(blobs_app[0], blobs_app[1]))
    print("\t\tSVC score :", svc_classifier.score(blobs_app[0], blobs_app[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(blobs_test[0], blobs_test[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(blobs_test[0], blobs_test[1]))
    print("\t\tSVC score :", svc_classifier.score(blobs_test[0], blobs_test[1]))

    # Class
    print("Class data :")
    lda_classifier.fit(class_app[0], class_app[1])
    random_forest_classifier.fit(class_app[0], class_app[1])
    svc_classifier.fit(class_app[0], class_app[1])

    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(class_app[0], class_app[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(class_app[0], class_app[1]))
    print("\t\tSVC score :", svc_classifier.score(class_app[0], class_app[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(class_test[0], class_test[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(class_test[0], class_test[1]))
    print("\t\tSVC score :", svc_classifier.score(class_test[0], class_test[1]))

    # Circles
    print("Circles data :")
    lda_classifier.fit(circles_app[0], circles_app[1])
    random_forest_classifier.fit(circles_app[0], circles_app[1])
    svc_classifier.fit(circles_app[0], circles_app[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(circles_app[0], circles_app[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(circles_app[0], circles_app[1]))
    print("\t\tSVC score :", svc_classifier.score(circles_app[0], circles_app[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(circles_test[0], circles_test[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(circles_test[0], circles_test[1]))
    print("\t\tSVC score :", svc_classifier.score(circles_test[0], circles_test[1]))

    # Moons
    print("Moons data :")
    lda_classifier.fit(moons_app[0], moons_app[1])
    random_forest_classifier.fit(moons_app[0], moons_app[1])
    svc_classifier.fit(moons_app[0], moons_app[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(moons_app[0], moons_app[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(moons_app[0], moons_app[1]))
    print("\t\tSVC scoEre :", svc_classifier.score(moons_app[0], moons_app[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(moons_test[0], moons_test[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(moons_test[0], moons_test[1]))
    print("\t\tSVC score :", svc_classifier.score(moons_test[0], moons_test[1]))

    # Breast cancer
    print("Breast cancer data :")
    lda_classifier.fit(breast_app[0], breast_app[1])
    random_forest_classifier.fit(breast_app[0], breast_app[1])
    svc_classifier.fit(breast_app[0], breast_app[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(breast_app[0], breast_app[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(breast_app[0], breast_app[1]))
    print("\t\tSVC score :", svc_classifier.score(breast_app[0], breast_app[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(breast_test[0], breast_test[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(breast_test[0], breast_test[1]))
    print("\t\tSVC score :", svc_classifier.score(breast_test[0], breast_test[1]))

    # Return can seem strange but we will need this breast_app in the main
    return breast_app


# Cross validation for SVC, data_set is an array with data in data_set[0] and classes in data_set[1]
# k must be superior or equal to len(data_set[0])
def cross_validation(data_set, k, classifier=""):
    scores = []
    data_zero, data_one = split_data_set(data_set)

    # number of each class in each block
    gap_zero = len(data_zero[0]) // k
    gap_one = len(data_one[0]) // k

    for i in range(k):

        # create new test_data with one block
        test_data = (data_zero[0][i * gap_zero:(i * gap_zero) + gap_zero]) + (
            data_one[0][i * gap_one:(i * gap_one) + gap_one])
        test_class = data_zero[1][i * gap_zero:(i * gap_zero) + gap_zero] + data_one[1][
                                                                            i * gap_one:(i * gap_one) + gap_one]

        # create new app_data
        app_data = data_zero[0][:i * gap_zero] + data_zero[0][i * gap_zero + gap_zero:] + data_one[0][:i * gap_one] + \
            data_one[0][i * gap_one + gap_one:]
        app_class = data_zero[1][:i * gap_zero] + data_zero[1][i * gap_zero + gap_zero:] + data_one[1][:i * gap_one] + \
            data_one[1][i * gap_one + gap_one:]

        if classifier == "SVC":
            svc_classifier = SVC(gamma='auto')
            svc_classifier.fit(app_data, app_class)
            scores.append(svc_classifier.score(test_data, test_class))
        if classifier == "LDA":
            lda_classifier = LinearDiscriminantAnalysis()
            lda_classifier.fit(app_data, app_class)
            scores.append(lda_classifier.score(test_data, test_class))
        if classifier == "RFC":
            random_forest_classifier = RandomForestClassifier()
            random_forest_classifier.fit(app_data, app_class)
            scores.append(random_forest_classifier.score(test_data, test_class))

    return np.mean(scores)


# ==============================================================================
# Main
# ==============================================================================
def main():
    breast_app = compare_classifiers()

    print("cross Validation")
    print("for SVC:")
    print(cross_validation(breast_app, 5, "SVC"))
    print("for LDA:")
    print(cross_validation(breast_app, 5, "LDA"))
    print("for RFC:")
    print(cross_validation(breast_app, 5, "RFC"))


main()
