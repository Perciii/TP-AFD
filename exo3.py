from math import floor

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.datasets import make_moons
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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


def generate_datasets():
    generated_datasets = dict()
    generated_datasets["Blobs"] = make_blobs(n_samples=1000, centers=2, n_features=2)
    generated_datasets["Class"] = make_classification(n_samples=1000, n_classes=2, n_features=2, n_redundant=0)
    generated_datasets["Circles"] = make_circles(n_samples=1000, noise=0.1)
    generated_datasets["Moons"] = make_moons(n_samples=1000, noise=0.1)
    generated_datasets["Breast cancer"] = load_breast_cancer(True)
    return generated_datasets


def create_app_and_test_data(dataset, percentage=1):
    len_dataset = len(dataset[0])
    one_percent_of_data = floor(len_dataset * percentage / 100)
    app_data = [dataset[0][:one_percent_of_data], dataset[1][:one_percent_of_data]]
    test_data = [dataset[0][one_percent_of_data:], dataset[1][one_percent_of_data:]]
    return app_data, test_data


datasets = generate_datasets()
# compare LDA with random forest for blobs appData
blobsApp, blobsTest = create_app_and_test_data(datasets["Blobs"])
# print("blobsApp data :", blobsApp[0], "\nblobsApp target :", blobsApp[1])
classApp, classTest = create_app_and_test_data(datasets["Class"])
circlesApp, circlesTest = create_app_and_test_data(datasets["Circles"])
moonsApp, moonsTest = create_app_and_test_data(datasets["Moons"])
breastApp, breastTest = create_app_and_test_data(datasets["Breast cancer"], 10)


# display_data(circlesTest[0], circlesTest[1], "Moons")

# ==============================================================================
def compare_classifiers():
    lda_classifier = LinearDiscriminantAnalysis()
    random_forest_classifier = RandomForestClassifier()
    svc_classifier = SVC()
    # Blobs
    print("Blobs data :")
    lda_classifier.fit(blobsApp[0], blobsApp[1])
    random_forest_classifier.fit(blobsApp[0], blobsApp[1])
    svc_classifier.fit(blobsApp[0], blobsApp[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(blobsApp[0], blobsApp[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(blobsApp[0], blobsApp[1]))
    print("\t\tSVC score :", svc_classifier.score(blobsApp[0], blobsApp[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(blobsTest[0], blobsTest[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(blobsTest[0], blobsTest[1]))
    print("\t\tSVC score :", svc_classifier.score(blobsTest[0], blobsTest[1]))

    # Class
    print("Class data :")
    lda_classifier.fit(classApp[0], classApp[1])
    random_forest_classifier.fit(classApp[0], classApp[1])
    svc_classifier.fit(classApp[0], classApp[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(classApp[0], classApp[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(classApp[0], classApp[1]))
    print("\t\tSVC score :", svc_classifier.score(classApp[0], classApp[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(classTest[0], classTest[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(classTest[0], classTest[1]))
    print("\t\tSVC score :", svc_classifier.score(classTest[0], classTest[1]))

    # Circles
    print("Circles data :")
    lda_classifier.fit(circlesApp[0], circlesApp[1])
    random_forest_classifier.fit(circlesApp[0], circlesApp[1])
    svc_classifier.fit(circlesApp[0], circlesApp[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(circlesApp[0], circlesApp[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(circlesApp[0], circlesApp[1]))
    print("\t\tSVC score :", svc_classifier.score(circlesApp[0], circlesApp[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(circlesTest[0], circlesTest[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(circlesTest[0], circlesTest[1]))
    print("\t\tSVC score :", svc_classifier.score(circlesTest[0], circlesTest[1]))

    # Moons
    print("Moons data :")
    lda_classifier.fit(moonsApp[0], moonsApp[1])
    random_forest_classifier.fit(moonsApp[0], moonsApp[1])
    svc_classifier.fit(moonsApp[0], moonsApp[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(moonsApp[0], moonsApp[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(moonsApp[0], moonsApp[1]))
    print("\t\tSVC score :", svc_classifier.score(moonsApp[0], moonsApp[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(moonsTest[0], moonsTest[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(moonsTest[0], moonsTest[1]))
    print("\t\tSVC score :", svc_classifier.score(moonsTest[0], moonsTest[1]))

    # Breast cancer
    print("Breast cancer data :")
    lda_classifier.fit(breastApp[0], breastApp[1])
    random_forest_classifier.fit(breastApp[0], breastApp[1])
    svc_classifier.fit(breastApp[0], breastApp[1])
    # App
    print("\tApp data :")
    print("\t\tLDA score :", lda_classifier.score(breastApp[0], breastApp[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(breastApp[0], breastApp[1]))
    print("\t\tSVC score :", svc_classifier.score(breastApp[0], breastApp[1]))

    # Test
    print("\tTest data :")
    print("\t\tLDA score :", lda_classifier.score(breastTest[0], breastTest[1]))
    print("\t\tRandom forest score :", random_forest_classifier.score(breastTest[0], breastTest[1]))
    print("\t\tSVC score :", svc_classifier.score(breastTest[0], breastTest[1]))


compare_classifiers()
