import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

moon_data = np.loadtxt("./moons/moons.x.csv", delimiter=',')
moon_data_labels = np.loadtxt("./moons/moons.y.csv", delimiter=',')

train_data, val_data, train_label, val_label = train_test_split(moon_data, moon_data_labels, test_size=0.2)

feature_1 = np.copy(train_data[:, 0])
label_1 = np.copy(train_label)

feature_2 = np.copy(train_data[:, 1])
label_2 = np.copy(train_label)

color = np.array(['red' if x == -1 else 'green' for x in train_label])


def get_stump_err_per_feature(sorted_feature_labels_weights):
    # Initialization left and right error with stump after zeroth element
    err_l = 0
    if sorted_feature_labels_weights[0][1] == 1:
        err_l = sorted_feature_labels_weights[0][2]

    err_r = 0
    for i in np.arange(1, 800, 1):
        if sorted_feature_labels_weights[i][1] == -1:
            err_r += sorted_feature_labels_weights[i][2]

    # Moving the stump, updating left and right errors
    stump_loc_greater_1 = 1
    stump_val_greater_1 = sorted_feature_labels_weights[0][0]
    stump_err_greater_1 = err_l + err_r

    stump_loc_lesser_1 = 1
    stump_val_lesser_1 = sorted_feature_labels_weights[0][0]
    stump_err_lesser_1 = 1 - (err_l + err_r)

    for i in np.arange(2, 799, 1):
        if sorted_feature_labels_weights[i][1] == -1:
            err_r -= sorted_feature_labels_weights[i][2]
        else:
            err_l += sorted_feature_labels_weights[i][2]

        if err_l + err_r < stump_err_greater_1:
            stump_loc_greater_1 = i
            stump_err_greater_1 = err_l + err_r
            stump_val_greater_1 = sorted_feature_labels_weights[i][0]

        if 1 - (err_r + err_l) < stump_err_lesser_1:
            stump_loc_lesser_1 = i
            stump_err_lesser_1 = 1 - (err_l + err_r)
            stump_val_lesser_1 = sorted_feature_labels_weights[i][0]

    if stump_err_lesser_1 < stump_err_greater_1:
        return 1, stump_loc_lesser_1, stump_err_lesser_1, stump_val_lesser_1
    else:
        return -1, stump_loc_greater_1, stump_err_greater_1, stump_val_greater_1


def get_classifier(weights_dist):
    weights_dist_1 = np.copy(weights_dist)
    weights_dist_2 = np.copy(weights_dist)

    sorted_feature_1_with_labels = sorted(zip(feature_1, label_1, weights_dist_1))
    sorted_feature_2_with_labels = sorted(zip(feature_2, label_2, weights_dist_2))

    stump_pred_1, stump_loc_1, stump_err_1, stump_val_1 = get_stump_err_per_feature(sorted_feature_1_with_labels)
    stump_pred_2, stump_loc_2, stump_err_2, stump_val_2 = get_stump_err_per_feature(sorted_feature_2_with_labels)

    if stump_err_1 < stump_err_2:
        return 1, stump_pred_1, stump_loc_1, stump_err_1, stump_val_1, stump_err_1, 1 - stump_err_1
    else:
        return 2, stump_pred_2, stump_loc_2, stump_err_2, stump_val_2, stump_err_2, 1 - stump_err_2


def get_error(classifiers, classifier_weights, train=True):
    err = 0
    for j in np.arange(0, 800, 1):
        pred_j = 0
        for k, w in zip(classifiers, classifier_weights):
            if k[1] == 1:
                pred_k_j = 1 if train_data[j, k[0] - 1] < k[4] else -1
            else:
                pred_k_j = -1 if train_data[j, k[0] - 1] < k[4] else 1
            pred_j += pred_k_j * w
        err += int(np.sign(pred_j) != train_label[j])
    err = err/800
    return err


debug = False


def adaboost(num_weak_classifiers):
    weights_dist = np.ones(train_data.shape[0])/train_data.shape[0]
    first_classifier = get_classifier(weights_dist)
    classifiers = [first_classifier]
    classifier_weights = [0.5 * np.log(first_classifier[6] / first_classifier[5])]
    training_err = [get_error(classifiers, classifier_weights)]

    # Plotting training data
    plt.title("Scatter plot and Linear boundary : Training data")
    plt.xlabel("First column of moons.x.csv")
    plt.ylabel("First column of moons.x.csv")
    plt.scatter(feature_1, feature_2, color=color, s=1)

    if first_classifier[0] - 1 == 0:
        plt.axvline(x=first_classifier[4])
    else:
        plt.axhline(y=first_classifier[4])

    plt.show()

    for i in np.arange(1, num_weak_classifiers, 1):
        weights_dist = []
        for j in np.arange(0, 800, 1):
            pred_j = 0
            for k, w in zip(classifiers, classifier_weights):
                if k[1] == 1:
                    pred_k_j = 1 if train_data[j, k[0] - 1] < k[4] else -1
                else:
                    pred_k_j = -1 if train_data[j, k[0] - 1] < k[4] else 1
                pred_j += pred_k_j * w

            pred_j = np.exp(- train_label[j] * pred_j)
            weights_dist.append(pred_j)
        iter_classifier = get_classifier(weights_dist/sum(weights_dist))
        classifiers.append(iter_classifier)
        classifier_weights.append(0.5 * np.log(iter_classifier[6] / iter_classifier[5]))
        training_err.append(get_error(classifiers, classifier_weights))

        if debug:
            # Plotting training data
            plt.title("Scatter plot and Linear boundary : Training data")
            plt.xlabel("First column of moons.x.csv")
            plt.ylabel("First column of moons.x.csv")
            plt.scatter(feature_1, feature_2, color=color, s=1)

            if iter_classifier[0] - 1 == 0:
                plt.axvline(x=iter_classifier[4])
            else:
                plt.axhline(y=iter_classifier[4])

            plt.show()

    plt.plot(training_err)
    plt.show()


adaboost(100)

