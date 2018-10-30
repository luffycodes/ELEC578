import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

moon_data = np.loadtxt("./moons/moons.x.csv", delimiter=',')
moon_data_labels = np.loadtxt("./moons/moons.y.csv", delimiter=',')

train_data, val_data, train_label, val_label = train_test_split(moon_data, moon_data_labels, test_size=0.2)


def plot_decision_boundary(title, classifier, col_1, col_2, label):
    plt.title(title)
    plt.xlabel("first column of moons dataset")
    plt.ylabel("second column of moons dataset")
    plt.scatter(col_1, col_2, color=np.array(['red' if x == -1 else 'green' for x in label]), s=1)

    if classifier[0] - 1 == 0:
        plt.axvline(x=classifier[4])
    else:
        plt.axhline(y=classifier[4])

    plt.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 0.2, x.max() + 0.2
    y_min, y_max = y.min() - 0.2, y.max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_boundary_adaboost(num_classifier_to_plot):
    xx, yy = make_meshgrid(val_data[:, 0], val_data[:, 1])
    _, Z1 = get_prediction_and_err(classifiers=ada_classifiers[0:num_classifier_to_plot],
                                   classifier_weights=ada_classifier_weights[0:num_classifier_to_plot],
                                   data=np.c_[xx.ravel(), yy.ravel()], data_labels=[], only_predict=True)
    Z1 = np.array(Z1)
    Z1 = Z1.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.4)
    ax.axis('off')
    colors = ['red', 'green']
    ax.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap=matplotlib.colors.ListedColormap(colors))
    ax.set_title('decision boundary when num of classifiers = ' + str(num_classifier_to_plot))
    plt.xlabel("first column of moons dataset", axes=ax)
    plt.ylabel("second column of moons dataset", axes=ax)
    plt.show()


def get_stump_err_per_feature(sorted_feature_labels_probs):
    err_l = sorted_feature_labels_probs[0][2] if sorted_feature_labels_probs[0][1] == 1 else 0
    err_r = 0
    for i in np.arange(1, 800, 1):
        if sorted_feature_labels_probs[i][1] == -1:
            err_r += sorted_feature_labels_probs[i][2]

    err_arr = [err_l + err_r]
    for i in np.arange(1, 799, 1):
        if sorted_feature_labels_probs[i][1] == -1:
            err_r -= sorted_feature_labels_probs[i][2]
        else:
            err_l += sorted_feature_labels_probs[i][2]
        err_arr.append(err_l + err_r)

    one_minus_err = 1 - np.array(err_arr)
    if np.min(one_minus_err) < np.min(err_arr):
        argmin_one_minus_err = np.argmin(one_minus_err)
        return 1, argmin_one_minus_err + 1, np.min(one_minus_err), sorted_feature_labels_probs[argmin_one_minus_err + 1][0]
    else:
        return -1, np.argmin(err_arr) + 1, np.min(err_arr), sorted_feature_labels_probs[np.argmin(err_arr) + 1][0]


def get_classifier(prob_dist):
    sorted_feature_1_with_labels = sorted(zip(feature_1, label_1, np.copy(prob_dist)))
    sorted_feature_2_with_labels = sorted(zip(feature_2, label_2, np.copy(prob_dist)))

    stump_pred_1, stump_loc_1, stump_err_1, stump_val_1 = get_stump_err_per_feature(sorted_feature_1_with_labels)
    stump_pred_2, stump_loc_2, stump_err_2, stump_val_2 = get_stump_err_per_feature(sorted_feature_2_with_labels)

    if stump_err_1 < stump_err_2:
        return 1, stump_pred_1, stump_loc_1, stump_err_1, stump_val_1, stump_err_1, 1 - stump_err_1
    else:
        return 2, stump_pred_2, stump_loc_2, stump_err_2, stump_val_2, stump_err_2, 1 - stump_err_2


def get_prediction_and_err(classifiers, classifier_weights, data, data_labels, only_predict=False):
    err = 0
    label_predict = []
    for j in np.arange(0, data.shape[0], 1):
        pred_j = 0
        for k, w in zip(classifiers, classifier_weights):
            pred_k_j = k[1] if data[j, k[0] - 1] < k[4] else -k[1]
            pred_j += pred_k_j * w
        pred_j = np.sign(pred_j)
        if not only_predict:
            err += int(pred_j != data_labels[j])
        label_predict.append(pred_j)

    if not only_predict:
        err = err/data.shape[0]
    return err, label_predict


def adaboost(num_weak_classifiers):
    weights_dist = np.ones(train_data.shape[0])/train_data.shape[0]
    first_classifier = get_classifier(weights_dist)
    classifiers = [first_classifier]
    classifier_weights = [0.5 * np.log(first_classifier[6] / first_classifier[5])]
    training_err = [get_prediction_and_err(classifiers, classifier_weights, train_data, train_label)[0]]
    test_err = [get_prediction_and_err(classifiers, classifier_weights, val_data, val_label)[0]]

    plot_decision_boundary("linear boundary for training data", first_classifier, feature_1, feature_2, train_label)
    plot_decision_boundary("linear boundary for test data", first_classifier, val_data[:, 0], val_data[:, 1], val_label)

    for i in np.arange(1, num_weak_classifiers, 1):
        weights_dist = []
        for j in np.arange(0, 800, 1):
            pred_j = 0
            for k, w in zip(classifiers, classifier_weights):
                pred_k_j = k[1] if train_data[j, k[0] - 1] < k[4] else -k[1]
                pred_j += pred_k_j * w

            pred_j = np.exp(- train_label[j] * pred_j)
            weights_dist.append(pred_j)
        iter_classifier = get_classifier(weights_dist/sum(weights_dist))
        classifiers.append(iter_classifier)
        classifier_weights.append(0.5 * np.log(iter_classifier[6] / iter_classifier[5]))
        training_err.append(get_prediction_and_err(classifiers, classifier_weights, train_data, train_label)[0])
        test_err.append(get_prediction_and_err(classifiers, classifier_weights, val_data, val_label)[0])

    plt.xlabel("no. of  classifiers in Adaboost")
    plt.ylabel("error")
    plt.plot(training_err, '-b', label='training error')
    plt.plot(test_err, '-r', label='test error')
    plt.legend(loc='upper right')
    plt.show()

    return classifiers, classifier_weights


feature_1 = np.copy(train_data[:, 0])
label_1 = np.copy(train_label)

feature_2 = np.copy(train_data[:, 1])
label_2 = np.copy(train_label)

ada_classifiers, ada_classifier_weights = adaboost(100)

plot_decision_boundary_adaboost(100)
plot_decision_boundary_adaboost(5)
