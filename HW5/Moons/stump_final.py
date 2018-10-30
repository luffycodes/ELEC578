import matplotlib
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

moon_data = np.loadtxt("./moons/moons.x.csv", delimiter=',')
moon_data_labels = np.loadtxt("./moons/moons.y.csv", delimiter=',')

train_data, val_data, train_label, val_label = train_test_split(moon_data, moon_data_labels, test_size=0.2)


def plot_train_test_err(training_err, test_err):
    plt.xlabel("no. of  classifiers in Adaboost")
    plt.ylabel("error")
    plt.plot(training_err, '-b', label='training error')
    plt.plot(test_err, '-r', label='test error')
    plt.legend(loc='upper right')
    plt.show()


def plot_decision_boundary(title, classifier, col_1, col_2, label):
    plt.title(title)
    plt.xlabel("first column of moons dataset")
    plt.ylabel("second column of moons dataset")
    plt.scatter(col_1, col_2, color=np.array(['red' if x == -1 else 'blue' for x in label]), s=1)

    if classifier[0] == 0:
        plt.axvline(x=classifier[4])
    else:
        plt.axhline(y=classifier[4])

    plt.show()


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 0.2, x.max() + 0.2
    y_min, y_max = y.min() - 0.2, y.max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_decision_region_adaboost(num_classifier_to_plot):
    xx, yy = make_meshgrid(val_data[:, 0], val_data[:, 1])
    _, Z1 = get_adaboost_prediction(classifiers=ada_classifiers[0:num_classifier_to_plot],
                                    classifier_weights=ada_classifier_weights[0:num_classifier_to_plot],
                                    data=np.c_[xx.ravel(), yy.ravel()], data_labels=[], only_predict=True)
    Z1 = np.array(Z1)
    Z1 = Z1.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.4)
    ax.axis('off')
    colors = ['red', 'blue']
    ax.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap=matplotlib.colors.ListedColormap(colors), s=1)
    ax.set_title('decision boundary when num of classifiers = ' + str(num_classifier_to_plot))
    plt.xlabel("first column of moons dataset", axes=ax)
    plt.ylabel("second column of moons dataset", axes=ax)
    plt.show()


def get_stump_attributes_per_feature(sorted_feature_labels_probs):
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


def get_classifier(prob_dist, *features_with_labels):
    stump_attributes_per_feature = []

    for feature_label in features_with_labels:
        sorted_feature_label_dist = sorted(zip(feature_label[0], feature_label[1], np.copy(prob_dist)))
        stump_attributes_per_feature.append(get_stump_attributes_per_feature(sorted_feature_label_dist))

    arg_min_feature = np.argmin([x[2] for x in stump_attributes_per_feature])
    stump_best_feature = stump_attributes_per_feature[arg_min_feature]
    return arg_min_feature,  stump_best_feature[0], stump_best_feature[1], stump_best_feature[2], stump_best_feature[3]


def get_adaboost_prediction(classifiers, classifier_weights, data, data_labels, only_predict=False):
    err = 0
    label_predict = []
    for j in np.arange(0, data.shape[0], 1):
        pred_j = 0
        for k, w in zip(classifiers, classifier_weights):
            pred_k_j = k[1] if data[j, k[0]] < k[4] else -k[1]
            pred_j += pred_k_j * w
        pred_j = np.sign(pred_j)
        if not only_predict:
            err += int(pred_j != data_labels[j])
        label_predict.append(pred_j)

    if not only_predict:
        err = err/data.shape[0]
    return err, label_predict


def run_adaboost(num_weak_classifiers):
    uniform_dist = np.ones(train_data.shape[0])/train_data.shape[0]
    uniform_dist_classifier = get_classifier(uniform_dist, (feature_1, label_1), (feature_2, label_2))
    classifiers_arr = [uniform_dist_classifier]
    classifier_weights = [0.5 * np.log((1 - uniform_dist_classifier[3]) / uniform_dist_classifier[3])]
    training_err = [get_adaboost_prediction(classifiers_arr, classifier_weights, train_data, train_label)[0]]
    test_err = [get_adaboost_prediction(classifiers_arr, classifier_weights, val_data, val_label)[0]]

    plot_decision_boundary("stump for training data", uniform_dist_classifier, feature_1, feature_2, train_label)
    plot_decision_boundary("stump for test data", uniform_dist_classifier, val_data[:, 0], val_data[:, 1], val_label)

    for _ in np.arange(1, num_weak_classifiers, 1):
        weights_dist = []
        for j in np.arange(0, train_data.shape[0], 1):
            pred_j = 0
            for k, w in zip(classifiers_arr, classifier_weights):
                pred_k_j = k[1] if train_data[j, k[0]] < k[4] else -k[1]
                pred_j += pred_k_j * w

            pred_j = np.exp(- train_label[j] * pred_j)
            weights_dist.append(pred_j)
        iter_classifier = get_classifier(weights_dist/sum(weights_dist), (feature_1, label_1), (feature_2, label_2))
        classifiers_arr.append(iter_classifier)
        classifier_weights.append(0.5 * np.log((1 - iter_classifier[3]) / iter_classifier[3]))
        training_err.append(get_adaboost_prediction(classifiers_arr, classifier_weights, train_data, train_label)[0])
        test_err.append(get_adaboost_prediction(classifiers_arr, classifier_weights, val_data, val_label)[0])

    plot_train_test_err(training_err, test_err)

    return classifiers_arr, classifier_weights


feature_1 = np.copy(train_data[:, 0])
label_1 = np.copy(train_label)

feature_2 = np.copy(train_data[:, 1])
label_2 = np.copy(train_label)

param_weak_classifiers = 100
ada_classifiers, ada_classifier_weights = run_adaboost(param_weak_classifiers)

plot_decision_region_adaboost(param_weak_classifiers)
plot_decision_region_adaboost(5)
