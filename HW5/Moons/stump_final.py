import matplotlib
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


def get_stump_err_per_feature(sorted_feature_labels_probs):
    err_l = sorted_feature_labels_probs[0][2] if sorted_feature_labels_probs[0][1] == 1 else 0
    err_r = 0
    for i in np.arange(1, 800, 1):
        if sorted_feature_labels_probs[i][1] == -1:
            err_r += sorted_feature_labels_probs[i][2]

    stump_loc_greater_1 = 1
    stump_val_greater_1 = sorted_feature_labels_probs[0][0]
    stump_err_greater_1 = err_l + err_r

    stump_loc_lesser_1 = 1
    stump_val_lesser_1 = sorted_feature_labels_probs[0][0]
    stump_err_lesser_1 = 1 - (err_l + err_r)

    for i in np.arange(2, 799, 1):
        if sorted_feature_labels_probs[i][1] == -1:
            err_r -= sorted_feature_labels_probs[i][2]
        else:
            err_l += sorted_feature_labels_probs[i][2]

        if err_l + err_r < stump_err_greater_1:
            stump_loc_greater_1 = i
            stump_err_greater_1 = err_l + err_r
            stump_val_greater_1 = sorted_feature_labels_probs[i][0]

        if 1 - (err_r + err_l) < stump_err_lesser_1:
            stump_loc_lesser_1 = i
            stump_err_lesser_1 = 1 - (err_l + err_r)
            stump_val_lesser_1 = sorted_feature_labels_probs[i][0]

    if stump_err_lesser_1 < stump_err_greater_1:
        return 1, stump_loc_lesser_1, stump_err_lesser_1, stump_val_lesser_1
    else:
        return -1, stump_loc_greater_1, stump_err_greater_1, stump_val_greater_1


def get_classifier(prob_dist):
    sorted_feature_1_with_labels = sorted(zip(feature_1, label_1, np.copy(prob_dist)))
    sorted_feature_2_with_labels = sorted(zip(feature_2, label_2, np.copy(prob_dist)))

    stump_pred_1, stump_loc_1, stump_err_1, stump_val_1 = get_stump_err_per_feature(sorted_feature_1_with_labels)
    stump_pred_2, stump_loc_2, stump_err_2, stump_val_2 = get_stump_err_per_feature(sorted_feature_2_with_labels)

    if stump_err_1 < stump_err_2:
        return 1, stump_pred_1, stump_loc_1, stump_err_1, stump_val_1, stump_err_1, 1 - stump_err_1
    else:
        return 2, stump_pred_2, stump_loc_2, stump_err_2, stump_val_2, stump_err_2, 1 - stump_err_2


def get_error(classifiers, classifier_weights, data, data_labels, only_predict=False):
    err = 0
    label_predict = []
    for j in np.arange(0, data.shape[0], 1):
        pred_j = 0
        for k, w in zip(classifiers, classifier_weights):
            if k[1] == 1:
                pred_k_j = 1 if data[j, k[0] - 1] < k[4] else -1
            else:
                pred_k_j = -1 if data[j, k[0] - 1] < k[4] else 1
            pred_j += pred_k_j * w

        pred_j = np.sign(pred_j)
        if not only_predict:
            err += int(pred_j != data_labels[j])
        label_predict.append(pred_j)

    if not only_predict:
        err = err/data.shape[0]
    return err, label_predict


debug = False


def adaboost(num_weak_classifiers):
    weights_dist = np.ones(train_data.shape[0])/train_data.shape[0]
    first_classifier = get_classifier(weights_dist)
    classifiers = [first_classifier]
    classifier_weights = [0.5 * np.log(first_classifier[6] / first_classifier[5])]
    training_err = [get_error(classifiers, classifier_weights, train_data, train_label)[0]]
    test_err = [get_error(classifiers, classifier_weights, val_data, val_label)[0]]

    # Plotting training data
    plt.title("Scatter plot and Linear boundary : Training data")
    plt.xlabel("First column of moons.x.csv")
    plt.ylabel("Second column of moons.x.csv")
    plt.scatter(feature_1, feature_2, color=color, s=1)

    if first_classifier[0] - 1 == 0:
        plt.axvline(x=first_classifier[4])
    else:
        plt.axhline(y=first_classifier[4])

    plt.show()

    # Plotting testing data
    plt.title("Scatter plot and Linear boundary : Test data")
    plt.xlabel("First column of moons.x.csv")
    plt.ylabel("Second column of moons.x.csv")
    plt.scatter(val_data[:, 0], val_data[:, 1], color=np.array(['red' if x == -1 else 'green' for x in val_label]), s=1)

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
        training_err.append(get_error(classifiers, classifier_weights, train_data, train_label)[0])
        test_err.append(get_error(classifiers, classifier_weights, val_data, val_label)[0])

        if debug:
            # Plotting training data
            plt.title("Scatter plot and Linear boundary : Training data")
            plt.xlabel("First column of moons.x.csv")
            plt.ylabel("Second column of moons.x.csv")
            plt.scatter(feature_1, feature_2, color=color, s=1)

            if iter_classifier[0] - 1 == 0:
                plt.axvline(x=iter_classifier[4])
            else:
                plt.axhline(y=iter_classifier[4])

            plt.show()

    plt.xlabel("#Classifiers")
    plt.ylabel("Training error")
    plt.plot(training_err)
    plt.show()

    plt.xlabel("#Classifiers")
    plt.ylabel("Test error")
    plt.plot(test_err)
    plt.show()

    return classifiers, classifier_weights


ada_classifiers, ada_classifier_weights = adaboost(100)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 0.2, x.max() + 0.2
    y_min, y_max = y.min() - 0.2, y.max() + 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


xx, yy = make_meshgrid(val_data[:, 0], val_data[:, 1])
_, Z1 = get_error(classifiers=ada_classifiers, classifier_weights=ada_classifier_weights,
                  data=np.c_[xx.ravel(), yy.ravel()], data_labels=[], only_predict=True)
Z1 = np.array(Z1)
Z1 = Z1.reshape(xx.shape)
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.4)
ax.axis('off')
colors = ['red', 'green']
ax.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap=matplotlib.colors.ListedColormap(colors))
ax.set_title('Decision boundary, #Classifiers = 100')
plt.xlabel("First column of moons.x.csv", axes=ax)
plt.ylabel("Second column of moons.x.csv", axes=ax)
plt.show()

xx, yy = make_meshgrid(val_data[:, 0], val_data[:, 1])
_, Z1 = get_error(classifiers=ada_classifiers[0:5], classifier_weights=ada_classifier_weights[0:5],
                  data=np.c_[xx.ravel(), yy.ravel()], data_labels=[], only_predict=True)
Z1 = np.array(Z1)
Z1 = Z1.reshape(xx.shape)
fig, ax = plt.subplots()
ax.contourf(xx, yy, Z1, cmap=plt.cm.coolwarm, alpha=0.4)
ax.axis('off')
colors = ['red', 'green']
ax.scatter(train_data[:, 0], train_data[:, 1], c=train_label, cmap=matplotlib.colors.ListedColormap(colors))
ax.set_title('Decision boundary, #Classifiers = 5')
plt.xlabel("First column of moons.x.csv", axes=ax)
plt.ylabel("Second column of moons.x.csv", axes=ax)
plt.show()
