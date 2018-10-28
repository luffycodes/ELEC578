import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# load feature variables and their names
X = np.loadtxt('./hitters/hitters.x.csv', delimiter=',', skiprows=1)
with open('./hitters/hitters.x.csv', 'r') as f:
    X_colnames = next(csv.reader(f))
    # load salaries
    y = np.loadtxt('./hitters/hitters.y.csv', delimiter=',', skiprows=1)

std_cols = np.std(X, axis=0, dtype=np.float64)
mean_cols = np.mean(X, axis=0, dtype=np.float64)
for i, (std_col, mean_col) in enumerate(zip(std_cols, mean_cols)):
    X[:, i] *= 1/std_col
    # X[:, i] -= mean_col

X_hat = np.insert(X, 0, 1, axis=1)
I_hat = np.insert(np.insert(np.identity(X.shape[1]), 0, 0, axis=1), 0, 0, axis=0)

l = np.logspace(-3, 7, 100, endpoint=True)

theta_ridges = []
theta_ridge_norm = []
theta_ridge_coefficient_norm = []

# Plotting l2 norm of regression estimate (without the first entry) for each lambda
plt.xlabel('log lambda')
plt.ylabel('log of l2 norm of regression estimate')
for _l in l:
    theta_ridges.append(np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X_hat), X_hat) + _l * I_hat), np.transpose(X_hat)), y))
    theta_ridge_norm.append(np.linalg.norm(theta_ridges[-1][1:]))
plt.semilogx(l, theta_ridge_norm)
plt.show()

# Plotting l2 norm of ridge_estimate - ls_estimate for starting lambdas as they are close to 0
theta_lr = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X_hat), X_hat)), np.transpose(X_hat)), y)
norm_theta_lr_minus_theta_ridge = []

for theta_ridge in theta_ridges:
    norm_theta_lr_minus_theta_ridge.append(np.linalg.norm(theta_lr - theta_ridge))

plt.xlabel('log lambda')
plt.ylabel('l2 norm of ridge_estimate - ls_estimate')
plt.semilogx(l[0:50], norm_theta_lr_minus_theta_ridge[0:50])
plt.show()

# Plotting each coefficient of regression estimate for each lambda
plt.xlabel('log lambda')
plt.ylabel('coefficient')
for i, c in enumerate(np.array(theta_ridges).T):
    plt.semilogx(l, c, label=[i])
plt.legend()
plt.show()

# Cross validation
index = np.arange(0, 263, 1)
np.random.seed(12)
np.random.shuffle(index)
validation_err = []
best_ridge_estimate = []

for _l in l:
    sum_err = 0
    cv_ridge_estimate_sum = np.zeros(20)
    for k in np.arange(0, 5):
        # Creating training and validation indices, indices were shuffled earlier
        val_idx = index[np.arange(k * 52, k * 52 + 52, 1)]
        if k == 4:
            val_idx = index[np.arange(k * 52, 263, 1)]
        train_idx = np.setdiff1d(index, val_idx)

        # Estimating estimate on training indices
        cv_ridge_estimate = np.dot(
            np.dot(np.linalg.inv(np.dot(np.transpose(X_hat[train_idx]), X_hat[train_idx]) + _l * I_hat),
                   np.transpose(X_hat[train_idx])), y[train_idx])
        cv_ridge_estimate_sum = cv_ridge_estimate_sum + cv_ridge_estimate

        # Predicting y on validation indices & calculating error
        val_prediction = np.dot(X_hat[val_idx], cv_ridge_estimate)
        val_err = np.power(np.linalg.norm(val_prediction - y[val_idx]), 2) / len(val_idx)
        sum_err = sum_err + val_err

    avg_err = sum_err/5
    validation_err.append(avg_err)

print("kfold optimal lambda", l[np.argmin(validation_err)])
print("kfold least validation error", np.min(validation_err))
best_ridge_estimate = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X_hat), X_hat) + l[np.argmin(validation_err)] * I_hat),
               np.transpose(X_hat)), y)
for (weightage, col_name) in zip(best_ridge_estimate[1:], X_colnames):
    print(col_name, np.abs(weightage))

plt.xlabel('log lambda')
plt.ylabel('Kfold log validation error')
plt.semilogx(l, validation_err)
plt.show()

# Sklearn Cross validation to cross validate
kfold = KFold(n_splits=5)
validation_err = []

for _l in l:
    sum_err = 0
    cv_ridge_estimate_sum = np.zeros(20)
    for train_idx, val_idx in kfold.split(X_hat):
        cv_ridge_estimate = np.dot(
            np.dot(np.linalg.inv(np.dot(np.transpose(X_hat[train_idx]), X_hat[train_idx]) + _l * I_hat),
                   np.transpose(X_hat[train_idx])), y[train_idx])
        cv_ridge_estimate_sum = cv_ridge_estimate_sum + cv_ridge_estimate
        val_prediction = np.dot(X_hat[val_idx], cv_ridge_estimate)
        val_err = np.power(np.linalg.norm(val_prediction - y[val_idx]), 2) / len(val_idx)
        sum_err = sum_err + val_err

    avg_err = sum_err/5
    validation_err.append(avg_err)

print("sklearn kfold optimal lambda", l[np.argmin(validation_err)])
print("sklearn kfold least validation error", np.min(validation_err))
best_ridge_estimate = np.dot(
        np.dot(np.linalg.inv(np.dot(np.transpose(X_hat), X_hat) + l[np.argmin(validation_err)] * I_hat),
               np.transpose(X_hat)), y)
for (weightage, col_name) in zip(best_ridge_estimate[1:], X_colnames):
    print(col_name, np.abs(weightage))
print("sklearn kfold optimal best_ridge_estimate", best_ridge_estimate)

plt.xlabel('log lambda')
plt.ylabel('sklearn kfold log validation error')
plt.loglog(l, validation_err)
plt.show()
