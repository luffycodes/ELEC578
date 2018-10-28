import numpy as np
import pandas as pd

spambase = np.loadtxt('spambase.data', delimiter=',')
np.random.shuffle(spambase)

X_train = spambase[:2000, :-1]
y_train = spambase[:2000, -1].astype(int)
y_test = spambase[2000:, -1].astype(int)

# Calculating median & Quantizing
X_train_median = np.median(X_train, axis=0)
X_train_quantized = np.greater(X_train, X_train_median).astype(int)
df_X_test_quantized = pd.DataFrame(np.greater(spambase[2000:, :-1], X_train_median).astype(int))

# Calculating P[X_i={0,1}|y] & P[y={0,1}] from training data
df_X_train_quantized_with_y = pd.DataFrame(np.append(X_train_quantized, y_train.reshape((y_train.shape[0], 1)), axis=1))
prob_X_train_cols_0_given_y = 1 - df_X_train_quantized_with_y.groupby([57]).mean()
prob_y_1 = df_X_train_quantized_with_y[57].mean()

# Predicting & Error calculations
df_prob_X_test_cols_given_y_0 = df_X_test_quantized.apply(
    lambda x: np.abs(x - prob_X_train_cols_0_given_y.iloc[0, x.name]), axis=0)
df_prob_X_test_cols_given_y_1 = df_X_test_quantized.apply(
    lambda x: np.abs(x - prob_X_train_cols_0_given_y.iloc[1, x.name]), axis=0)

df_prob = pd.concat([df_prob_X_test_cols_given_y_0.prod(axis=1) * (1 - prob_y_1),
                     df_prob_X_test_cols_given_y_1.prod(axis=1) * prob_y_1], axis=1)

test_err = np.sum(np.abs(df_prob.idxmax(axis=1) - y_test))/2601
popular_test_err = np.sum(np.abs(0 - y_test))/2601
print(test_err, " ", popular_test_err)
