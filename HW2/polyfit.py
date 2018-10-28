import numpy as np
import matplotlib.pyplot as plt

# Part 5
n_range = np.concatenate((np.arange(2, 14, 3), np.arange(15, 125, 15)))
d_range = np.concatenate((np.arange(0, 10, 1), np.arange(10, 50, 5)))
noise_range = np.arange(0, 10, 1)
for n in n_range:
    alpha = np.random.uniform(-1, 1, n)
    y = alpha + 1
    avg_err_d = []
    for d in d_range:
        sum_err = 0
        for x in noise_range:
            z = np.random.normal(0, 1, n)
            y_noise = y + z
            np_fit = np.polynomial.polynomial.polyfit(alpha, y_noise, deg=d)
            sum_err = sum_err + np.power(np.linalg.norm(np.polynomial.polynomial.polyval(alpha, np_fit) - y), 2) / n
        avg_err = sum_err / len(noise_range)
        avg_err_d.append(avg_err)
    plt.semilogy(d_range, avg_err_d, label="n="+str(n))
plt.xlabel('degree of polynomial')
plt.ylabel('log error')
plt.legend()
plt.show()


# Part 6
n_range = np.concatenate((np.arange(2, 14, 3), np.arange(15, 125, 15)))
d_range = np.concatenate((np.arange(0, 10, 1), np.arange(10, 50, 5)))
noise_range = np.arange(0, 10, 1)
for n in n_range:
    alpha = np.random.uniform(-4, 3, n)
    y = np.exp(alpha)
    avg_err_d = []
    for d in d_range:
        sum_err = 0
        for x in noise_range:
            z = np.random.normal(0, 1, n)
            y_noise = y + z
            np_fit = np.polynomial.polynomial.polyfit(alpha, y_noise, deg=d)
            sum_err = sum_err + np.power(np.linalg.norm(np.polynomial.polynomial.polyval(alpha, np_fit) - y), 2) / n
        avg_err = sum_err / len(noise_range)
        avg_err_d.append(avg_err)
    plt.semilogy(d_range, avg_err_d, label="n="+str(n))
plt.xlabel('degree of polynomial')
plt.ylabel('log error')
plt.legend()
plt.show()




