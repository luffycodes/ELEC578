import numpy as np
import matplotlib.pyplot as plt

n = 1000
d = 4
x = np.random.uniform(0, 1, n)
e = np.random.normal(0, 0.5, n)
y = x + e

x_axes = np.random.rand(n)
x_axes = np.sort(x_axes)
# plt.scatter(x_axes, y)

line_slope_estimate = np.dot(x, y) / np.dot(x, x)
# plt.plot(x_axes, [line_slope_estimate * t1 for t1 in x_axes])
# plt.show()

vandermat_X = np.polynomial.polynomial.polyvander(x, deg=4)
poly_X = np.polynomial.polynomial.polyval(x, c=[1.0546875, -11.25, 41.25, -60, 30])
poly_Y = poly_X + np.random.normal(0, 0.1, n)
plt.plot(x_axes, poly_Y)


poly_coefficients_estimate = np.dot(
    np.dot(np.linalg.inv(np.dot(np.transpose(vandermat_X), vandermat_X)), np.transpose(vandermat_X)), poly_Y)

plt.plot(x_axes, np.polynomial.polynomial.polyval(x_axes, c=poly_coefficients_estimate))
plt.show()

np_fit = np.polynomial.polynomial.polyfit(x, poly_Y, deg=4)
print()





