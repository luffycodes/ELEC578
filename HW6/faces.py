import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('yalefaces.csv', delimiter=',')
data_centered = data - data.mean(axis=1, keepdims=True)

P, D, Q = np.linalg.svd(data_centered, full_matrices=False)
eigen_value_mat = np.power(D, 2)

plt.plot(eigen_value_mat)
plt.xlabel("principal component index")
plt.ylabel("eigenvalue")
plt.show()

energy_dist = np.cumsum(eigen_value_mat) / np.sum(eigen_value_mat) * 100
print(np.argmax(energy_dist > 95.00))
print(np.argmax(energy_dist > 99.00))

eigen_faces = plt.figure()
ax = eigen_faces.add_subplot(4, 5, 1)
ax.imshow(data.mean(axis=1, keepdims=True).reshape((48, 42)), cmap='gray')
ax.axis('off')

for i in np.arange(2, 21, 1):
    ax = eigen_faces.add_subplot(4, 5, i)
    ax.imshow(P[:, i].reshape((48, 42)), cmap='gray')
    ax.axis('off')

eigen_faces.show()

