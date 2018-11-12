import numpy as np
import matplotlib.pyplot as plt

n = 1000
sqrt_2 = np.sqrt(2)


def get_data():
    data_arr = []
    for _ in np.arange(0, n, 1):
        i = np.random.randint(0, 100)
        if i < 4:
            data_arr.append(np.array([0, np.random.normal(0, np.sqrt(0.4)), np.random.normal(0, np.sqrt(0.4))]))
        elif i - 4 < 30:
            data_arr.append(np.array([1, np.random.normal(0, sqrt_2), np.random.normal(5, sqrt_2)]))
        elif i - 34 < 30:
            data_arr.append(np.array([2, np.random.normal(5, sqrt_2), np.random.normal(0, sqrt_2)]))
        else:
            data_arr.append(np.array([3, np.random.normal(5, sqrt_2), np.random.normal(5, sqrt_2)]))
    return np.asarray(data_arr)


def get_cluster_assignments(cluster_means_arr):
    J = 0
    cluster_assignment_arr = []
    cluster_size = np.zeros(4)

    for pt in data:
        dist_pt_to_cluster_mean = []
        for cluster_mean in cluster_means_arr:
            dist_pt_to_cluster_mean.append(np.power(cluster_mean[0] - pt[1], 2) + np.power(cluster_mean[1] - pt[2], 2))

        cluster_assignment = np.argmin(dist_pt_to_cluster_mean)
        J += np.min(dist_pt_to_cluster_mean)
        cluster_assignment_arr.append(cluster_assignment)
        cluster_size[cluster_assignment] += 1

    return cluster_assignment_arr, cluster_size, J


def update_cluster_means(cluster_assignment_arr, cluster_size):
    iter_cluster_means_arr = np.zeros((4, 2))

    for pt, cluster_assignment in zip(data, cluster_assignment_arr):
        iter_cluster_means_arr[cluster_assignment] += pt[1:]

    for i in np.arange(0, 4, 1):
        iter_cluster_means_arr[i] /= cluster_size[i]

    return iter_cluster_means_arr


def get_plus_plus_cluster_means_initializations(k):
    cluster_means_arr = []
    cluster_means_arr.append(data[np.random.randint(0, 1000), 1:])

    for _ in np.arange(1, k, 1):
        prob_dist = []
        prob_dist_sum = 0
        for pt in data:
            dist_pt_to_cluster_mean = []
            for cluster_mean in cluster_means_arr:
                dist_pt_to_cluster_mean.append(np.power(cluster_mean[0] - pt[1], 2) + np.power(cluster_mean[1] - pt[2], 2))

            prob_dist.append(np.min(dist_pt_to_cluster_mean))
            prob_dist_sum += np.min(dist_pt_to_cluster_mean)

        plus_plus_rand_index = np.random.choice(np.arange(0, 1000, 1), p=prob_dist / prob_dist_sum)
        cluster_means_arr.append(data[plus_plus_rand_index, 1:])

    return cluster_means_arr


def get_random_cluster_means_initializations(k):
    return np.random.rand(k, 2)


data = get_data()

J_arr_plus = []
cluster_means_arr = get_plus_plus_cluster_means_initializations(4)
for iter_num in np.arange(0, 20, 1):
    cluster_assignment_arr, cluster_size, J = get_cluster_assignments(cluster_means_arr)
    cluster_means_arr = update_cluster_means(cluster_assignment_arr, cluster_size)
    J_arr_plus.append(J)

plt.plot(J_arr_plus)


J_arr = []
cluster_means_arr = get_random_cluster_means_initializations(4)
for iter_num in np.arange(0, 20, 1):
    cluster_assignment_arr, cluster_size, J = get_cluster_assignments(cluster_means_arr)
    cluster_means_arr = update_cluster_means(cluster_assignment_arr, cluster_size)
    J_arr.append(J)

plt.plot(J_arr)

plt.show()
