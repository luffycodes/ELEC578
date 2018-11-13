import numpy as np
import matplotlib.pyplot as plt


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


def get_cluster_assignments(cluster_means_arr, k=4):
    J = 0
    cluster_assignment_arr = []
    cluster_size = np.zeros(k)

    for pt in data:
        dist_pt_to_cluster_mean = []
        for cluster_mean in cluster_means_arr:
            dist_pt_to_cluster_mean.append(np.power(cluster_mean[0] - pt[1], 2) + np.power(cluster_mean[1] - pt[2], 2))

        cluster_assignment = np.argmin(dist_pt_to_cluster_mean)
        J += np.min(dist_pt_to_cluster_mean)
        cluster_assignment_arr.append(cluster_assignment)
        cluster_size[cluster_assignment] += 1

    return cluster_assignment_arr, cluster_size, J


def update_cluster_means(cluster_assignment_arr, cluster_size, k=4):
    iter_cluster_means_arr = np.zeros((k, 2))

    for pt, cluster_assignment in zip(data, cluster_assignment_arr):
        iter_cluster_means_arr[cluster_assignment] += pt[1:]

    for i in np.arange(0, k, 1):
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


def run_kmeans(k=4, kmeans_plus_plus=True, plot=False):
    j_arr_plus = []
    if kmeans_plus_plus:
        cluster_means_arr = get_plus_plus_cluster_means_initializations(k)
    else:
        cluster_means_arr = get_random_cluster_means_initializations(k)
    for _ in np.arange(0, 20, 1):
        cluster_assignment_arr, cluster_size, J = get_cluster_assignments(cluster_means_arr, k)
        cluster_means_arr = update_cluster_means(cluster_assignment_arr, cluster_size, k)
        j_arr_plus.append(J)

    if plot:
        if kmeans_plus_plus:
            plt.plot(j_arr_plus, label="k-means++")
        else:
            plt.plot(j_arr_plus, label="k-means")
        plt.ylabel("J")
        plt.xlabel("iterations")

    return j_arr_plus


n = 1000
sqrt_2 = np.sqrt(2)
data = get_data()

run_kmeans(kmeans_plus_plus=True, plot=True)
run_kmeans(kmeans_plus_plus=False, plot=True)

plt.legend()
plt.show()

bayes_info = []
for num_clusters in np.arange(2, 15, 1):
    n_ = run_kmeans(k=num_clusters)[-1] + num_clusters * np.log(n)
    bayes_info.append(n_)

plt.ylabel("Bayes Info")
plt.xlabel("clusters")
plt.plot(bayes_info)
plt.show()
