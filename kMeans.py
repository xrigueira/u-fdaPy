import numpy as np
import seaborn as sns; sns.set()
from matplotlib import pyplot as plt

# Create dataset
from sklearn.datasets._samples_generator import make_blobs

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.scatter(X[:, 0], X[:, 1], s=50)

# Implement k-Means
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X) # this is the assigned cluster label for each point
# print(y_kmeans)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.show()

# Simple implementation
from sklearn.metrics import pairwise_distances_argmin


def find_clusters(X, n_clusters, rseed=2):

    # Randonmly start up the clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:

        # Assing labels based on the closest center
        labels = pairwise_distances_argmin(X, centers)

        # Find new centers
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])

        # Check for convergence
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X, 4)
print(centers)
print(labels)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');

plt.show()