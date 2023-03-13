import numpy as np


def dbscan(X, eps, min_samples):
    # Get distance and neighbors
    D = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=-1)
    neighborhoods = [np.where(D[i] <= eps)[0] for i in range(X.shape[0])]

    # Initialize all points as unvisited
    visited = np.zeros(X.shape[0], dtype=bool)

    # Initialize all labels as noise
    labels = np.full(X.shape[0], -1, dtype=int)

    # Start with cluster label 0
    cluster_label = 0

    # Iterate over each point in the dataset
    for i in range(X.shape[0]):
        # If the point has already been visited, skip it
        if visited[i]:
            continue

        # Mark the point as visited
        visited[i] = True

        # Get the neighborhood of the point
        neighbors = neighborhoods[i]

        # If the number of neighbors is less than the minimum required, mark
        # the point as noise
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            # Expand the cluster starting with this point
            cluster_label += 1
            expand_cluster(neighborhoods, visited, labels, i, neighbors, cluster_label, min_samples)

    return labels


def expand_cluster(neighborhoods, visited, labels, i, neighbors, cluster_label, min_samples):
    # Add the core point to the cluster
    labels[i] = cluster_label

    # Iterate over each neighbor of the core point
    for j in neighbors:
        # If the neighbor has not been visited, mark it as visited
        if not visited[j]:
            visited[j] = True

            # Get the neighborhood of the neighbor
            new_neighbors = neighborhoods[j]

            # If the number of neighbors is greater than or equal to the
            # minimum required, add the neighbor to the cluster
            if len(new_neighbors) >= min_samples:
                neighbors = np.union1d(neighbors, new_neighbors)

            # If the neighbor has not been assigned to a cluster, add it to
            # the current cluster
            if labels[j] == -1:
                labels[j] = cluster_label
