import matplotlib.pyplot as plt
from sklearn import neighbors
from dbscan import dbscan


def plot_clusters(X_train, X_test, y_train, y_test, label):
    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict the classes for each point on the meshgrid using k-NN
    Z = knn.predict(X_test)

    labels = dbscan(X_test, eps=0.5, min_samples=5)

    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].scatter(X_test[:, 0], X_test[:, 1], c=labels, edgecolors='black', cmap=plt.cm.RdYlBu)
    axes[0].set_title('DBSCAN Clustering')
    axes[1].scatter(X_test[:, 0], X_test[:, 1], c=Z, edgecolors='black', cmap=plt.cm.RdYlBu)
    axes[1].set_title('k-NN Classification')
    axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='black', cmap=plt.cm.RdYlBu)
    axes[2].set_title('Real Data')
    fig.suptitle(label)
    plt.show()
