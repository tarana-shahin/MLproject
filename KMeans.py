#K Means Clustering
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import math

# st.set_option('deprecation.showPyplotGlobalUse', False)

iris = pd.read_csv("iris.csv")

df = iris
#df
df['categorical_species'] = np.zeros( df.shape[0]).astype(int)
#print (df)

X = df[{'sepal_length', 'sepal_width','petal_length', 'petal_width'}]


import numpy as np
import matplotlib.pyplot as plt

#np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


def predict( X, K, max_iters):
        #self.X = X
        n_samples, n_features = X.shape
        plot_steps = True
        plt = None
        clusters = [[] for _ in range(K)]
        centroids = []

    
        # initialize 
        random_sample_idxs = np.random.choice(n_samples, K, replace=False)
        centroids = [X[idx] for idx in random_sample_idxs]

        # Optimize clusters
        for _ in range(max_iters):
            # Assign samples to closest centroids (create clusters)
            clusters = _create_clusters( centroids, K, X)
            

            if plot_steps:
                plt = plot(clusters, X, centroids)

            # Calculate new centroids from the clusters
            centroids_old = centroids
            centroids = _get_centroids( clusters, X, K,  n_features)
            
            # check if clusters have changed
            if _is_converged(centroids_old, centroids, K):
                break

            if plot_steps:
                plt = plot(clusters, X, centroids)

        # Classify samples as the index of their clusters
        return _get_cluster_labels(clusters, n_samples), clusters, centroids, plt


def _get_cluster_labels( clusters, n_samples):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

def _create_clusters( centroids, K, X):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range( K)]
        for idx, sample in enumerate( X):
            centroid_idx = _closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

def _closest_centroid( sample, centroids):
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

def _get_centroids( clusters, X, K,  n_features):
        # assign mean value of clusters to centroids
        centroids = np.zeros( ( K,  n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean( X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

def _is_converged( centroids_old, centroids, K):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range( K)]
        return sum(distances) == 0

def plot(clusters, X, centroids):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate( clusters):
            point = X[index].T
            ax.scatter(*point)

        for point in centroids:
            ax.scatter(*point, marker="x", linewidth=12)

        plt.show()
        return plt


def kmeans_cluster_model( K , no_of_iterations) :

  X_arr = np.array( X)
  y_pred, clusters, centroid, plt = predict( X_arr, K, no_of_iterations )

  print( ' predictions : ', y_pred )

  return X, y_pred, clusters, centroid, plt
  



def k_means( k, no_of_iterations) :
    return kmeans_cluster_model( k, no_of_iterations)

