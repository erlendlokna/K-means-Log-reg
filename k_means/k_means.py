import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)
import random
import time

class KMeans:
    def __init__(self, X, k, it_criteria=1000, silent=False):
        self.X = X.to_numpy(); self.k=k;
        self.it_criteria = it_criteria; self.silent = silent

    def fit(self):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            k (int): number of clusters
        """
        if not self.silent:print("starting fit to data..")
 
        t0 = time.time()
        num_its = 0

        X = self.X
        k = self.k
        it_criteria = self.it_criteria
        
        #grabbing random initial centroids from X
        centroids = np.take(X, random.sample(range(len(X)), k), axis=0)

        labels = np.zeros(len(X)) #labels
        
        while(True):
            #creating a empty cluster list
            updated_labels = np.zeros(len(X))

            #checking distances and adding points to their respective cluster
            for i, point in enumerate(X):
                distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
                closest_cluster = np.argmin(distances)
                updated_labels[i] = closest_cluster

            #calculating new centroids
            centroids = np.array([np.mean(X[updated_labels == j], axis=0) for j in range(k)])

            #finish criteria
            if all(updated_labels == labels) or num_its >= it_criteria: break
            
            #update
            labels = updated_labels
            num_its += 1

        self.labels = labels
        self.centroids = centroids

        t1 = time.time()

        if not self.silent: print(f"fit complete.. \n number of its: {num_its} \n time: {round(t1 - t0, 5)}s")

    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m integer array with cluster assignments
            for each point. E.g., if X is a 10xn matrix and 
            there are 3 clusters, then a possible assignment
            could be: array([2, 0, 0, 1, 2, 1, 1, 0, 2, 2])
        """
        X = X.to_numpy()
        

        predictions = np.zeros(len(X))
        centroids = self.centroids
    
        #checking distances and adding points to their respective cluster
        for i, point in enumerate(X):
            distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
            closest_cluster = np.argmin(distances)
            predictions[i] = closest_cluster
        
        return predictions.astype(int)
    
    
    def get_centroids(self):
        """
        Returns the centroids found by the K-mean algorithm
        
        Example with m centroids in an n-dimensional space:
        >>> model.get_centroids()
        numpy.array([
            [x1_1, x1_2, ..., x1_n],
            [x2_1, x2_2, ..., x2_n],
                    .
                    .
                    .
            [xm_1, xm_2, ..., xm_n]
        ])
        """
        return self.centroids
    
    @property
    def WSS(self):
        X = self.X
        wss = 0

        for i in range(self.k):
            cluster_points = X[self.labels == i]
            centroid = self.centroids[i]
            cluster_wss = np.sum((cluster_points - centroid) ** 2)
            wss += cluster_wss

        return wss

    
    
    
    
# --- Some utility functions 
def kmeans_tester(X, ks, test_per=5):
    wss = np.zeros(len(ks))
    for i, k in enumerate(ks):
        k_wss = np.zeros(test_per)
        for j in range(test_per):
            m = KMeans(X, k, silent=True)
            m.fit()
            k_wss[j] = m.WSS
        wss[i] = np.mean(k_wss)
    return wss

    
def euclidean_distance(x, y):
    """
    Computes euclidean distance between two sets of points 
    
    Note: by passing "y=0.0", it will compute the euclidean norm
    
    Args:
        x, y (array<...,n>): float tensors with pairs of 
            n-dimensional points 
            
    Returns:
        A float array of shape <...> with the pairwise distances
        of each x and y point
    """
    return np.linalg.norm(x - y, ord=2, axis=-1)

def cross_euclidean_distance(x, y=None):
    """
    
    
    """
    y = x if y is None else y 
    assert len(x.shape) >= 2
    assert len(y.shape) >= 2
    return euclidean_distance(x[..., :, None, :], y[..., None, :, :])


def euclidean_distortion(X, z):
    """
    Computes the Euclidean K-means distortion
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the raw distortion measure 
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    distortion = 0.0
    clusters = np.unique(z)
    for i, c in enumerate(clusters):
        Xc = X[z == c]
        mu = Xc.mean(axis=0)
        distortion += (np.linalg.norm((Xc - mu) ** 2)).sum(axis=0)
    return distortion


def euclidean_silhouette(X, z):
    """
    Computes the average Silhouette Coefficient with euclidean distance 
    
    More info:
        - https://www.sciencedirect.com/science/article/pii/0377042787901257
        - https://en.wikipedia.org/wiki/Silhouette_(clustering)
    
    Args:
        X (array<m,n>): m x n float matrix with datapoints 
        z (array<m>): m-length integer vector of cluster assignments
    
    Returns:
        A scalar float with the silhouette score
    """
    X, z = np.asarray(X), np.asarray(z)
    assert len(X.shape) == 2
    assert len(z.shape) == 1
    assert X.shape[0] == z.shape[0]
    
    # Compute average distances from each x to all other clusters
    clusters = np.unique(z)
    D = np.zeros((len(X), len(clusters)))
    for i, ca in enumerate(clusters):
        for j, cb in enumerate(clusters):
            in_cluster_a = z == ca
            in_cluster_b = z == cb
            d = cross_euclidean_distance(X[in_cluster_a], X[in_cluster_b])
            div = d.shape[1] - int(i == j)
            D[in_cluster_a, j] = d.sum(axis=1) / np.clip(div, 1, None)
    
    # Intra distance 
    
    a = D[np.arange(len(X)), z]
    # Smallest inter distance 
    inf_mask = np.where(z[:, None] == clusters[None], np.inf, 0)
    b = (D + inf_mask).min(axis=1)
    
    return np.mean((b - a) / np.maximum(a, b))
  
