import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def relabel(labels):
    # make labels from 0, m
    unique_labels = sorted(set(labels))
    mapping = {label: i for i, label in enumerate(unique_labels)}
    new_labels = np.empty_like(labels, dtype=int)
    for i, l in enumerate(labels):
        new_labels[i] = mapping[l]
    return new_labels


def convert_probs_to_labels(samples, R, point_estimate_method="hc"):
    if point_estimate_method == "hc":
        point_clusters = cluster_point_estimate_hc(R)
    elif point_estimate_method == "dahl":
        point_clusters = cluster_point_estimate_dahls(samples, R)
    elif point_estimate_method == "binder":
        point_clusters = cluster_point_estimate_binder(samples, R)
    else:
        raise TabError("Unknown point estimate method!")
    
    labels = relabel(point_clusters)
    return labels


def cluster_point_estimate_hc(R, threshold=0.5):
    # 1. Convert similarity (co-occurrence probabilities) to distance
    distance_matrix = 1.0 - R
    # Best practice: ensure the diagonal is exactly 0.0 before converting, 
    # as floating-point inaccuracies can sometimes upset squareform()
    np.fill_diagonal(distance_matrix, 0.0)
    # Convert to a condensed distance matrix
    condensed_dist = squareform(distance_matrix)
    # 2. Perform hierarchical clustering
    # hc_sol = linkage(condensed_dist, method='single')
    hc_sol = linkage(condensed_dist, method='average')
    # cut the tree
    point_clusters = fcluster(hc_sol, t=threshold, criterion='distance')
    return point_clusters

def cluster_point_estimate_dahls(samples, R):
    n = R.shape[0]
    m = len(samples)

    idx1, idx2 = np.triu_indices(n=n, k=1)
    mse = np.zeros(m)
    for i, sample_i in enumerate(samples):
        mse[i] = np.sum(R[idx1, idx2] - 1*(sample_i[:, None] == sample_i[None, :])[idx1, idx2])**2
    
    best_idx = np.argmin(mse)
    return samples[best_idx]


def cluster_point_estimate_binder(samples, R):
    n = R.shape[0]
    idx1, idx2 = np.triu_indices(n=n, k=1)
    r = R[idx1, idx2]  # (n choose 2,) vector, computed once

    binder_losses = np.empty(len(samples))
    for l, sample_l in enumerate(samples):
        same = sample_l[idx1] == sample_l[idx2]      # boolean (n choose 2,) vector
        binder_losses[l] = np.sum(np.where(same, 1 - r, r))

    best_idx = np.argmin(binder_losses)

    # slow version below without vectorizaton
    # n = R.shape[0]
    # m = len(samples)

    # idx1, idx2 = np.triu_indices(n=n, k=1)

    # binder_losses = np.zeros(m)
    # for l, sample_l in enumerate(samples):
    #     for i,j in zip(idx1, idx2):
    #         if sample_l[i] == sample_l[j]:
    #             binder_losses[l] += 1-R[i,j]
    #         else:
    #             binder_losses[l] += R[i,j]

    # best_idx = np.argmin(binder_losses)
    return samples[best_idx]


def find_medoids(X, centroids):

    k = centroids.shape[0]
    medoid_idx = np.zeros(k, dtype=int)
    for i in range(k):
        dists_sq = cdist(X, centroids[i].reshape(1, -1), metric="sqeuclidean").flatten()
        medoid_idx[i] = np.argmin(dists_sq)
    return medoid_idx


def compute_centers_neighbors(X, labels_micro, unique_labels, current_k, n_knn, d):

    # compute micro centers
    centers = np.zeros((current_k, d))
    for i, k in enumerate(unique_labels):
        centers[i] = np.mean(X[labels_micro == k], axis=0)
    
    # distances between micro clusters
    if current_k == 1:
        D_centers = np.array([[0.0]])
        knn_idx = np.array([[0]])
    else:
        # compute pairwise centroid distances
        sq_dist = pdist(centers, metric="sqeuclidean")
        D_centers = squareform(sq_dist)
        # scale the distance by the median distance
        vals = D_centers[D_centers > 0]
        D_centers = D_centers/np.median(vals)

        # knn
        n_neighbors = min(current_k, n_knn)
        knn = NearestNeighbors(n_neighbors=n_neighbors)
        knn.fit(centers)
        _, knn_idx = knn.kneighbors(centers)
    
    # find medoids (closest point to the centroid)
    medoid_ids = find_medoids(X, centers)

    return centers, D_centers, knn_idx, medoid_ids