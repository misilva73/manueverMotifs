import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyclustering.cluster import kmedoids
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as shc
from dtaidistance import dtw


def compute_dtw_distance_matrix(ts_list, **kwargs):
    """
    This function computes the pairwise distance matrix of a list of time-series with Dynamic Time Warping distance.
    It is based on dtaidistance package
    :param ts_list: list of time-series to compare pairwise
    :param kwargs: extra arguments for the dtaidistance.dtw.distance_matrix() function
    :return: dist_matrix
    """
    start = time.time()
    dist_matrix_vec = dtw.distance_matrix(ts_list, **kwargs)
    dist_matrix = np.triu(dist_matrix_vec) + np.triu(dist_matrix_vec).T
    np.fill_diagonal(dist_matrix, 0)
    print('Distance matrix computed in {} minutes'.format(round((time.time()-start)/60, 1)))
    return dist_matrix


def run_kmedoids_evaluation(dist_matrix, k_list, n_runs):
    scores_list = []
    for k in k_list:
        k_scores = []
        for i in range(n_runs):
            initial_medoids = np.random.choice(dist_matrix.shape[0], k, replace=False)
            kmedoids_model = kmedoids.kmedoids(dist_matrix, initial_medoids, data_type='distance_matrix')
            kmedoids_model.process()
            clusters = kmedoids_model.get_clusters()
            cluster_labels = np.zeros((dist_matrix.shape[0],))
            for j in range(k):
                cluster_labels[clusters[j]] = j
            temp_score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
            k_scores.append(temp_score)
            # compute medioids consistency????
        scores_list.append(k_scores)
    model_scores_df = pd.DataFrame(scores_list) \
                        .assign(k=k_list) \
                        .melt(id_vars=['k'], var_name='n_run', value_name='score')
    # plot mean and standard deviation of scores for each k
    plt.figure(figsize=(15, 5))
    plt.suptitle('Silhouette analysis')
    sns.pointplot(x='k', y='score', data=model_scores_df, ci='sd')
    plt.ylabel('Silhouette score \n Mean & Standard deviation')
    plt.xlabel('Number of clusters')
    plt.show()
    return model_scores_df


def run_agglomerative_evaluation(dist_matrix, k_list):
    # get condensed distance vector - needed for scipy package
    triu_idx = np.triu_indices(dist_matrix.shape[0], 1)
    dist_vec = dist_matrix[triu_idx]
    method_list = ['average', 'single', 'complete']
    linkage_list = []
    global_scores = []
    # plot dendograms and compute scores
    fig1, axs = plt.subplots(ncols=3, figsize=(15, 8))
    plt.suptitle('Dendograms')
    for i in range(3):
        model = shc.linkage(dist_vec, method=method_list[i])
        linkage_list.append(model)
        shc.dendrogram(model, truncate_mode='lastp', p=25, orientation='right', distance_sort='descending', ax=axs[i])
        axs[i].set_title(method_list[i])
        method_scores = []
        for k in k_list:
            cluster_labels = shc.fcluster(model, k, criterion='maxclust')
            k_score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
            method_scores.append(k_score)
        global_scores.append(method_scores)
    # plot scores
    model_scores_df = pd.DataFrame(global_scores).T
    model_scores_df.columns = method_list
    model_scores_df = model_scores_df \
        .assign(k=k_list) \
        .melt(id_vars=['k'], var_name='method', value_name='score')
    plt.figure(figsize=(15, 5))
    plt.title('Silhouette analysis')
    sns.lineplot(x="k", y="score", hue="method", style='method', data=model_scores_df,
                 markers=True, estimator=None, ci=None)
    plt.ylabel('Silhouette score')
    plt.xlabel('Number of clusters')
    plt.xticks(k_list)
    plt.show()
    return model_scores_df


def fit_best_kmedoids_model(dist_matrix, k, n_runs):
    # initialize variables
    best_score = -1
    best_labels = None
    for i in range(n_runs):
        # set random initial medoids
        initial_medoids = np.random.choice(dist_matrix.shape[0], k, replace=False)
        # create K-Medoids algorithm for processing distance matrix instead of points
        kmedoids_model = kmedoids.kmedoids(dist_matrix,
                                           initial_medoids,
                                           data_type='distance_matrix')
        # run cluster algorithm and get clusters
        kmedoids_model.process()
        clusters = kmedoids_model.get_clusters()
        # get cluster labels from clusters variable
        cluster_labels = np.zeros((dist_matrix.shape[0],))
        for j in range(k):
            cluster_labels[clusters[j]] = j
        # Calculate Silhouette score
        temp_score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
        # Best model test
        if temp_score > best_score:
            best_score = temp_score
            best_labels = cluster_labels
    return best_labels, best_score


def fit_agglomerative_model(dist_matrix, k, method='single'):
    triu_idx = np.triu_indices(dist_matrix.shape[0], 1)
    dist_vec = dist_matrix[triu_idx]
    linkage_matrix = shc.linkage(dist_vec, method=method)
    cluster_labels = shc.fcluster(linkage_matrix, k, criterion='maxclust')
    score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    return cluster_labels, score

