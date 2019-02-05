import time
import numpy as np
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


def get_motif_k_neighbors(motif_dic_list, motif_index, dist_mat, k):
    dist_vec = dist_mat[motif_index]
    sorted_index = np.argsort(dist_vec)
    neighbors_dic_list = [motif_dic_list[i] for i in sorted_index[0:k]]
    return neighbors_dic_list


def return_motif_index_with_pattern(motif_dic_list, pattern):
    pattern_list = [dic['pattern'] for dic in motif_dic_list]
    if pattern in pattern_list:
        return pattern_list.index(pattern)
    else:
        return None
