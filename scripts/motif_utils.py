import time
import numpy as np
import pandas as pd
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
    print(
        "Distance matrix computed in {} minutes".format(
            round((time.time() - start) / 60, 1)
        )
    )
    return dist_matrix


def get_motif_k_neighbors(motif_dic_list, motif_index, dist_mat, k):
    dist_vec = dist_mat[motif_index]
    sorted_index = np.argsort(dist_vec)
    neighbors_dic_list = [motif_dic_list[i] for i in sorted_index[0:k]]
    return neighbors_dic_list


def get_motif_turn_labels_df(motif_dic_list, trip_df):
    motif_labels_list = []
    for motif_dic in motif_dic_list:
        motif_events = []
        motif_lcs = []
        for member_pointers in motif_dic["members_ts_pointers"]:
            member_events = list(
                trip_df.iloc[member_pointers].groupby("event_type").size().index
            )
            motif_events += member_events
            member_lc = list(
                trip_df.iloc[member_pointers].groupby("lc_event").size().index
            )
            motif_lcs += member_lc
        motif_row = {
            "pattern": motif_dic["pattern"],
            "mdl_cost": motif_dic["mdl_cost"],
            "mean_dist": motif_dic["mean_dist"],
            "pattern_len": len(motif_dic["center_ts_pointers"]),
            "n_motifs": len(motif_dic["members_ts_pointers"]),
            "turn_events": sum(np.array(motif_events) == 2),
            "lc_right": sum(np.array(motif_lcs) == 1),
            "lc_left": sum(np.array(motif_lcs) == -1),
        }
        motif_labels_list.append(motif_row)
    motif_labels_df = pd.DataFrame(motif_labels_list)
    motif_labels_df["turn_ratio"] = (
        motif_labels_df["turn_events"] / motif_labels_df["n_motifs"]
    )
    motif_labels_df["lc_right_ratio"] = (
        motif_labels_df["lc_right"] / motif_labels_df["n_motifs"]
    )
    motif_labels_df["lc_left_ratio"] = (
        motif_labels_df["lc_left"] / motif_labels_df["n_motifs"]
    )
    motif_labels_df = motif_labels_df[
        [
            "pattern",
            "mdl_cost",
            "mean_dist",
            "pattern_len",
            "n_motifs",
            "turn_events",
            "turn_ratio",
            "lc_right",
            "lc_right_ratio",
            "lc_left",
            "lc_left_ratio",
        ]
    ]
    return motif_labels_df


def get_motif_lon_labels_df(motif_dic_list, trip_df):
    motif_labels_list = []
    for motif_dic in motif_dic_list:
        motif_events = []
        for member_pointers in motif_dic["members_ts_pointers"]:
            member_events = list(
                trip_df.iloc[member_pointers].groupby("event_type").size().index
            )
            motif_events += member_events
        motif_row = {
            "pattern": motif_dic["pattern"],
            "mdl_cost": motif_dic["mdl_cost"],
            "mean_dist": motif_dic["mean_dist"],
            "pattern_len": len(motif_dic["center_ts_pointers"]),
            "n_motifs": len(motif_dic["members_ts_pointers"]),
            "acc_events": sum(np.array(motif_events) == 3),
            "brake_events": sum(np.array(motif_events) == 1),
        }
        motif_labels_list.append(motif_row)
    motif_labels_df = pd.DataFrame(motif_labels_list)
    motif_labels_df["acc_ratio"] = (
        motif_labels_df["acc_events"] / motif_labels_df["n_motifs"]
    )
    motif_labels_df["brake_ratio"] = (
        motif_labels_df["brake_events"] / motif_labels_df["n_motifs"]
    )
    motif_labels_df = motif_labels_df[
        [
            "pattern",
            "mdl_cost",
            "mean_dist",
            "pattern_len",
            "n_motifs",
            "acc_events",
            "acc_ratio",
            "brake_events",
            "brake_ratio",
        ]
    ]
    return motif_labels_df


def get_limits_df(motif_dic_list, trip_df):
    ts_df = trip_df[["timestamp"]]
    limits_df_list = []
    for motif_dic in motif_dic_list:
        member_pointers = motif_dic["members_ts_pointers"]
        pattern = str(motif_dic["pattern"])
        limits_list = [(p[0], p[-1], pattern) for p in member_pointers]
        temp_limits_df = pd.DataFrame(limits_list)
        temp_limits_df.columns = ["start", "end", "pattern"]
        limits_df_list.append(temp_limits_df)
    limits_df = (
        pd.concat(limits_df_list)
        .join(ts_df, on="start")
        .rename(columns={"timestamp": "start_ts"})
        .join(ts_df, on="end")
        .rename(columns={"timestamp": "end_ts"})
    )
    return limits_df


def compute_lc_detection_rate(motif_dic_list, trip_df):
    all_members_ts = get_all_members_ts(motif_dic_list, trip_df)
    trip_lc_ts = trip_df.loc[trip_df["lc_event"] != 0]["timestamp"].values
    if len(trip_lc_ts) == 0:
        detection_rate = None
    else:
        detected_lc = [1 for lc_ts in trip_lc_ts if lc_ts in all_members_ts]
        detection_rate = sum(detected_lc) / len(trip_lc_ts)
    return detection_rate


def compute_turn_detection_rate(motif_dic_list, trip_df):
    all_members_ts = get_all_members_ts(motif_dic_list, trip_df)
    trip_turn_ts = trip_df.loc[trip_df["event_type"] == 2]["timestamp"].values
    detected_turns = [1 for lc_ts in trip_turn_ts if lc_ts in all_members_ts]
    detection_rate = sum(detected_turns) / len(trip_turn_ts)
    return detection_rate


def compute_acc_detection_rate(motif_dic_list, trip_df):
    all_members_ts = get_all_members_ts(motif_dic_list, trip_df)
    trip_acc_ts = trip_df.loc[trip_df["event_type"] == 3]["timestamp"].values
    detected_acc = [1 for lc_ts in trip_acc_ts if lc_ts in all_members_ts]
    detection_rate = sum(detected_acc) / len(trip_acc_ts)
    return detection_rate


def compute_brake_detection_rate(motif_dic_list, trip_df):
    all_members_ts = get_all_members_ts(motif_dic_list, trip_df)
    trip_brake_ts = trip_df.loc[trip_df["event_type"] == 1]["timestamp"].values
    detected_brakes = [1 for lc_ts in trip_brake_ts if lc_ts in all_members_ts]
    detection_rate = sum(detected_brakes) / len(trip_brake_ts)
    return detection_rate


def get_all_members_ts(motif_dic_list, trip_df):
    all_members_ts = set()
    for motif_dic in motif_dic_list:
        for member_pointers in motif_dic["members_ts_pointers"]:
            member_ts = trip_df.iloc[member_pointers]["timestamp"].values
            all_members_ts = all_members_ts.union(set(member_ts))
    return all_members_ts
