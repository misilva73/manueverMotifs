import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_k_lon_motifs(k, trip_df, motif_dic_list, add_suptitle=True):
    """
    This function shows the base visualisation for the first k motifs in motif_dic_list

    :param k: number of motifs to plot
    :param trip_df: pandas dataframe with trip measurements and event labels
    :param motif_dic_list: list of dictionaries, where a dic is related to a single motif
    :param add_suptitle: bool indicating whether to add a suptitle with the motif's pattern
    :return: No return - shows the plots
    """
    for motif_dic in motif_dic_list[0:k]:
        plot_single_motif(trip_df, motif_dic, add_suptitle)
        plt.show()


def plot_single_motif(trip_df, motif_dic, add_suptitle):
    """
    Base visualization of a single motif

    :param trip_df: pandas dataframe with trip measurements and event labels
    :param motif_dic: dictionary of the motif to plot is related to a single motif
    :param add_suptitle: bool indicating whether to add a suptitle with the motif's pattern
    :return: matplotlib figure with the final visualization
    """
    member_pointers = motif_dic["members_ts_pointers"]
    center_pointers = motif_dic["center_ts_pointers"]
    # Plot
    fig = plt.figure(figsize=(12, 3))
    if add_suptitle:
        plt.suptitle(motif_dic["pattern"], y=1.05)
    # subplot 1
    plt.subplot2grid((1, 5), (0, 0), colspan=4)
    plt.plot(trip_df["timestamp"], trip_df["az"], "xkcd:grey", alpha=0.5)
    for temp_point in member_pointers:
        temp_member_df = trip_df.iloc[temp_point]
        plt.plot(temp_member_df["timestamp"], temp_member_df["az"], "xkcd:dark grey")
    sns.scatterplot(
        x="timestamp",
        y="az",
        hue="event_type",
        data=trip_df.loc[(trip_df["event_type"] == 1) | (trip_df["event_type"] == 3)],
        legend=False,
        palette=sns.xkcd_palette(["red", "grass green"]),
    )
    plt.title("Position of motif's members and center on full trip")
    plt.ylabel("Longitudinal acceleration (Gs)")
    plt.xlabel("Time (seconds)")
    plt.ylim(trip_df["az"].min() - 0.01, trip_df["az"].max() + 0.01)
    # subplot 2
    ax = plt.subplot2grid((1, 5), (0, 4))
    for temp_point in member_pointers:
        temp_member_df = trip_df.iloc[temp_point]
        temp_member_df["timestamp"] = (
            temp_member_df["timestamp"] - temp_member_df["timestamp"].min()
        )
        plt.plot(temp_member_df["timestamp"], temp_member_df["az"], "xkcd:dark grey")
    center_df = trip_df.iloc[center_pointers]
    center_df["timestamp"] = center_df["timestamp"] - center_df["timestamp"].min()
    plt.plot(center_df["timestamp"], center_df["az"], "xkcd:azure")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Longitudinal acceleration (Gs)")
    plt.title("Members & center")
    plt.ylim(trip_df["az"].min() - 0.01, trip_df["az"].max() + 0.01)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return fig


def plot_all_clusters(trip_df, motif_dic_list, cluster_labels):
    for cluster_id in set(cluster_labels):
        cluster_motif_dic_list = [
            motif_dic
            for i, motif_dic in enumerate(motif_dic_list)
            if cluster_labels[i] == cluster_id
        ]
        plot_cluster_centers(trip_df, cluster_motif_dic_list, cluster_id)
        plt.show()


def plot_cluster_centers(trip_df, cluster_motif_dic_list, cluster_id):
    cluster_center_pointers = [
        dic["center_ts_pointers"] for dic in cluster_motif_dic_list
    ]
    # Plot
    fig = plt.figure(figsize=(12, 3))
    plt.suptitle(cluster_id, y=1.05)
    # subplot 1
    plt.subplot2grid((1, 5), (0, 0), colspan=4)
    plt.plot(trip_df["timestamp"], trip_df["az"], "xkcd:grey", alpha=0.5)
    for temp_point in cluster_center_pointers:
        temp_member_df = trip_df.iloc[temp_point]
        plt.plot(temp_member_df["timestamp"], temp_member_df["az"], "xkcd:dark grey")
    sns.scatterplot(
        x="timestamp",
        y="az",
        hue="event_type",
        data=trip_df.loc[(trip_df["event_type"] == 1) | (trip_df["event_type"] == 3)],
        legend=False,
        palette=sns.xkcd_palette(["red", "grass green"]),
    )
    plt.title("Position of motif's centers belonging to the cluster on full trip")
    plt.ylabel("Longitudinal acceleration (Gs)")
    plt.xlabel("Time (seconds)")
    plt.ylim(trip_df["az"].min() - 0.01, trip_df["az"].max() + 0.01)
    # subplot 2
    ax = plt.subplot2grid((1, 5), (0, 4))
    for temp_point in cluster_center_pointers:
        temp_member_df = trip_df.iloc[temp_point]
        temp_member_df["timestamp"] = (
            temp_member_df["timestamp"] - temp_member_df["timestamp"].min()
        )
        plt.plot(temp_member_df["timestamp"], temp_member_df["az"], "xkcd:dark grey")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Longitudinal acceleration (Gs)")
    plt.title("Cluster members")
    plt.ylim(trip_df["az"].min() - 0.01, trip_df["az"].max() + 0.01)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    return fig


def plot_zoomin_motif(trip_df, motif_dic, x_lim):
    member_pointers = motif_dic["members_ts_pointers"]
    # Plot
    fig = plt.figure(figsize=(12, 3))
    plt.plot(trip_df["timestamp"], trip_df["az"], "xkcd:grey", alpha=0.5)
    for temp_point in member_pointers:
        temp_member_df = trip_df.iloc[temp_point]
        plt.plot(temp_member_df["timestamp"], temp_member_df["az"], "xkcd:dark grey")
    sns.scatterplot(
        x="timestamp",
        y="az",
        hue="event_type",
        data=trip_df.loc[(trip_df["event_type"] == 1) | (trip_df["event_type"] == 3)],
        legend=False,
        palette=sns.xkcd_palette(["red", "grass green"]),
    )
    plt.ylabel("Longitudinal acceleration (Gs)")
    plt.xlabel("Time (seconds)")
    plt.ylim(trip_df["az"].min() - 0.01, trip_df["az"].max() + 0.01)
    plt.xlim(x_lim)
    return fig
