import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


def plot_single_motif(ts, events_ts, motif_dic, y_label='Full 1-d time-series'):
    """
    This function creates the base visualization for a single motif:
     1) plot with the whole time-series highlighting the labels and the position of each motif's member
     2) plot with all the motif's members
     3) plot with the motif's center
    :param ts: original 1-dimensional time-series
    :param events_ts: list of labels for each entry in ts
    :param motif_dic: dictionary related to the motif
    :param y_label: label to add in the y-axis of the first plot (optional and defaults to 'Full 1-d time-series')
    :return: fig - figure with the motif plot
    """
    member_pointers = motif_dic['members_ts_pointers']
    center_pointers = motif_dic['center_ts_pointers']
    raw_event_df = pd.DataFrame([ts, events_ts]).T.reset_index()
    raw_event_df.columns = ['index', 'var', 'event']
    event_df = raw_event_df[raw_event_df['event'] > 0]
    # Plots:
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(motif_dic['pattern'])
    # subplot 1
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in member_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=event_df, legend=False,
                    palette=sns.xkcd_palette(['red', 'tangerine', 'grass green']))
    plt.ylabel(y_label)
    plt.xlabel('')
    plt.ylim(min(ts), max(ts))
    # subplot 2
    plt.subplot2grid((2, 2), (1, 0))
    for temp_point in member_pointers:
        plt.plot(ts[temp_point], 'xkcd:dark grey')
    plt.ylabel("Motif's members")
    plt.ylim(min(ts), max(ts))
    # subplot 3
    plt.subplot2grid((2, 2), (1, 1))
    plt.plot(ts[center_pointers], 'xkcd:dark grey')
    plt.ylabel("Motif's center")
    plt.ylim(min(ts), max(ts))
    return fig


def plot_k_motifs(k, ts, events_ts, motif_dic_list):
    """
    This function shows the base visualisation for the first k motifs in motif_dic_list for the original 1-d time-series
    :param k: number of motifs to plot
    :param ts: original 1-dimensional time-series
    :param events_ts: list of labels for each entry in ts
    :param motif_dic_list: list of dictionaries, where a dic is related to a single motif
    :param yaxis_label:
    :return: No return - shows the plots
    """
    for motif_dic in motif_dic_list[0:k]:
        plot_single_motif(ts, events_ts, motif_dic)
        plt.show()



def plot_motif_clusters(ts, events_ts, motif_event_dic_list, cluster_labels):
    cluster_set = set(cluster_labels)
    for cluster in cluster_set:
        cluster_motif_dic_list = [dic for i, dic in enumerate(motif_event_dic_list) if cluster_labels[i] == cluster]
        plot_single_motif_cluster(ts, events_ts, cluster_motif_dic_list, cluster)
        plt.show()


def plot_single_motif_cluster(ts, events_ts, cluster_motif_dic_list, cluster):
    center_pointers = [dic['center_ts_pointers'] for dic in cluster_motif_dic_list]
    event_df = pd.DataFrame([ts, events_ts]).T.reset_index()
    event_df.columns = ['index', 'var', 'event']
    motif_event_df = pd.DataFrame([dic['event_label'] for dic in cluster_motif_dic_list], columns=['event_label'])
    # Plots:
    fig = plt.figure(figsize=(15, 6))
    plt.suptitle('Cluster {} - {} motifs'.format(cluster, len(cluster_motif_dic_list)))
    # subplot 1
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in center_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=event_df[event_df['event'] > 0], legend=False,
                    palette=sns.xkcd_palette(['red', 'tangerine', 'grass green']))
    plt.ylabel('Original time-series')
    plt.xlabel('')
    plt.ylim(min(ts), max(ts))
    # subplot 2
    plt.subplot2grid((2, 2), (1, 0))
    for temp_point in center_pointers:
        plt.plot(ts[temp_point], 'xkcd:dark grey')
    plt.ylabel("Motif's centers")
    plt.ylim(min(ts), max(ts))
    # subplot 3
    plt.subplot2grid((2, 2), (1, 1))
    sns.countplot(x="event_label", data=motif_event_df,
                  palette=sns.xkcd_palette(['navy blue', 'red', 'tangerine', 'grass green']))
    plt.xlabel("Motif's label")
    return fig


def plot_label_distribution(cluster_list, y_main, y_sec=None):
    # get unique cluster labels
    cluster_set = set(cluster_list)
    k = len(cluster_set)
    # define grid configs for the plots
    rows = math.ceil(k/5)
    if rows == 1:
        cols = k
    else:
        cols = 5
    # plot main label distribution
    i = 1
    plt.figure(figsize=(17, 3.5*rows))
    plt.suptitle('Main label distribution')
    for cluster in cluster_set:
        temp_list = y_main[cluster_list == cluster]
        values, counts = np.unique(temp_list, return_counts=True)
        plt.subplot(rows, cols, i)
        plt.bar(values, counts)
        plt.title('Cluster {} - {} motifs'.format(cluster, len(temp_list)))
        i += 1
    plt.show()
    # plot secondary label distribution
    if y_sec is not None:
        i = 1
        plt.figure(figsize=(17, 4*rows))
        plt.suptitle('Secondary label distribution')
        for cluster in cluster_set:
            temp_list = y_sec[cluster_list == cluster]
            values, counts = np.unique(temp_list, return_counts=True)
            plt.subplot(rows, cols, i)
            plt.bar(values, counts)
            plt.title('Cluster {} - {} motifs'.format(cluster, len(temp_list)))
            i += 1
        plt.show()
