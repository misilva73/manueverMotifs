import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns


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
