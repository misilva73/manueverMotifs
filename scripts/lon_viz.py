import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def add_lon_events_label_to_motif(motif_dic_list, events_ts):
    motif_event_dic_list = []
    for motif_dic in motif_dic_list:
        motif_event_dic = motif_dic
        motif_event_dic['n_brakes'] = 0
        motif_event_dic['n_accelerations'] = 0
        for member_pointers in motif_dic['members_ts_pointers']:
            member_events_list = [events_ts[i] for i in member_pointers]
            motif_event_dic['n_brakes'] += np.sum([member_event == 1 for member_event in member_events_list])
            motif_event_dic['n_accelerations'] += np.sum([member_event == 3 for member_event in member_events_list])
        motif_event_list = [motif_event_dic['n_brakes'], motif_event_dic['n_accelerations']]
        if np.sum(motif_event_list) == 0:
            motif_event_dic['event_label'] = 0
        else:
            motif_event_dic['event_label'] = np.argmax(motif_event_list) + 1
        motif_event_dic_list.append(motif_event_dic)
    return motif_event_dic_list


def plot_k_lon_motifs(k, ts, events_ts, motif_dic_list):
    """
    This function shows the base visualisation for the first k motifs in motif_dic_list for the original 1-d time-series
    :param k: number of motifs to plot
    :param ts: original 1-dimensional time-series
    :param events_ts: list of labels for each entry in ts
    :param motif_dic_list: list of dictionaries, where a dic is related to a single motif
    :return: No return - shows the plots
    """
    for motif_dic in motif_dic_list[0:k]:
        plot_single_lon_motif(ts, events_ts, motif_dic)
        plt.show()


def plot_single_lon_motif(ts, events_ts, motif_dic, y_label='Original time-series'):
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
    lon_event_df = raw_event_df[(raw_event_df['event'] == 1) | (raw_event_df['event'] == 3)]
    # Plots:
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(motif_dic['pattern'])
    # subplot 1
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in member_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=lon_event_df, legend=False,
                    palette=sns.xkcd_palette(['red', 'grass green']))
    plt.ylabel(y_label)
    plt.xlabel('Time')
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 2
    plt.subplot2grid((2, 2), (1, 0))
    for temp_point in member_pointers:
        plt.plot(ts[temp_point], 'xkcd:dark grey')
    plt.xlabel('Time')
    plt.ylabel("Motif's members")
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 3
    plt.subplot2grid((2, 2), (1, 1))
    plt.plot(ts[center_pointers], 'xkcd:dark grey')
    plt.xlabel('Time')
    plt.ylabel("Motif's center")
    plt.ylim(min(ts)-0.01, max(ts)+0.01)
    return fig


def plot_lon_motif_groups(ts, events_ts, motif_dic_list, group_labels):
    motif_event_dic_list = add_lon_events_label_to_motif(motif_dic_list, events_ts)
    group_set = set(group_labels)
    for group in group_set:
        group_motif_dic_list = [dic for i, dic in enumerate(motif_event_dic_list) if group_labels[i] == group]
        plot_single_lon_motif_group(ts, events_ts, group_motif_dic_list, group)
        plt.show()


def plot_single_lon_motif_group(ts, events_ts, group_motif_dic_list, group):
    center_pointers = [dic['center_ts_pointers'] for dic in group_motif_dic_list]
    raw_event_df = pd.DataFrame([ts, events_ts]).T.reset_index()
    raw_event_df.columns = ['index', 'var', 'event']
    lon_event_df = raw_event_df[(raw_event_df['event'] == 1) | (raw_event_df['event'] == 3)]
    # Plots:
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle('Group {} - {} motifs'.format(group, len(group_motif_dic_list)))
    # subplot 1
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in center_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=lon_event_df, legend=False,
                    palette=sns.xkcd_palette(['red', 'grass green']))
    plt.xlabel('Time')
    plt.ylabel('Original time-series')
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 2
    plt.subplot2grid((2, 2), (1, 0))
    for temp_point in center_pointers:
        plt.plot(ts[temp_point], 'xkcd:dark grey')
    plt.xlabel('Time')
    plt.ylabel("Motif's centers")
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 3
    plt.subplot2grid((2, 2), (1, 1))
    brakes_number_list = [dic['n_brakes'] for dic in group_motif_dic_list]
    brake_values, brake_counts = np.unique(brakes_number_list, return_counts=True)
    brakes_df = pd.DataFrame([brake_values, brake_counts], index=['values', 'counts']).T.assign(Event='Brakes')
    acc_number_list = [dic['n_accelerations'] for dic in group_motif_dic_list]
    acc_values, acc_counts = np.unique(acc_number_list, return_counts=True)
    acc_df = pd.DataFrame([acc_values, acc_counts], index=['values', 'counts']).T.assign(Event='Accelerations')
    event_counts_df = pd.concat([brakes_df, acc_df])
    ax = sns.barplot(x="values", y='counts', hue='Event', data=event_counts_df,
                     palette=sns.xkcd_palette(['red', 'grass green']))
    ax.legend_.remove()
    plt.xlabel("Number of turns covered by the motifs")
    plt.ylabel("Motif count")
    return fig


def plot_zoomin_lon_motif(ts, motif_dic, events_ts,  x_lim):
    member_pointers = motif_dic['members_ts_pointers']
    raw_event_df = pd.DataFrame([ts, events_ts]).T.reset_index()
    raw_event_df.columns = ['index', 'var', 'event']
    lon_event_df = raw_event_df[(raw_event_df['event'] == 1) | (raw_event_df['event'] == 3)]
    fig = plt.figure(figsize=(15, 3))
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in member_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=lon_event_df, legend=False,
                    palette=sns.xkcd_palette(['red', 'grass green']))
    plt.ylabel('Original time-series')
    plt.xlabel('Time')
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    plt.xlim(x_lim)
    return fig


def plot_single_lon_motif_for_paper(ts, events_ts, motif_dic):
    member_pointers = motif_dic['members_ts_pointers']
    center_pointers = motif_dic['center_ts_pointers']
    raw_event_df = pd.DataFrame([ts, events_ts]).T.reset_index()
    raw_event_df.columns = ['index', 'var', 'event']
    lon_event_df = raw_event_df[(raw_event_df['event'] == 1) | (raw_event_df['event'] == 3)]
    # Plots:
    fig = plt.figure(figsize=(9, 6))
    # subplot 1
    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    plt.plot(ts, 'xkcd:grey', alpha=0.5)
    for temp_point in member_pointers:
        plt.plot(temp_point, ts[temp_point], 'xkcd:dark grey')
    sns.scatterplot(x="index", y="var", hue="event", data=lon_event_df, legend=False,
                    palette=sns.xkcd_palette(['red', 'grass green']))
    plt.ylabel('')
    plt.xlabel('')
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 2
    plt.subplot2grid((2, 2), (1, 0))
    for temp_point in member_pointers:
        plt.plot(ts[temp_point], 'xkcd:dark grey')
    plt.xlabel("Motif's members")
    plt.ylabel("")
    plt.ylim(min(ts) - 0.01, max(ts) + 0.01)
    # subplot 3
    plt.subplot2grid((2, 2), (1, 1))
    plt.plot(ts[center_pointers], 'xkcd:dark grey')
    plt.xlabel("Motif's center")
    plt.ylabel("")
    plt.ylim(min(ts)-0.01, max(ts)+0.01)
    return fig
