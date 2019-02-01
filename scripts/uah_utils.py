import os
import pandas as pd
import numpy as np
from scipy import signal as signal


def get_full_point_uah_data(data_path, compute_jerk=False):
    # initialize data variable as list
    data_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        if len(files) < 2:
            continue
        else:
            # import individual trip files
            inertial_df, gps_df, events_df = import_uah_trip_data(root, compute_jerk=compute_jerk)
            # transform and join trip files
            trip_df = transform_uah_trip_data(inertial_df, gps_df, events_df)
            # add ids and labels to trip_df
            trip_df['user_id'] = root.split('/')[3].split('-')[2]
            trip_df['trip_id'] = root.split('/')[3].split('-')[0]
            trip_df['trip_label'] = get_trip_labels(root, len(trip_df.index))
            trip_df['road'] = get_road_type(root)
            # append trip to data_list
            data_list.append(trip_df)
    return pd.concat(data_list, ignore_index=True, sort=False)


def get_full_windowed_uah_data(data_path, window_size, rolling_step):
    data_list = []
    event_labels_list = []
    trip_labels_list = []
    for root, dirs, files in os.walk(data_path, topdown=False):
        if len(files) < 2:
            continue
        else:
            temp_data, temp_event_labels = get_windowed_data_from_uah_trip(root, window_size, rolling_step)
            temp_trip_labels = get_trip_labels(root, len(temp_data))
            data_list.append(temp_data)
            event_labels_list.append(temp_event_labels)
            trip_labels_list.append(temp_trip_labels)
    X = np.concatenate(data_list)
    y_event = np.concatenate(event_labels_list)
    y_trip = np.concatenate(trip_labels_list)
    return X, y_event, y_trip


def get_windowed_data_from_uah_trip(data_path, window_size, rolling_step):
    inertial_df, gps_df, events_df = import_uah_trip_data(data_path)
    clean_df = transform_uah_trip_data(inertial_df, gps_df, events_df)
    windowed_array, events_array = create_rolling_windows(clean_df, window_size, rolling_step)
    return windowed_array, events_array
    

def import_uah_trip_data(root_path, compute_jerk=False):
    # import raw inertial data
    inertial_file_path = os.path.join(root_path, 'RAW_ACCELEROMETERS.txt')
    raw_inertial_df = pd.read_csv(inertial_file_path, sep=" ", header=None).iloc[:, 0:11]
    # add the column names
    raw_inertial_df.columns = ["timestamp", "activated", "raw_ax", "raw_ay",
                               "raw_az", "ax", "ay", "az", "roll", "pitch", "yaw"]
    # change the timestamp to seconds (there's 10 observations per second)
    inertial_df = raw_inertial_df.assign(timestamp = lambda x: np.round(x.timestamp,1))
    # compute the jerk
    if compute_jerk:
        inertial_df['a_intensity'] = np.sqrt(inertial_df['ay']**2 + inertial_df['az']**2)
        inertial_df['jerk'] = (inertial_df['a_intensity'] - inertial_df['a_intensity'].shift(1))/0.1
    # import raw gps data
    gps_file_path = os.path.join(root_path, 'RAW_GPS.txt')
    raw_gps_df = pd.read_csv(gps_file_path, sep=" ", header=None).iloc[:,0:9]
    # add the column names
    raw_gps_df.columns = ["timestamp", "speed", "latitude", "longitude",
                          "altitude", "v_accuracy", "h_accuracy", "course", "difcourse"]
    # change the timestamp to integer second (there's an observation per second)
    gps_df = raw_gps_df.assign(timestamp=lambda x: np.round(x.timestamp, 0))
    # import  inertial events data
    events_file_path = os.path.join(root_path, 'EVENTS_INERTIAL.txt')
    try:
        raw_events_df = pd.read_csv(events_file_path, sep=" ", header=None).iloc[:,0:6]
        # add the column names
        raw_events_df.columns = ["timestamp", "type", "level", "latitude", "longitude", "date"]
        # change the timestamp to seconds (same precision as acceleration data!)
        events_df = raw_events_df.assign(timestamp = lambda x: np.round(x.timestamp,1))
    except:
        events_df = pd.DataFrame(columns=["timestamp", "type", "level", "latitude", "longitude", "date"])
    return inertial_df, gps_df, events_df


def transform_uah_trip_data(inertial_df, gps_df, events_df):
    # get vector with speed observations
    speed_vec = gps_df['speed']
    # apply FIR upsampling to get 10 observations per second
    upsampled_speed_vec = signal.resample_poly(speed_vec , 10, 1)
    upsampled_timestamp_vec = np.concatenate(gps_df['timestamp'].apply(lambda x: np.round(np.arange(x, x+1, 0.1), 1)))
    # build upsampled speed df and exclude 3s
    # of observations in the begginign and end of trip
    upsampled_speed_df = pd.DataFrame({'timestamp': upsampled_timestamp_vec[30:-30],
                                       'speed': upsampled_speed_vec[30:-30]})
    # add upsampled speed to inertial df and drop unnecessary columns
    measurements_df = inertial_df\
        .drop(columns=['activated', 'raw_ax', 'raw_ay', 'raw_az'])\
        .merge(upsampled_speed_df, on='timestamp', how='inner')
    # left-join data from event_df
    joined_df = measurements_df\
        .merge(events_df[["timestamp", "type", "level"]], how='left', on='timestamp') \
        .rename(columns={'type' : 'event_type', 'level' : 'event_level'})
    # concatenate events and levels and drop the two columns
    final_df = joined_df\
        .assign(event_type=lambda x: x['event_type'].apply(lambda y: 0 if np.isnan(y) else y),
                event_level=lambda x: x['event_level'].apply(lambda y: 0 if np.isnan(y) else y))\
        .assign(event=lambda x: x['event_type'].astype(int).astype(str) + x['event_level'].astype(int).astype(str))
    return final_df


def create_rolling_windows(df, window_size, rolling_step):
    # create empty lists to add the windows and events
    windows_lists = []
    events_list = []
    # create the windows and add to windows_lists
    # create the event for each window and save in evens_list
    for win_start in range(0, len(df), rolling_step):
        events_set = set(df.iloc[win_start:win_start+window_size, 10])
        if len(events_set)>1:
            events_set = events_set.difference({'00'})
        for event in events_set:
            windows_lists.append(np.array(df.iloc[win_start:win_start+window_size, 1:8]))
            events_list.append(event)
    # exclude the windows that are shorter than window_size (usually the last ones)
    events_list = [events_list[i] for i in range(len(events_list)) if len(windows_lists[i]) == window_size]
    windows_lists = [window for window in windows_lists if len(window) == window_size]
    # convert lists to an np.arrays
    windows_array = np.asarray(windows_lists)
    events_array = np.asarray(events_list)
    return windows_array, events_array


def get_trip_labels(file_path, array_size):
    if "NORMAL" in file_path:
        label = 1
    elif "AGGRESSIVE" in file_path:
        label = 2
    elif "DROWSY" in file_path:
        label = 0
    else:
        label = None
    return np.repeat(label, array_size)


def get_road_type(file_path):
    if "SECONDARY" in file_path:
        return 'secondary'
    elif "MOTORWAY" in file_path:
        return 'motorway'
    else:
        return None
    

def undersample_uah_observations(X, y, y_trip, sampling_rate=0.05, no_events_label='00'):
    # Get arrays with events
    X_with_events = X[~(y == no_events_label)]
    y_with_events = y[~(y == no_events_label)]
    y_trip_with_events = y_trip[~(y == no_events_label)]
    # Get arrays without events, i.e., label == 00
    X_without_events = X[y == no_events_label]
    y_without_events = y[y == no_events_label]
    y_trip_without_events = y_trip[y == no_events_label]
    # Undersample arrays without events and concatenate with arrays with events
    n = X_without_events.shape[0]
    idx = np.random.choice(n, int(n*sampling_rate), replace=False)
    X_under = np.concatenate((X_without_events[idx], X_with_events))
    y_under = np.concatenate((y_without_events[idx], y_with_events))
    y_trip_under = np.concatenate((y_trip_without_events[idx], y_trip_with_events))
    return X_under, y_under, y_trip_under
