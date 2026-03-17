''' I need to produce a file that enables others to :
1) create new timeseries indexes (day, hour, minute index)

2) create a new column "event" that is 1 when the value of chosen sensor
is between a threshold range, and 0 otherwise

3) filter the dataframe to only keep the rows during which an "event" was detected

4) create a new column "event_id" that gives a unique
id to each period during which an event was detected,
for a given minimum duration of the period (e.g. x seconds)

5)
enables team-mates to easily filter the dataframe to only keep the rows (days and periods)
during which an event was detected
- therefore providing whatever the chosen sensor, a dictionnary of said dates and periods

6) create a summary dataframe that provides,
for each period during which an event was detected,
the start and end time, the duration, the max concentration,
and the mean concentration during the period
'''

import pandas as pd
import numpy as np

#step 1+2
def preprocess_data(df, captor, low_range, high_range):
    # create time indexes
    df["day_nb"] = df["datetime"].dt.dayofyear
    df["day_nb_ind"] = df["day_nb"]-df["day_nb"].min()+1
    df["hour_ind"] = df["day_nb_ind"]*24 + df["datetime"].dt.hour-24
    df["min_ind"] = df["hour_ind"] * 60 + df["datetime"].dt.minute
    df["sec_ind"] = df["min_ind"] * 60 + df["datetime"].dt.second

    # create event column
    df["event"] = ((df[captor] >= low_range) & (df[captor] <= high_range)).astype(int)

    return df

def captor_incr_and_accel(df, captor):
    time_seconds = (df["datetime"] - df["datetime"].iloc[0]).dt.total_seconds()
    df[f"{captor}_incr"] = np.gradient(df[captor], time_seconds)
    df[f"{captor}_accel"] = np.gradient(df[f"{captor}_incr"], time_seconds)
    return df

# step 3
def filter_events(df):
    # filter the dataframe to only keep the rows during which an event was detected
    df_events = df[df["event"] == 1].copy()
    return df_events

# step 4
def create_event_id(df_events, gap_threshold):
    "the minimum time gap between two events to consider them separate"
    # create event_id column
    df_events["time_gap"] = df_events.sort_values("datetime")["datetime"].diff()
    period_length = pd.Timedelta(seconds=gap_threshold)
    df_events["event_id"] = (df_events["time_gap"] > period_length).cumsum()+1
    return df_events

# step 5
def event_days(df_events):
    # get the days during which an event was detected
    event_days = df_events["day_nb_ind"].unique()
    # for each day during which an event was detected, provide event_id
    df_day_to_period_id = (
        df_events.loc[df_events["day_nb_ind"].isin(event_days), ["day_nb_ind", "event_id"]]
        .drop_duplicates()
        .sort_values(["day_nb_ind", "event_id"])
        .reset_index(drop=True)
    )
    return df_day_to_period_id

#step 6
def create_summary(df_events, captor):
    # create a summary dataframe that provides, for each period during which an event was detected,
    # the start and end time, the duration, the max concentration, and the mean concentration during the period
    summary = df_events.groupby("event_id").agg(
        start_time=("datetime", "min"),
        end_time=("datetime", "max"),
        duration=("datetime", lambda x: (x.max() - x.min()).total_seconds()),
        max_captor=(captor, "max"),
        mean_captor=(captor, "mean"),
        num_measurements=("datetime", "count")
    ).reset_index()
    return summary
