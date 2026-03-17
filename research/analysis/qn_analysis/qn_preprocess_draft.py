# I need to produce a file that enables others to :
1) create new timeseries indexes (day, hour, minute index)

2) create a new column "event" that is 1 when the value of chosen sensor
is between a threshold range, and 0 otherwise

3) filter the dataframe to only keep the rows (days and periods) during which an event was detected

4) create a new column "event_id" that gives a unique
id to each period during which an event was detected

- enables team-mates to easily filter the dataframe to only keep the rows (days and periods)
during which an event was detected
- therefore providing whatever the chosen sensor, a dictionnary of said dates and periods

- create a summary dataframe that provides,
for each period during which an event was detected,
the start and end time, the duration, the max concentration,
and the mean concentration during the period

1+2
def preprocess_data(df, captor, low_range, high_range):
    # create time indexes
    df["day_nb"] = df["datetime"].dt.dayofyear
    df["day_nb_ind"] = df["day_nb"]-df["day_nb"].min()+1
    df["hour_ind"] = df["day_nb_ind"]*24 + df["datetime"].dt.hour-24
    df["min_ind"] = df["hour_ind"] * 60 + df["datetime"].dt.minute
    df["sec_ind"] = df["min_ind"] * 60 + df["datetime"].dt.second

    # create event column
    df["event"] = df[captor].apply(lambda x: 1 if low_range <= x <= high_range else 0)

    return df

3
def filter_events(df):
    # filter the dataframe to only keep the rows during which an event was detected
    df_events = df[df["event"] == 1].copy()
    return df_events

4
def create_event_id(df_events, period_length):
    "period_length is the minimum duration (in seconds) for a period to be considered an event"

    # create event_id column
    df_events["time_gap"] = df_events["datetime"].diff()
    period_length = pd.Timedelta(seconds=period_length)
    df_events["event_id"] = (df_events["time_gap"] > period_length).cumsum()+1

    return df_events
