import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def filter_timeserie(ts_data_df, is_series=True):
    # defining sampling properties
    ts1 = time.mktime(datetime.strptime(ts_data_df.index.values[0], "%d.%m.%y %H:%M").timetuple())
    ts2 = time.mktime(datetime.strptime(ts_data_df.index.values[1], "%d.%m.%y %H:%M").timetuple())
    acq_period = ts2 - ts1
    T = acq_period # in seconds

    max_period_filt, min_period_filt = self.ui.spin_filter_max.value(), self.ui.spin_filter_min.value()
    if (min_period_filt != 0 and max_period_filt != 0):
        filter_type = 'bandpass'
        filter_limit = [1/(3600*max_period_filt), 1/(3600*min_period_filt)]
    elif (min_period_filt != 0):
        filter_type = 'lowpass'
        filter_limit = 1/(3600*min_period_filt)
    else:
        filter_type = 'highpass'
        filter_limit = 1/(3600*max_period_filt)
    b, a = butter(4, filter_limit, filter_type, fs=1/T)

    if (not is_series):
        filtered_data = []
        for var in ts_data_df.columns.values:
            # timeserie = np.array(ts_data_df.loc[:, var].values)
            timeserie = ts_data_df.loc[:, var].values
            filtered_data.append(filtfilt(b,a, timeserie))
    else:
        timeserie = ts_data_df.iloc[:].values.T
        filtered_data = filtfilt(b,a, timeserie)

    return filtered_data

def filter_timeseries_in_df(data: pd.DataFrame, min_period_filt: float, max_period_filt: float):

    filtered_data = data.copy()

    ts1 = time.mktime(datetime.strptime(filtered_data.index.values[0], "%d.%m.%y %H:%M").timetuple())
    ts2 = time.mktime(datetime.strptime(filtered_data.index.values[1], "%d.%m.%y %H:%M").timetuple())
    acq_period = ts2 - ts1
    T = acq_period # in seconds

    if (min_period_filt != 0 and max_period_filt != 0):
        filter_type = 'bandpass'
        filter_limit = [1/(3600*max_period_filt), 1/(3600*min_period_filt)]
    elif (min_period_filt != 0):
        filter_type = 'lowpass'
        filter_limit = 1/(3600*min_period_filt)
    else:
        filter_type = 'highpass'
        filter_limit = 1/(3600*max_period_filt)
    b, a = butter(4, filter_limit, filter_type, fs=1/T)

    for var in filtered_data.columns:
        filtered_data.loc[:,var] = filtfilt(b,a, filtered_data.loc[:,var].values)
    
    return filtered_data

def is_timeserie_froozen (timeserie, limit_percentage = 0.5):
    is_froozen = False
    _, counts_elem = np.unique(timeserie, return_counts=True)
    repeated_elem = 0
    for count in counts_elem:
        if (count > 1):
            repeated_elem += count - 1
    percentage_repeated = repeated_elem/len(timeserie)
    print(percentage_repeated)
    if (percentage_repeated > limit_percentage):
        is_froozen = True
    return is_froozen

def generate_dynamic_timeseries_data ():
    # trimming original df to get sequencial periods of x hours (1h40, 2h etc.)
    output = {}
    # for pv, data in zip(self.loaded_pvs, self.data):
    for data in self.data:
        # number of periods considering sample period
        ts1 = time.mktime(datetime.strptime(data.index.values[0], "%d.%m.%y %H:%M").timetuple())
        ts2 = time.mktime(datetime.strptime(data.index.values[1], "%d.%m.%y %H:%M").timetuple())
        sample_period = int((ts2 - ts1)/60) # in minutes
        chunck_time = self.ui.spin_chunck_day.value() * 24 * 60 + self.ui.spin_chunck_hour.value() * 60 + self.ui.spin_chunck_min.value() # in minutes
        num_period_in_records = math.ceil(chunck_time/sample_period) 
        # total number of records
        num_records = len(data.iloc[:,0])
        # number of chuncks to trim data
        total_periods = int(num_records/num_period_in_records)
        # trim data for each of the columns
        for col_idx, pv in enumerate(data.columns):
            trimmed_data = []
            for i in range(total_periods):
                trimmed_data.append(data.iloc[i*num_period_in_records : (i+1)*num_period_in_records, col_idx])
            output[pv] = trimmed_data
    
    return output