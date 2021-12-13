import time
from typing import Dict
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.fft import rfftfreq, rfft

from calculations.timeseries import filter_timeserie, is_timeserie_froozen


def calculate_fft(data: Dict[str, pd.DataFrame], apply_filter: bool, filter_limits: list):
    """ calculate the Discrete Fast Fourier Transformation from a timeseries """

    fft_plot_data = []
    for df in data.values():
        timestamp = df.index.values[1]
        for i, var_name in enumerate(df.columns.values):
            timeserie = df.loc[:, var_name].values
            pv = df.columns.values[i]

            # ignoring datasets with more than 50% of repeated values (=no value)
            perc_cutoff = 0.5 if 'MARE' not in pv else 0.8
            if is_timeserie_froozen(timeserie, perc_cutoff):
                continue

            # defining sampling properties
            ts1 = time.mktime(pd.to_datetime(df.index.values[0]).timetuple())
            ts2 = time.mktime(pd.to_datetime(df.index.values[1]).timetuple())
            acq_period = ts2 - ts1
            T = acq_period # in seconds
            N = len(timeserie)

            # creating frequency x axis data
            W = rfftfreq(N, T)

            # applying filter if needed
            if (apply_filter):
                timeserie = filter_timeserie(df.loc[:, var_name], *filter_limits, is_series=True)

            # calculating fft
            yr = rfft(timeserie)
            yr = np.abs(yr[1:])**2
            yp = 2/(N/2) * yr
            xf = np.array(W[1:])
            xp = 1/xf/60/60
            # just append all series
            fft_plot_data.append({'var': pv, 'xp': xp, 'yp': yp, 'ts': timestamp})

    return fft_plot_data

def find_fft_properties (fft_serie):
    # gathering data for calculations
    xp = fft_serie['xp']
    yp = fft_serie['yp']
    # calculating fft properties
    peaks, _ = find_peaks(yp)
    prominences = peak_prominences(yp, peaks)[0]
    widths = peak_widths(yp, peaks, rel_height=0.5)[0]
    try:
        y_max_peak = max(yp[peaks])
    except ValueError:
        raise
    x_max_peak = np.where(yp == y_max_peak)[0][0]
    period_max_peak = xp[x_max_peak]
    idx_max_peak = np.where(peaks == x_max_peak)[0][0]
    width_max_peak = widths[idx_max_peak]
    return {'peaks': peaks, 'y_max_peak': y_max_peak, 'period_max_peak': period_max_peak, 'prominences': prominences, 'widths': widths, 'width_max_peak': width_max_peak}
