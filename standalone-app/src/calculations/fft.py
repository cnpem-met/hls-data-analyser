from datetime import datetime
import time
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, peak_widths, spectrogram
from scipy.fft import rfftfreq, rfft


""" --------------------------------------------------------------------------------------------------------------------
    Desc.: calculate the Discrete Fast Fourier Transformation from a timeseries
    Args: 
        data_list: tupple containing timeserie data
        pv: the name of the pv that holds the data
    Output: dict containing fft and filtered time data
        -------------------------------------------------------------------------------------------------------------------- """
def calculate_fft(data_df_list):
    fft_plot_data = []
    for data in data_df_list:
        timestamp = data.index.values[1]
        for i, var_name in enumerate(data.columns.values):
            timeserie = data.loc[:, var_name].values
            pv = data.columns.values[i]
            # ignoring datasets with more than 50% of repeated values (=no value)
            perc_cutoff = 0.5 if 'MARE' not in pv else 0.8
            if self.is_timeserie_froozen(timeserie, perc_cutoff):
                print('cut')
                continue
            # defining sampling properties
            ts1 = time.mktime(datetime.strptime(data.index.values[0], "%d.%m.%y %H:%M").timetuple())
            ts2 = time.mktime(datetime.strptime(data.index.values[1], "%d.%m.%y %H:%M").timetuple())
            acq_period = ts2 - ts1
            T = acq_period # in seconds
            N = len(timeserie)
            # creating frequency x axis data
            W = rfftfreq(N, T)
            # applying filter if needed
            if (self.ui.check_applyFilter.isChecked()):
                timeserie = self.filter_timeserie(data.loc[:, var_name], is_series=True)
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
