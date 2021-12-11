# %%

# https://stackoverflow.com/questions/61842432/pyqt5-and-asyncio

import time
import asyncio
from functools import partial
import sys
from datetime import datetime
import math
import os
import aiohttp
from itertools import combinations
import json

from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtGui import (QColor)
from matplotlib.pyplot import jet
from scipy.signal.windows.windows import blackman
from ui import Ui_MainWindow

import qasync
from qasync import asyncSlot, QApplication

import numpy as np
import pandas as pd
from pandas import DataFrame

from scipy.optimize import minimize
from scipy.fft import rfftfreq, rfft
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, peak_widths, spectrogram
from scipy.interpolate import make_interp_spline, interp1d
from scipy.stats import pearsonr, spearmanr

from matplotlib.pylab import plot, figure, savefig, show
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.offsetbox as offsetbox

from ui.handler import Ui_handler


# ---- packages above are called inside the calling functions to optimize app's startup time -----
# import seaborn as sns
# from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


"""
IDEIAS P/ IMPLEMENTAR:
    - plotar a serie temporal pra cada data que tem uma correlação calculada, para fins de confirmação de que os dados estão integros
    - tirar manualmente pontos no grafico de correlação (correlações em dias com dados não integros)

"""



class App(QWidget):
    _SESSION_TIMEOUT = 1.0
    RFFREQ_PV = 'RF-Gen:GeneralFreq-RB'
    WELLPRESSURE_PV = 'INF:POC01:Pressure_mca'
    EARTHTIDES_PVS_LIST = ['LNLS:MARE:NORTH', 'LNLS:MARE:EAST', 'LNLS:MARE:UP']
    # HLS_OPOSITE_PVS_LIST = ['TU-17C:SS-HLS-Ax04NE5:Level-Mon', 'TU-06C:SS-HLS-Ax33SW5:Level-Mon',\
    #                         'TU-11C:SS-HLS-Ax48NW5:Level-Mon', 'TU-01C:SS-HLS-Ax18SE5:Level-Mon']
    HLS_OPOSITE_PVS_LIST = ['TU-17C:SS-HLS-Ax04NE5:Level-Mon', 'TU-06C:SS-HLS-Ax33SW5:Level-Mon',\
                            'TU-11C:SS-HLS-Ax48NW5:Level-Mon', 'TU-03C:SS-HLS-Ax24SW1:Level-Mon']

    ARCHIVER_URL = 'http://10.0.38.42/retrieval/data/getData.json'
    HLS_LEGEND = [17, 16, 15, 14, 13, 1.5, 1.2, 20, 19, 18, 6.8, 6.2, 5, 4, 3, 11.5, 11.2, 10, 9, 8]
    
    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: class function called when create an instance
        -------------------------------------------------------------------------------------------------------------------- """
    def __init__(self, app):
        super().__init__()
        self.mainWindow = QMainWindow()
        self.mainWindow.show()

        # initializing ui components
        mainwindow_widget = Ui_MainWindow()
        mainwindow_widget.setupUi(self.mainWindow)
        self.ui = Ui_handler(mainwindow_widget, self)

        # app state variables
        self.app = app
        self.data = []
        self.loaded_pvs = []

        # constants
        self.hls_pvs = self.generate_all_sensors_list()


    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: create a video file from images in directory defined in ui's textbox
        -------------------------------------------------------------------------------------------------------------------- """
    def make_video(self):
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        image_folder='./output/' + self.ui.inputTxt_dirFig.text()
        fps_movie=12
        image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
        image_files.sort(key=lambda i: int(i.split('/')[-1].split('-')[-1].split('.')[0]))
        clip = ImageSequenceClip(image_files, fps=fps_movie)
        image_folder+'/'+image_folder.split('/')[-1]+'.mp4'
        clip.write_videofile(image_folder+'/'+image_folder.split('/')[-1]+'.mp4')


    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: Fetch data from Archiver
    Args:
        session: aiohttp client object
        pv: the name of PV from which data will be pulled
        time_from: init of timespam
        time_to: end of timespam
        is_optimized: flag to indicate to Archiver to shrink or not the data by applying a mean
        mean_minutes: time range in which the mean will be applied is case of optimization
    Output: raw data translated from json returned by Archiver
        -------------------------------------------------------------------------------------------------------------------- """
    @asyncSlot()
    async def fetch(self, session, pv, time_from, time_to, is_optimized, mean_minutes):
        if is_optimized:
            pv_query = f'mean_{int(60*mean_minutes)}({pv})'
        else:
            pv_query = pv
        query = {'pv': pv_query, 'from': time_from, 'to': time_to}
        async with session.get(self.ARCHIVER_URL, params=query) as response:
            response_as_json = await response.json()
            return response_as_json


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: mid function that fetches data from multiple pvs
    Args:
        pvs: list of pvs
        time_from: init of timespam
        time_to: end of timespam
        is_optimized: flag to indicate to Archiver to shrink or not the data by applying a mean
        mean_minutes: time range in which the mean will be applied is case of optimization
    Output: all raw data collected from Archiver
        -------------------------------------------------------------------------------------------------------------------- """
    @asyncSlot()
    async def retrieve_data(self, pvs, time_from, time_to, isOptimized=False, mean_minutes=0):
        async with aiohttp.ClientSession() as session:
            data = await asyncio.gather(*[self.fetch(session, pv, time_from, time_to, isOptimized, mean_minutes) for pv in pvs])
            return data


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: utility function to create a list with all the current HLS PVs names
    Output: list with all HLS PVs
        -------------------------------------------------------------------------------------------------------------------- """
    def generate_all_sensors_list (self):
        sectors = [17, 16, 15, 14, 13, 1, 1, 20, 19, 18, 6, 6, 5, 4, 3, 11, 11, 10, 9, 8]
        axis = [4, 1, 59, 57, 54, 18, 16, 14, 12, 9, 33, 31, 29, 27, 24, 48, 46, 44, 42, 39]
        quadrant = ['NE5', 'NE4', 'NE3', 'NE2', 'NE1', 'SE5', 'SE4', 'SE3', 'SE2', 'SE1', 'SW5', 'SW4', 'SW3', 'SW2', 'SW1', 'NW5', 'NW4', 'NW3', 'NW2', 'NW1']
        sensors_list = []
        for sector, ax, quad in zip(sectors, axis, quadrant):
            sensors_list.append(f'TU-{sector:02d}C:SS-HLS-Ax{ax:02d}{quad}:Level-Mon')
        return sensors_list


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: function to calculate the 1D or 2D least square between points
    Args: 
        params: dynamic variables that varies between calls
        *args: static variables
    Output: current least square value
        -------------------------------------------------------------------------------------------------------------------- """
    def bestfit(self, params, *args):
        x0 = np.array(args[0])
        x_ref = np.array(args[1])
        type_of_bf = args[2] or '1d'

        # inicializando variável para cálculo do(s) valor a ser minimizado
        diff = []

        if (type_of_bf == '1d'):
            Ty = params[0]
            for i in range(len(x0)):
                xt = x0[i] + Ty
                diff.append(((x_ref[i]-xt)**2).sum())
        elif (type_of_bf == '2d'):
            Tx = params[0]
            Ty = params[1] 
            for i in range(len(x0[0])):
                # print('x0: ', x0[0][i])
                # print('x_ref: ', x_ref[0][i])
                xt = x0[0][i] + Tx
                yt = x0[1][i] + Ty
                # print('xt: ', xt)
                diff.append(((x_ref[0][i]-xt)**2).sum())
                diff.append(((x_ref[1][i]-yt)**2).sum())
        return np.sqrt(np.sum(diff))


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: calculates offset by the means of minimizing the least square between the 2 curves
    Args:
        pts: list of coordinates from curve 1
        pts_ref: list of coordinates from curve 2 that is intended to stay static (reference)
        type_of_bf: specification of the degrees of freedom
            values: '1d' or '2d'
    Output: the offset to be used to minimize the distances between the curves 
        -------------------------------------------------------------------------------------------------------------------- """
    def calc_offset(self, pts, pts_ref, type_of_bf='1d'):
        # inicializando array com parâmetros a serem manipulados durante as iterações da minimização
        params = [0] if type_of_bf == '1d' else [1,0]

        # aplicando a operação de minimização para achar os parâmetros de transformação
        offset = minimize(fun=self.bestfit, x0=params, args=(pts, pts_ref, type_of_bf),method='SLSQP', options={'ftol': 1e-10, 'disp': False})['x']

        return offset


    def generate_HLS_data_to_dynamic_time_plot(self):
        data_df = self.data[0].copy()
        data_df.columns = data_df.columns.map(float)

        # filtering slow (and big) movements
        ts1 = time.mktime(datetime.strptime(data_df.index.values[0], "%d.%m.%y %H:%M").timetuple())
        ts2 = time.mktime(datetime.strptime(data_df.index.values[1], "%d.%m.%y %H:%M").timetuple())
        acq_period = ts2 - ts1
        T = acq_period # in seconds
        if (self.ui.check_applyFilter.isChecked()):
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
            for sensor in data_df.columns:
                data_df.loc[:,sensor] = filtfilt(b,a, data_df.loc[:,sensor].values)

        # referencing in one sensor
        sens_ref = 19
        data_df = data_df.sub(data_df.loc[:,sens_ref], axis=0)

        # --------- TEMPORARIO -------------
        # tirando sensores que estão com aparente problema de leitura
        data_df.drop(columns=[9,14], inplace=True)
        # mapeando sensores do setor do anel para eixos do prédio
        # data_df.columns = [4,1,59,54,18,16,14,12,9,33,31,29,27,24,48,46,44,39]
        # data_df.columns = [4,1,59,57,54,18,16,14,12,9,33,31,29,27,24,48,46,44,42,39]

        # sorting sensors
        data_df = data_df.sort_index(axis=1)

        # calculating bestfit between 2 epochs and applying shift in curve
        for i in range(1, len(data_df.iloc[:,0])):
            offset = self.calc_offset(data_df.iloc[i,:].values, data_df.iloc[i-1,:].values)
            data_df.iloc[i,:] += offset[0]

        # setting first record as time reference
        data_df = data_df - data_df.iloc[0,:]

        return data_df


    def filter_timeserie(self, ts_data_df, is_series=True):
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
            print(timeserie)
            print(f'linhas: {len(timeserie)},  colunas: {len(timeserie[0])}')
            filtered_data = filtfilt(b,a, timeserie)

        return filtered_data


    def find_fft_properties (self, fft_serie):
        # gathering data for calculations and exception raising
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

    def filter_serie_by_fft_properties (self, fft_data, serie_info):
        pv = serie_info['pv']
        # finding the peaks of fft and its properties
        try:
            fft_props = self.find_fft_properties(fft_data)
        except ValueError:
            raise
        # testing conditions that will indicate if the fft serie is valid or not
        # obs: values are based on empirical observations
        if (pv == 'RF-Gen:GeneralFreq-RB'):
            if (fft_props['y_max_peak'] > 5000 and fft_props['y_max_peak'] < 300000 and fft_props['width_max_peak'] < 4):
                return True
        elif (pv == 'HLS:C4_S2_LEVEL'):
            if (fft_props['y_max_peak'] > 1e-3 and fft_props['width_max_peak'] < 3):
                return True
        else:
            print(f"{pv} not configured for selecting fft series from its properties")
            return True
        # if program reachs here, conditions were not satisfied
        return False

    def is_timeserie_froozen (self, timeserie, limit_percentage = 0.5):
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

    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: calculate the Discrete Fast Fourier Transformation from a timeseries
    Args: 
        data_list: tupple containing timeserie data
        pv: the name of the pv that holds the data
    Output: dict containing fft and filtered time data
        -------------------------------------------------------------------------------------------------------------------- """
    def calculate_fft(self, data_df_list):
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
                if (False):
                    # filtering curves by the means of the amplitude of max peak and its width
                    try:
                        appendSerie = self.filter_serie_by_fft_properties({'yp': yp, 'xp': xp}, {'ts': timestamp, 'pv': pv})
                    except ValueError:
                        continue

                    if appendSerie:
                        fft_plot_data.append({'xp': xp, 'yp': yp, 'ts': timestamp})
                else:
                    # just append all series
                    fft_plot_data.append({'var': pv, 'xp': xp, 'yp': yp, 'ts': timestamp})

        return fft_plot_data


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: plot static fft data
    Args: 
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_fft_static(self):
        fig, ax = plt.subplots()
        fft_data = self.calculate_fft(self.data)
        for data in fft_data:
            ax.plot(data['xp'], data['yp'], label=data['var'])
        ax.legend()
        show()


    def plot_cross_correl (self):
        # creating sliced (based on the input of 'time chuncks') dfs
        sliced_data_df = self.generate_dynamic_timeseries_data()
        # print(sliced_data_df)
        # creating timeserie lists from sliced dfs
        sliced_data = []
        for pv in sliced_data_df:
            sliced_data.append({'pv': pv,\
                                'val': {'ts': [datetime.strptime(df.index[0], "%d.%m.%y %H:%M") for df in sliced_data_df[pv]],\
                                        'serie': np.array([df.to_numpy().reshape(1,df.size)[0] for df in sliced_data_df[pv]])}})

        # bin_dates = np.array([datetime.strptime(df.index[0], "%d.%m.%y %H:%M") for df in sliced_data_df[self.loaded_pvs[0]]])

        # checking all possible combinations
        comb = list(combinations(np.arange(0,len(sliced_data)), 2))

        # print(f'bin_dates -> len: {len(bin_dates)}\n')

        cross_corr_all = []
        # cross-correlation calculation upon time series
        for comb_idx in comb:
            cross_corr = []
            ts = []
            # print(f'pv: {sliced_data[comb_idx[0]]["pv"]} -> len: {len(sliced_data[comb_idx[0]]["val"])}')
            # print(f'pv: {sliced_data[comb_idx[1]]["pv"]} -> len: {len(sliced_data[comb_idx[1]]["val"])}\n')
            if (len(sliced_data[comb_idx[0]]['val']['serie']) != len(sliced_data[comb_idx[1]]['val']['serie'])):
                self.logMessage(f'Not evaluating correlation between {sliced_data[comb_idx[0]]["pv"]} and {sliced_data[comb_idx[1]]["pv"]}: divergent array lenghts ({len(sliced_data[comb_idx[0]]["val"]["serie"])} and {len(sliced_data[comb_idx[1]]["val"]["serie"])})', 'danger')
                continue

            if (sliced_data[comb_idx[0]]['val']['ts'] != sliced_data[comb_idx[1]]['val']['ts']):
                self.logMessage(f'Not evaluating correlation between {sliced_data[comb_idx[0]]["pv"]} and {sliced_data[comb_idx[1]]["pv"]}: datetimes not coincident', 'danger')
                continue

            for s1, s2 in zip(sliced_data[comb_idx[0]]['val']['serie'], sliced_data[comb_idx[1]]['val']['serie']):
                # earth tide pvs needs a bigger repeated percentaged cutoff value
                perc_cutoff_s1 = 0.5 if 'MARE' not in sliced_data[comb_idx[0]]['pv'] else 0.7
                perc_cutoff_s2 = 0.5 if 'MARE' not in sliced_data[comb_idx[1]]['pv'] else 0.7
                # avoiding series that hasn't enough data to be correlated
                if self.is_timeserie_froozen(s1, perc_cutoff_s1) or self.is_timeserie_froozen(s2, perc_cutoff_s2):
                    cross_corr.append(None)
                    continue
                # calculating time-based normalized cross-correlation
                s0 = (s1 - np.mean(s1))/(np.std(s1)*len(s1))
                s1 = (s2 - np.mean(s2))/(np.std(s2))
                corr = np.correlate(s0, s1, mode='full')
                # storing only the maximum coeficient - when series are in phase
                cross_corr.append(max(corr))

            ts = sliced_data[comb_idx[0]]['val']['ts']
            cross_corr_all.append({'label': f'{sliced_data[comb_idx[0]]["pv"]} x {sliced_data[comb_idx[1]]["pv"]}', 'val': cross_corr, 'ts': ts})
        
        # plotting bars containing time and frequency correlation information

        plot = Plot()
        fig, ax = plot.get_plot_props()

        lines, pts = [], []
        for cross_corr in cross_corr_all:
            line, = ax.plot(cross_corr['ts'], cross_corr['val'], alpha=0.3, label=cross_corr['label'])
            pt = ax.scatter(cross_corr['ts'], cross_corr['val'])
            lines.append(line)
            pts.append(pt)

        ax.set_ylabel('Correlation coefficient')
        leg = ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)

        plot.define_legend_items(leg, lines, pts)
        plot.change_legend_alpha(leg)

        # generic plot parameters and call
        ax.yaxis.labelpad = 10
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both')        
        ax.grid()
        fig.canvas.mpl_connect('pick_event', partial(Plot.on_pick, fig=fig, lined=plot.get_lined()))
        fig.tight_layout()
        show()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: plot dynamic (i.e. multiple series) fft data
    Args:
        trimmed_fft_data: fft data divided into segregated chuncks
        -------------------------------------------------------------------------------------------------------------------- """
    # def plot_fft_dynamic(self):
    #     # generating data structrures
    #     trimmed_time_data = self.generate_dynamic_timeseries_data()
    #     trimmed_fft_data = {}
    #     for pv in self.loaded_pvs:
    #         trimmed_fft_data[pv] = self.calculate_fft(trimmed_time_data[pv])

    #     # creating canvas and axes
    #     fig = figure(figsize=(18,9))#(10,6)
    #     ax = fig.add_subplot()
        
    #     # case in which all data from all pvs will be plotted together
    #     for i, pv in enumerate(self.loaded_pvs):
    #         max_peak = []
    #         for fft_data in trimmed_fft_data[pv]:
    #             fft_props = self.find_fft_properties({'xp': fft_data['xp'], 'yp': fft_data['yp']})
    #             max_peak.append(fft_props['period_max_peak'])
    #         ax.plot(list(map(lambda j: j['ts'], trimmed_fft_data[pv])), max_peak, 'o--', markersize=4, label=pv, lw=0.8)
    #     ax.set_ylabel('Oscilation period [h]', fontsize=18)
        

    #     # plotting first filtered time data from first loaded pv if requested by user
    #     if (self.ui.check_plotFiltSignal.isChecked()):
    #         fig3 = figure()
    #         ax31 = fig3.add_subplot()
    #         ax32 = ax31.twinx()
    #         # pv1_ts = trimmed_fft_data[self.loaded_pvs[0]]['filtered_data_list'][0]/np.linalg.norm(trimmed_fft_data[self.loaded_pvs[0]]['filtered_data_list'][0])
    #         pv1_ts = trimmed_time_data[self.loaded_pvs[0]][0]
    #         pv2_ts = trimmed_time_data[self.loaded_pvs[1]][0]

    #         color = ['#32a83a','#c78212']
    #         ax31.plot(pv1_ts, label=self.loaded_pvs[0], color=color[0])
    #         ax32.plot(pv2_ts, label=self.loaded_pvs[1], color=color[1])
    #         ax31.legend()
    #         ax32.legend()
    #         show()

    #     # saving fft curves if requested by user
    #     # if (self.ui.check_saveFig.isChecked()):
    #     #     self.save_fft_figures(timestamp, xp, yf)


    def plot_fft_dynamic(self):
        import matplotlib.pyplot as plt
        timeserie = self.data[0].iloc[:,0].values

        # applying filter if needed
        if (self.ui.check_applyFilter.isChecked()):
            timeserie = self.filter_timeserie(self.data[0])

        ts1 = time.mktime(datetime.strptime(self.data[0].index.values[0], "%d.%m.%y %H:%M").timetuple())
        ts2 = time.mktime(datetime.strptime(self.data[0].index.values[1], "%d.%m.%y %H:%M").timetuple())
        fs = 1/(ts2 - ts1)

        f, t, Sxx = spectrogram(timeserie, fs)
        plt.pcolormesh(t, f, Sxx, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')

        fig = figure()
        ax = fig.add_subplot()
        ax.plot(timeserie)
        show()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: save images of the fft plots in output's directory 
    Args:
        timestamp: dict containing a list of timestamps from every fft record of every loaded pv
        xp: dict containing a list of x axis (periods) data list of fft of every loaded pv
        yf: dict containing a list of y axis (power fft) data list of fft of every loaded pv
        -------------------------------------------------------------------------------------------------------------------- """
    def save_fft_figures(self, timestamp, xp, yf):
        # creating canvas and axes
        fig = figure(figsize=(18,9))
        ax = fig.add_subplot()
        # creating directory to save images
        dir_path = './output/' + self.ui.inputTxt_dirFig.text()
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        # defining a reference to plot only figures containing it (specially in case of multiple pvs)
        if ('RF-Gen:GeneralFreq-RB' in self.loaded_pvs):
            ref = timestamp['RF-Gen:GeneralFreq-RB']
        else:
            ref = timestamp[self.loaded_pvs[0]]
        # curve's colors
        color = ['#32a83a','#c78212']
        # iterating over reference's timestamps
        for i, ts in enumerate(ref):
            # iterating over loaded pvs
            for j, pv in enumerate(self.loaded_pvs):
                # checking if current ref's timestamp is presented in pv's timestamp and only plot in this case
                if (ts in timestamp[pv]):
                    # finding index based on item on timestamp structure
                    idx = timestamp[pv].index(ts)
                    # normalizing
                    norm = np.linalg.norm(yf[pv][idx])
                    norm_yf = yf[pv][idx]/norm
                    # finding peaks
                    peaks, _ = find_peaks(norm_yf)
                    # ploting
                    ax.plot(xp[pv][idx], norm_yf, label=pv, color=color[j])
                    # ax.plot(xp[pv][idx][peaks], norm_yf[peaks], 'x', ms=12, color='red')

            # increasing x ticks density
            ax.set_xticks(np.arange(0,46, 1))
            ax.set_xticks(np.arange(0,46, 0.5), minor=True)
            # drawing box containing the datetime of the plotted serie
            ob = offsetbox.AnchoredText(ts, loc=2, pad=0.25, borderpad=0.5, prop=dict(fontsize=18, weight=550))#3
            ax.add_artist(ob)
            # limiting x axis
            ax.set_xlim(0.5, 5)

            ax.grid(True, axis='x', which='both')
            ax.legend(prop={'size': 16})
            ax.xaxis.labelpad = 10
            ax.yaxis.labelpad = 10

            ax.set_xlabel('Period [h]', fontsize=25) #16
            ax.set_ylabel(r'Normalized power FFT [$\frac{{mm}^2}{{mm}^2}$]', fontsize=25) #16

            ax.tick_params(axis='both', labelsize=23)
            # saving image
            savefig(f"{dir_path}/fft-{i}.png", dpi=150)
            # cleaning ax
            ax.cla()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_2D_static(self):
        plot = Plot()
        fig, ax = plot.get_plot_props()

        num_of_var = len(self.data)
        lines, legends = [], []
        for data_df in self.data:
            if (self.ui.check_applyFilter.isChecked()):
                y_data = np.transpose(self.filter_timeserie(data_df))
            else:
                y_data = data_df.iloc[:,:].values
            dt = [datetime.strptime(record, "%d.%m.%y %H:%M") for record in data_df.index.values]


            # if more than one type os PV is plotted, normalize it to handle scale diference
            if (num_of_var > 1):
                data = y_data.T
                for i in range(len(data)):
                    mapping = interp1d([min(data[i]), max(data[i])],[-1,1])
                    data[i] = mapping(data[i])
                y_data = data.T

            line = ax.plot(dt, y_data)
            [legends.append(var) for var in data_df.columns.values]
            [lines.append(l) for l in line] 
        
        leg = ax.legend(lines, legends)        
        plot.define_legend_items(leg, lines)
        plot.change_legend_alpha(leg)

        # generic plot parameters and call
        ax.yaxis.labelpad = 10
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both')        
        ax.grid()
        fig.canvas.mpl_connect('pick_event', partial(Plot.on_pick, fig=fig, lined=plot.get_lined()))
        fig.tight_layout()
        show()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_2D(self, plot_data):
        # importing specific package on demand
        try:
            sns
        except NameError:
            import seaborn as sns

        # setting plot style
        sns.set()
        sns.set_style('ticks')
        sns.set_palette(sns.light_palette("green", n_colors=len(plot_data.index.values)))

        # creating directory to save images
        dir_path = './output/' + self.ui.inputTxt_dirFig.text()
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        
        # finding min and max limits for the y axis
        max_lim = 1.1 * max(plot_data.max())
        min_lim = 1.1 * min(plot_data.min())

        # smoothing curve (x axis)
        x = plot_data.columns.values
        x_smooth = np.linspace(min(x), max(x), 300)

        fig = figure(figsize=(18,9))#(10,6)
        ax = fig.add_subplot()
        ax2 = ax.twinx()
        i = 0
        num_data = len(plot_data.index)
        self.printProgressBar(0, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for timestamp, values in plot_data.iterrows():

            # smoothing curve (y axis)
            # a_BSpline = make_interp_spline(x, values, k=2)
            # y_smooth = a_BSpline(x_smooth)

            # ax.plot(x_smooth, y_smooth, color='limegreen')
            # ax2.scatter(x, values, color='forestgreen', s=4)

            # ax.plot(x_smooth, y_smooth)
            ax.plot(x, values)

            ax2.axes.get_yaxis().set_visible(False)
            ax.tick_params(axis='both', labelsize=15) #size=13
            ax.grid(True)

            tickpos = np.linspace(1,20,20)
            # tickpos = np.linspace(1,60,20)
            ax.set_xticks(tickpos)
            ax.set_xticklabels(tickpos)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

            ax.set_ylim(min_lim, max_lim)
            ax2.set_ylim(min_lim, max_lim)

            # drawing box containing the datetime of the plotted serie
            text = ax.text(0.5, ax.get_ylim()[1]*0.85, timestamp, fontsize=18, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
            dt = datetime.strptime(timestamp, "%d.%m.%y %H:%M")
            if (dt > datetime(2021, 6, 25, 9, 20) and dt < datetime(2021, 6, 25, 13, 6)):
                text.set_color('red')

            # ax.set_title("HLS @ Poço desligado", weight='semibold', fontsize=18, y=1.05)
            ax.set_ylabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=17) #14
            # ax.set_xlabel('Eixos do prédio', fontsize=17, labelpad=6)
            ax.set_xlabel('Setor do Anel de Armazenamento', fontsize=17, labelpad=6)

            # saving images if needed
            if (self.ui.check_saveFig.isChecked()):
                savefig(f"{dir_path}/hls-ax-{i}.png", dpi=150)
                ax.cla()
                self.printProgressBar(i + 1, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)
            else:
                if i % 50 == 0:
                    self.printProgressBar(i + 1, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)


            ax2.cla()


            i += 1

        if (not self.ui.check_saveFig.isChecked()):
            plt.show()

    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_3D(self, plot_data):
        fig = figure(figsize=(9.8,7))
        ax = fig.add_subplot(111, projection='3d')
        
        density = len(plot_data.columns.values)
        repeat_times = 2
        R = np.linspace(70, 90, repeat_times*(density+1))
        u = np.linspace(0,  2*np.pi, repeat_times*(density+1))
        x = np.outer(R, np.cos(u))
        y = np.outer(R, np.sin(u))

        R_smooth = np.linspace(70, 90, 300)
        u_smooth = np.linspace(0,  2*np.pi, 300)
        x_smooth = np.outer(R_smooth, np.cos(u_smooth))
        y_smooth = np.outer(R_smooth, np.sin(u_smooth))

        # finding min and max limits for the y axis
        max_lim = 1.2 * max(plot_data.max())
        min_lim = 1.2 * min(plot_data.min())

        # finding x and y coords of cut and fill line
        x_cutfill = [x[-1][6*repeat_times], x[-1][17*repeat_times]]
        y_cutfill = [y[-1][6*repeat_times], y[-1][17*repeat_times]]
        i=0
        size_plot_array = plot_data.index.size

        dir_path = './output/' + self.ui.inputTxt_dirFig.text()
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        for timestamp, values in plot_data.iterrows():
            # printing progress
            if (i%10==0):
                print(f'\r[{str(round(i/size_plot_array*100,2))}%]',end='')

            level = values.to_numpy()
            level = np.append(level, level[0])
            # level = np.repeat(level, repeat_times)

            # smoothing curve
            a_BSpline = make_interp_spline(np.linspace(0, 20, len(level)), level)
            level_smooth = a_BSpline(np.linspace(0, 20, 300))
            z_smooth = np.outer(level_smooth, np.ones(300)).T
            surf = ax.plot_surface(x_smooth,y_smooth,z_smooth,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h




            # z = np.outer(level, np.ones(repeat_times*(density+1))).T
            # surf = ax.plot_surface(x,y,z,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h
            # ax.plot(x_cutfill,y_cutfill, [level[6*repeat_times], level[17*repeat_times]])
            colorbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.13)

            # ax.set_title("HLS @ 25/06/2021 (Artesian Well shutdown)", weight='semibold', fontsize=15, y=1.02, x=0.65)
            ax.set_title("HLS @ Poço desligado", weight='semibold', fontsize=15, y=1.02, x=0.65)
            ax.set_xlabel('x [m]', fontsize=12)
            ax.set_zlabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=12, labelpad=7) #14
            ax.set_ylabel('y [m]', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)

            ax.view_init(15, -55+180+20)
            # ax.view_init(15, -55)
            # ax.view_init(8, -20)

            ax.set_zlim(min_lim, max_lim)
            surf.set_clim(min_lim, max_lim)

            # drawing box containing the datetime of the plotted serie
            box_text = timestamp
            dt = datetime.strptime(timestamp, "%d.%m.%y %H:%M")

            ax.text(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[1]*1.25, box_text, fontsize=13, bbox=dict(boxstyle='round', facecolor='white'))
        
            
            fig.tight_layout()
            savefig(f"{dir_path}/hls-timeseries-{i}.png", dpi=150)
            ax.cla()
            colorbar.remove()

            i+=1

    def plot_data_2D3D(self, plot_data):
        fig = figure(figsize=(16,7))
        ax_2d = fig.add_subplot(121)
        ax_2d_2 = ax_2d.twinx()
        ax_3d = fig.add_subplot(122, projection='3d')
        
        ##### 2D stuff #####
        # finding min and max limits for the y axis
        max_lim_2d = 1.1 * max(plot_data.max())
        min_lim_2d = 1.1 * min(plot_data.min())

        # smoothing curve (x axis)
        x = plot_data.columns.values
        x_smooth_2d = np.linspace(min(x), max(x), 300)

        ##### 3D stuff #####
        R_smooth = np.linspace(70, 90, 300)
        u_smooth = np.linspace(0,  2*np.pi, 300)
        x_smooth = np.outer(R_smooth, np.cos(u_smooth))
        y_smooth = np.outer(R_smooth, np.sin(u_smooth))

        # finding min and max limits for the y axis
        max_lim = 1.2 * max(plot_data.max())
        min_lim = 1.2 * min(plot_data.min())

        dir_path = './output/' + self.ui.inputTxt_dirFig.text()
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        i=0
        num_data = len(plot_data.index)
        self.printProgressBar(0, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)

        for timestamp, values in plot_data.iterrows():
            ##### 2D stuff #####
            # smoothing curve (y axis)
            a_BSpline = make_interp_spline(x, values, k=2)
            y_smooth_2d = a_BSpline(x_smooth_2d)

            ax_2d.plot(x_smooth_2d, y_smooth_2d, color='limegreen')
            # ax2.plot(x_smooth, y_smooth)
            ax_2d_2.scatter(x, values, color='forestgreen', s=4)


            ax_2d_2.axis('off')
            ax_2d.tick_params(axis='both', labelsize=15) #size=13
            ax_2d.grid(True)

            # tickpos = np.linspace(1,20,20)
            tickpos = np.linspace(1,60,10)
            ax_2d.set_xticks(tickpos)
            ax_2d.set_xticklabels(tickpos)
            ax_2d.xaxis.set_major_formatter(FormatStrFormatter('%d'))

            ax_2d.set_ylim(min_lim_2d, max_lim_2d)
            ax_2d_2.set_ylim(min_lim_2d, max_lim_2d)

            # drawing box containing the datetime of the plotted serie
            _ = ax_2d.text(0.5, ax_2d.get_ylim()[1]*0.85, timestamp, fontsize=18, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

            ax_2d.set_title("HLS @ Poço ligado", weight='semibold', fontsize=18, y=1.05)
            ax_2d.set_ylabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=17) #14
            ax_2d.set_xlabel('Eixos do prédio', fontsize=17, labelpad=6)

            ##### 3D stuff #####

            level = values.to_numpy()
            level = np.append(level, level[0])

            # smoothing curve
            a_BSpline = make_interp_spline(np.linspace(0, 20, len(level)), level)
            level_smooth = a_BSpline(np.linspace(0, 20, 300))
            z_smooth = np.outer(level_smooth, np.ones(300)).T
            surf = ax_3d.plot_surface(x_smooth,y_smooth,z_smooth,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h

            colorbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.13)

            # ax.set_title("HLS @ 25/06/2021 (Artesian Well shutdown)", weight='semibold', fontsize=15, y=1.02, x=0.65)
            # ax_3d.set_title("HLS @ Poço desligado", weight='semibold', fontsize=15, y=1.02, x=0.65)
            ax_3d.set_xlabel('x [m]', fontsize=12)
            # ax_3d.set_zlabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=12, labelpad=7) #14
            ax_3d.set_ylabel('y [m]', fontsize=12)
            ax_3d.tick_params(axis='both', labelsize=10)
            

            ax_3d.view_init(15, -55+180+20)
            # ax.view_init(15, -55)
            # ax.view_init(8, -20)

            ax_3d.set_zlim(min_lim, max_lim)
            surf.set_clim(min_lim, max_lim)

            # ax_3d.text(ax_3d.get_xlim()[0], ax_3d.get_ylim()[0], ax_3d.get_zlim()[1]*1.25, box_text, fontsize=13, bbox=dict(boxstyle='round', facecolor='white'))
        
            
            fig.tight_layout()
            savefig(f"{dir_path}/hls-timeseries-{i}.png", dpi=150)
            ax_3d.cla()
            colorbar.remove()

            ax_2d.cla()
            ax_2d_2.cla()

            self.printProgressBar(i + 1, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)
            i+=1

    def calc_cross_corr (self, serie1, serie2):
        s0 = (serie1 - np.mean(serie1))/(np.std(serie1)*len(serie1))
        s1 = (serie2 - np.mean(serie2))/(np.std(serie2))
        corr = np.correlate(s0, s1, mode='full')
        # storing only the maximum coeficient - when series are in phase
        return max(corr)

    def plot_directional(self):
        # extracting data from HLS sensors and earth tides
        for data_df in self.data:
            if (data_df.columns[0] == self.HLS_LEGEND[0]):
                hls_df = data_df
            elif (data_df.columns[0] == self.EARTHTIDES_PVS_LIST[0]):
                tides_df = data_df
            elif (data_df.columns[0] == self.WELLPRESSURE_PV):
                well_df = data_df
            elif (data_df.columns[0] == self.RFFREQ_PV):
                rf_df = data_df
        
        hls_pvs = np.array(self.generate_all_sensors_list())
        # generating list of index for the hls oposite sensors
        pairs_idx = []
        for i in range(10):
            pairs_idx.append([i,i+10])
        pairs_idx.append([4, 5])
        pairs_idx.append([15, 14])
        pairs_idx.append([0, 19])
        pairs_idx.append([9, 10])

        # creating dfs
        hls_pairs_df = pd.DataFrame(columns=['tide', 'var', 'pearson', 'spearman', 'cross'])
        well_rf_df = pd.DataFrame(columns=['tide', 'var', 'pearson', 'spearman', 'cross'])

        # subtracting oposite sensors' readings
        levels = []
        for idx in pairs_idx:
            levels.append({'var': f'{hls_pvs[idx[0]].split(":")[1][-3:]} x {hls_pvs[idx[1]].split(":")[1][-3:]}', 'val': (hls_df.iloc[:,idx[0]] - hls_df.iloc[:,idx[1]]).values})
        # calculating correlation in relation to each earth tides
        for tide_name in tides_df.columns.values:
            tide = tides_df.loc[:,tide_name]
            print(f'------- {tide_name} -------')
            print('vars\t\tpearson\tspearman cross')
            for level in levels:
                corr_p, _ = pearsonr(level['val'], tide)
                corr_s, _ = spearmanr(level['val'], tide)
                corr_cc = self.calc_cross_corr(level['val'], tide)
                print(f'{level["var"]}\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
                hls_pairs_df = hls_pairs_df.append({'tide': tide_name, 'var': level['var'],\
                                                     'pearson': "{:.2f}".format(abs(corr_p)),\
                                                     'spearman': "{:.2f}".format(abs(corr_s)),\
                                                     'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
            print('\n\n')

        # calculating correlation in relation to earth tides summed up
        print(f'------- SUM OF EARTH TIDES -------')
        print('vars\t\tpearson\tspearman cross')
        tides = tides_df.iloc[:,0] + tides_df.iloc[:,1] + tides_df.iloc[:,2]
        for level in levels:
            corr_p, _ = pearsonr(level['val'], tides)
            corr_s, _ = spearmanr(level['val'], tides)
            corr_cc = self.calc_cross_corr(level['val'], tides)
            print(f'{level["var"]}\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
            hls_pairs_df = hls_pairs_df.append({'tide': 'SUM OF 3 TIDES', 'var': level['var'],\
                                                'pearson': "{:.2f}".format(abs(corr_p)),\
                                                'spearman': "{:.2f}".format(abs(corr_s)),\
                                                'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
        print('\n')
        for i, hls_pv in enumerate(hls_pvs):
            corr_p, _ = pearsonr(hls_df.iloc[:,i], tides)
            corr_s, _ = spearmanr(hls_df.iloc[:,i], tides)
            corr_cc = self.calc_cross_corr(hls_df.iloc[:,i], tides)
            print(f'{hls_pvs[i]}\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
            hls_pairs_df = hls_pairs_df.append({'tide': 'SUM OF 3 TIDES', 'var': hls_pvs[i],\
                                                'pearson': "{:.2f}".format(abs(corr_p)),\
                                                'spearman': "{:.2f}".format(abs(corr_s)),\
                                                'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
        print('\n\n')

        # calculating correlation between well height and earth tides
        for tide_name in tides_df.columns.values:
            tide = tides_df.loc[:,tide_name]
            well = well_df.iloc[:,0]
            rf = rf_df.iloc[:,0]
            print(f'------- {tide_name} -------')
            print('vars\tpearson\tspearman cross')
            corr_p, _ = pearsonr(well, tide)
            corr_s, _ = spearmanr(well, tide)
            corr_cc = self.calc_cross_corr(well, tide)
            print(f'Well\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
            well_rf_df = well_rf_df.append({'tide': tide_name, 'var': 'Well',\
                                            'pearson': "{:.2f}".format(abs(corr_p)),\
                                            'spearman': "{:.2f}".format(abs(corr_s)),\
                                            'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
            corr_p, _ = pearsonr(rf, tide)
            corr_s, _ = spearmanr(rf, tide)
            corr_cc = self.calc_cross_corr(rf, tide)
            print(f'RF\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}\n')
            well_rf_df = well_rf_df.append({'tide': tide_name, 'var': 'RF',\
                                                'pearson': "{:.2f}".format(abs(corr_p)),\
                                                'spearman': "{:.2f}".format(abs(corr_s)),\
                                                'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
            
        hls_pairs_df = hls_pairs_df.append(well_rf_df)
        hls_pairs_df.to_excel('./output/tides_analysis.xlsx')
            

        # fig, ax = plt.subplots()
        # ax.plot(tides)
        # ax.axes.get_xaxis().set_visible(False)
        # plt.show()    


    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()






    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: ui function to display messages into text element
    Args:
        message: text to be displayed
        severity: indicator of the type of the message
            values: 'normal', 'danger', 'alert' or 'success'
        -------------------------------------------------------------------------------------------------------------------- """
    def logMessage(self, message, severity='normal'):
        if (severity != 'normal'):
            if (severity == 'danger'):
                color = 'red'
            elif (severity == 'alert'):
                color = 'yellow'
            elif (severity == 'success'):
                color = 'green'

            # saving properties
            tc = self.ui.log.textColor()
            self.ui.log.setTextColor(QColor(color))
            self.ui.log.append(message)
            self.ui.log.setTextColor(tc)
        else:
            self.ui.log.append(message)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: function to clean all the app level variables
        -------------------------------------------------------------------------------------------------------------------- """
    def clean_loaded_data(self):
        # app level variables
        self.data = []
        self.loaded_pvs = []
        # ui
        self.ui.txt_loadedPvs.clear()
        self.ui.btn_plot.setEnabled(False)
        self.ui.btn_makeVideo.setEnabled(False)
        self.ui.label_dataLoaded.setStyleSheet("background-color:rgb(255, 99, 101);color:rgb(0, 0, 0);padding:3;")
        self.ui.label_dataLoaded.setText("No data loaded")


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: ui function to display a pv that was loaded
    Args:
        pv: PV's name
        -------------------------------------------------------------------------------------------------------------------- """
    def add_loaded_pv(self, pvs):
        for pv in pvs:
            self.loaded_pvs.append(pv)
            self.ui.txt_loadedPvs.append('▶ '+ pv)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: function that takes current loaded data and divide into small time chunks
    Output: dict containing a list of multiple timeseries for each of loaded pvs
        -------------------------------------------------------------------------------------------------------------------- """
    def generate_dynamic_timeseries_data (self):
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


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: event function called when fetch button is pressed on ui; fetch data from Archiver according to user's selections
        -------------------------------------------------------------------------------------------------------------------- """
    @asyncSlot()
    async def on_btn_fetchFromArchiver_clicked(self):
        self.logMessage('Fetching from Archiver...')

        # get datetime from ui
        date_ini = self.ui.datetime_init.date().toString('yyyy-MM-dd')
        time_ini = self.ui.datetime_init.time().addSecs(3*60*60).toString('THH:mm:ss.zzzZ')
        date_end = self.ui.datetime_end.date().toString('yyyy-MM-dd')
        time_end = self.ui.datetime_end.time().addSecs(3*60*60).toString('THH:mm:ss.zzzZ')
        if (self.ui.datetime_end.time().hour() >= 21):
            date_end = self.ui.datetime_end.date().addDays(1).toString('yyyy-MM-dd')
        timespam = (date_ini + time_ini, date_end + time_end)
        
        special_case = None
        # user options
        if (self.ui.check_allPvs.isChecked()):
            pvs = self.generate_all_sensors_list()
            # pv_internal_name = 'all-hls'
            column_names = self.HLS_LEGEND
        elif (self.ui.check_rfPv.isChecked()):
            # pv_internal_name = self.RFFREQ_PV
            pvs = [self.RFFREQ_PV]
            column_names = pvs
        elif (self.ui.check_wellpressure.isChecked()):
            # pv_internal_name = self.WELLPRESSURE_PV
            pvs = [self.WELLPRESSURE_PV]
            column_names = pvs
        elif (self.ui.check_earthtides.isChecked()):
            # pv_internal_name = 'tides'
            pvs = self.EARTHTIDES_PVS_LIST
            column_names = self.EARTHTIDES_PVS_LIST
        elif (self.ui.check_selectPvs.isChecked()):
            # pvs = [self.ui.inputTxt_pvs.text()]
            pvs = ['TU-11C:SS-HLS-Ax46NW4:Level-Mon', self.RFFREQ_PV]
            column_names = pvs
        elif (self.ui.check_opositeHLS.isChecked()):
            # pv_internal_name = self.ui.inputTxt_pvs.text()
            pvs = self.HLS_OPOSITE_PVS_LIST
            column_names = ['HLS Easth-West', 'HLS North-South']
            special_case = 'oposite HLS'
        else:
            self.logMessage('PV not recognized', 'danger')
            return
        optimize = self.ui.check_optimize.isChecked()
        time_in_minutes = self.ui.spin_timeOptimize.value()

        try:
            # retrieving raw data from Archiver
            json_data = await self.retrieve_data(pvs, timespam[0], timespam[1], optimize, time_in_minutes)

            # mapping pv's values
            data = [np.array(list(map(lambda i: i['val'], serie))) for serie in map(lambda j: j[0]['data'], json_data)]
            # mapping timestamps
            time_fmt = list(map(lambda data: datetime.fromtimestamp(data['secs']).strftime("%d.%m.%y %H:%M"), json_data[0][0]['data']))

            # checking if data needs some early treatment
            if (special_case):
                if (special_case == 'oposite HLS'):
                    data = [(data[0] - data[1]), (data[2] - data[3])]

            # creating pandas dataframe object
            d = {'datetime': time_fmt}
            for l_data, name in zip(data, column_names):
                d[name] = l_data
            data = DataFrame(data=d)
            # indexing by datetime
            data.reset_index(drop=True, inplace=True)
            data = data.set_index('datetime')
            # saving to app level variable
            self.data.append(data)

            # storing excel if needed
            if (False):
                self.data[0].to_excel('./output/out-data.xlsx')

        except IndexError:
            self.logMessage('No data retrieved for this timespam', 'danger')
        else:
            self.logMessage('Fetched!', 'success')
        finally:
            self.ui.btn_plot.setEnabled(True)
            self.ui.btn_makeVideo.setEnabled(True)
            self.ui.label_dataLoaded.setStyleSheet("background-color:rgb(163, 255, 138);color:rgb(0, 0, 0);padding:3;")
            self.ui.label_dataLoaded.setText("Data loaded")
            self.add_loaded_pv(pvs)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: event function called when plot button is pressed on ui; plots data based on user's selections
        -------------------------------------------------------------------------------------------------------------------- """ 
    def on_btn_plot_clicked(self):

        # correlation analysis
        if (self.ui.check_plotCorrel.isChecked()):
            self.app.processEvents()
            self.plot_cross_correl()
            return

        # directional analysis
        if (self.ui.check_plotDirectional.isChecked()):
            self.app.processEvents()
            self.plot_directional()

        # ploting fft series
        if (self.ui.check_plotFFT.isChecked()):
            if (self.ui.check_plotDynamic.isChecked()):
                self.plot_fft_dynamic()
            else:
                self.app.processEvents()
                self.plot_fft_static()
            return
        
        # ploting time series
        if (self.ui.check_plotTime.isChecked()):
            if (self.ui.check_plotDynamic.isChecked()):
                # arranging data structure
                plot_data = self.generate_HLS_data_to_dynamic_time_plot()
                # ploting
                if (self.ui.check_plot2D.isChecked()):
                    self.plot_data_2D(plot_data)
                else:
                    self.plot_data_3D(plot_data)
            else:
                self.plot_data_2D_static()
            return
                

""" --------------------------------------------------------------------------------------------------------------------
    Desc.: main function called when app starts
    -------------------------------------------------------------------------------------------------------------------- """
async def main():
    def close_future(future, loop):
        loop.call_later(10, future.cancel)
        # future.cancel("Close Application")
        future.cancel()

    loop = asyncio.get_event_loop()
    future = asyncio.Future()

    app = QApplication.instance()
    if hasattr(app, 'aboutToQuit'):
        getattr(app, 'aboutToQuit').connect(partial(close_future, future, loop))

    application = App(app)

    await future
    return True

if __name__ == "__main__":
    try:
        qasync.run(main())
    except asyncio.CancelledError:
        sys.exit(0)
# %%
