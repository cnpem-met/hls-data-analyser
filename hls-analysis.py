# https://stackoverflow.com/questions/61842432/pyqt5-and-asyncio

import asyncio
import functools
import sys
from datetime import datetime
import time
import math
import os

import aiohttp

from PyQt5.QtWidgets import (QWidget, QMainWindow, QVBoxLayout, QPushButton)
from PyQt5.QtGui import (QColor)
from PyQt5 import uic
from ui import Ui_MainWindow

import qasync
from qasync import asyncSlot, asyncClose, QApplication

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
from scipy.fft import fft, fftfreq, rfft, ifft, irfft
from scipy.signal import butter, sosfilt, filtfilt, blackman
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal import correlate

import matplotlib.pylab as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import matplotlib.offsetbox as offsetbox
import seaborn as sns

import moviepy.video.io.ImageSequenceClip

class MainWindow(QMainWindow):
    _SESSION_TIMEOUT = 1.0

    archiver_url = 'http://10.0.38.42/retrieval/data/getData.json'
    hls_legend = [17, 16, 15, 14, 13, 1.5, 1.2, 20, 19, 18, 6.8, 6.2, 5, 4, 3, 11.5, 11.2, 10, 9, 8]
    
    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: class function called when create an instance
        -------------------------------------------------------------------------------------------------------------------- """
    def __init__(self):
        super().__init__()

        # initializing ui components
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.btn_fetchFromArchiver.clicked.connect(self.on_btn_fetchFromArchiver_clicked)
        self.ui.btn_plot.clicked.connect(self.on_btn_plot_clicked)
        self.ui.btn_cleanData.clicked.connect(self.clean_loaded_data)
        self.ui.check_selectPvs.toggled.connect(self.toggle_pv_input)
        self.ui.btn_makeVideo.clicked.connect(self.make_video)

        # app state variables
        self.data = []
        self.loaded_pvs = []

        # constants
        self.hls_pvs = self.generate_all_sensors_list()


    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: ui function to trigger 'enable' state of a textbox
        -------------------------------------------------------------------------------------------------------------------- """
    def toggle_pv_input(self):
        self.ui.inputTxt_pvs.setEnabled(self.ui.check_selectPvs.isChecked())


    """  --------------------------------------------------------------------------------------------------------------------
    Desc.: create a video file from images in directory defined in ui's textbox
        -------------------------------------------------------------------------------------------------------------------- """
    def make_video(self):
        image_folder='./output/' + self.ui.inputTxt_dirFig.text()
        fps=6
        image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
        image_files.sort(key=lambda i: int(i.split('/')[-1].split('-')[-1].split('.')[0]))
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
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
        async with session.get(self.archiver_url, params=query) as response:
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
        sensors_list = []
        for c_num in range(1, 5):
            for s_num in range(1, 6):
                sensors_list.append(f'HLS:C{c_num}_S{s_num}_LEVEL')
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
        b, a = butter(4, [5.55e-5, 2.77e-4], 'bandpass', fs=1/T) #bandpass from 1h to 5h
        # for sensor in data_df.columns:
        #     data_df.loc[:,sensor] = filtfilt(b,a, data_df.loc[:,sensor].values)

        # referencing in one sensor
        sens_ref = 19
        data_df = data_df.sub(data_df.loc[:,sens_ref], axis=0)

        # sorting sensors
        data_df = data_df.sort_index(axis=1)

        # calculating bestfit between 2 epochs and applying shift in curve
        for i in range(1, len(data_df.iloc[:,0])):
            offset = self.calc_offset(data_df.iloc[i,:].values, data_df.iloc[i-1,:].values)
            data_df.iloc[i,:] += offset[0]

        # setting first record as time reference
        data_df = data_df - data_df.iloc[0,:]

        return data_df

    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def generate_per_sensor_dataset(self, data_orig):
        data_df = data_orig.copy()
        # referencing in one sensor
        data_df.columns = data_df.columns.map(float)
        sens_ref = 19
        data_df = data_df.sub(data_df.loc[:,sens_ref], axis=0)

        # sorting sensors
        data_df = data_df.sort_index(axis=1)

        # trimming original df to get sequencial periods of x hours (1h40, 2h etc.)
        trimmed_data = []
        # num of periods considering 3 minutes in interval
        period_in_records = 34 # 1h40
        # period_in_records = 40 # 2h
        # period_in_records = 25 # 1h15
        num_records = len(data_df.iloc[:,0])
        total_periods = int(num_records/period_in_records)
        for i in range(total_periods):
            trimmed_data.append(data_df.iloc[i*period_in_records : (i+1)*period_in_records, :])
        print(f'num. periodos: {total_periods}\nrecords sobrando: {num_records%period_in_records}')

        # treating each dataset
        plot_data = []
        for period in trimmed_data:
            # calculating bestfit between 2 epochs and applying shift in curve
            for i in range(1, len(period.iloc[:,0])):
                offset = self.calc_offset(period.iloc[i,:].values, period.iloc[i-1,:].values)
                period.iloc[i,:] += offset[0]

            # setting first record as time reference
            period = period - period.iloc[0,:]

            # structuing plot data
            level = period.iloc[:,:].values
            legend = period.index.values
            sensors = period.columns.values

            plot_data.append((level, legend, sensors))

        return plot_data


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: calculate the Discrete Fast Fourier Transformation from a timeseries
    Args: 
        data_list: tupple containing timeserie data
        pv: the name of the pv that holds the data
    Output: dict containing fft and filtered time data
        -------------------------------------------------------------------------------------------------------------------- """
    def calculate_fft(self, data_df_list):
        fft_plot_data = []
        filtered_timeseries = []
        for data in data_df_list:
            timeserie = np.array(data.iloc[:, 0].values)
            timestamp = data.index.values[1]
            pv = data.columns.values[0]

            # ignoring datasets with more than 50% of repeated values (=no value)
            _, counts_elem = np.unique(timeserie, return_counts=True)
            repeated_elem = 0
            for count in counts_elem:
                if (count > 1):
                    repeated_elem += count - 1
            percentage_repeated = repeated_elem/len(timeserie)
            if (percentage_repeated > 0.5):
                # print(f'Data from {pv} on {data[1]} ignored: percentage repeated fail [{percentage_repeated}].')
                continue
            # defining sampling properties
            ts1 = time.mktime(datetime.strptime(data.index.values[0], "%d.%m.%y %H:%M").timetuple())
            ts2 = time.mktime(datetime.strptime(data.index.values[1], "%d.%m.%y %H:%M").timetuple())
            acq_period = ts2 - ts1
            T = acq_period # in seconds
            N = len(timeserie)
            # creating generic time x axis data
            time_ax = np.linspace(0, T*N, N)
            # creating frequency x axis data
            W = fftfreq(N, T)[:N//2+1]
            # applying filter
            b, a = butter(4, [5.55e-5, 2.77e-4], 'bandpass', fs=1/T) #bandpass from 1h to 5h
            try:
                filtered_data = filtfilt(b,a, timeserie)
            except ValueError:
                # print(f'Filter not applyed in ts {timestamp}')
                filtered_data = timeserie
            # calculating fft
            yr = rfft(filtered_data)
            yr = np.abs(yr)**2
            # finding the peaks of fft and its properties
            yp = 2/(N/2) * yr
            peaks, _ = find_peaks(yp)
            # prominences = peak_prominences(yp, peaks)[0]
            widths = peak_widths(yp, peaks, rel_height=0.5)[0]
            xf = np.array(W)
            xp = 1/xf/60/60
            try:
                y_max_peak = max(yp[peaks])
            except ValueError:
                print(f'Not calculated fft for pv {pv} in {timestamp}')
                continue
            x_max_peak = np.where(yp == y_max_peak)[0][0]
            period_max_peak = xp[x_max_peak]
            idx_max_peak = np.where(peaks == x_max_peak)[0][0]
            width_max_peak = widths[idx_max_peak]
            # filtering curves by the means of the amplitude of max peak and its width
            if (pv == 'RF-Gen:GeneralFreq-RB'):
                if (y_max_peak > 5000 and y_max_peak < 300000 and width_max_peak < 4):
                    fft_plot_data.append((W, yr, N, timestamp, period_max_peak))
                    filtered_timeseries.append(filtered_data)
            elif (pv == 'HLS:C4_S2_LEVEL'):
                if (y_max_peak > 1e-3 and width_max_peak < 3):
                    fft_plot_data.append((W, yr, N, timestamp, period_max_peak))
                    filtered_timeseries.append(filtered_data)
            else:
                self.logMessage(f"Doesn't know how to plot {pv}, skiping it...", 'danger')
                continue
        output = {'fft_plot_data': fft_plot_data, 'filtered_data_list': filtered_timeseries, 'time': time_ax}
        return output


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: plot static fft data
    Args: 
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_fft_static(self, fft_data_raw):
        fig = plt.figure(figsize=(18,9))#(10,6)
        ax = fig.add_subplot()

        for fft_data in fft_data_raw['fft_plot_data']:
            (xf, yf, N) = fft_data
            
            # maping freq to period
            xf = np.array(xf)
            xp = 1/xf/60/60
            # ploting fft
            ax.plot(xp, 2/(N/2) * yf)
        # formating plot
        ax.grid(True, axis='x', which='both')
        # increasing x ticks density
        ax.set_xticks(np.arange(0,46, 2))
        ax.set_xticks(np.arange(0,46, 1), minor=True)

        ax.set_xlim(0.5, 8)
        # ax.set_ylim(-0.0002, 0.025)

        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10

        ax.set_xlabel('Period [h]', fontsize=15) #16
        ax.set_ylabel('Absolute power FFT [mm^2]', fontsize=15) #16

        ax.tick_params(axis='both', labelsize=23)


        # ploting filtered timeseries for checking purpouses
        if (self.ui.check_plotFiltSignal.isChecked()):
            fig2 = plt.figure(figsize=(18,9))#(10,6)
            ax2 = fig2.add_subplot()
            for filtered_data in fft_data_raw['filtered_data_list']:
                ax2.plot(fft_data_raw['time'], filtered_data)

        plt.show()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: save images of the fft plots in output's directory 
    Args:
        timestamp: dict containing a list of timestamps from every fft record of every loaded pv
        xp: dict containing a list of x axis (periods) data list of fft of every loaded pv
        yf: dict containing a list of y axis (power fft) data list of fft of every loaded pv
        -------------------------------------------------------------------------------------------------------------------- """
    def save_fft_figures(self, timestamp, xp, yf):
        # creating canvas and axes
        fig = plt.figure(figsize=(18,9))
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
            plt.savefig(f"{dir_path}/fft-{i}.png", dpi=150)
            # cleaning ax
            ax.cla()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: plot dynamic (i.e. multiple series) fft data
    Args:
        trimmed_fft_data: fft data divided into segregated chuncks
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_fft_dynamic(self, trimmed_fft_data):
        # creating canvas and axes
        fig = plt.figure(figsize=(18,9))#(10,6)
        ax = fig.add_subplot()

        # destructuring input data and creating separeted data structures
        timestamp, max_peak, xp, yf, N = {}, {}, {}, {}, {}
        for i, pv in enumerate(self.loaded_pvs):
            fft_data = trimmed_fft_data[self.loaded_pvs[i]]['fft_plot_data']
            timestamp[pv], max_peak[pv], xp[pv], yf[pv], N[pv] = [], [], [], [], []
            for record in fft_data:
                xf = np.array(record[0])
                n = record[2]
                timestamp[pv].append(datetime.strptime(record[3], "%d.%m.%y %H:%M"))
                max_peak[pv].append(record[4])
                xp[pv].append(1/xf/60/60)
                N[pv].append(n)
                yf[pv].append(2/(n/2) * record[1])
        
        # plot visual parameters
        marker = ['o-', 'o--']
        marker_size = [4,3]
        plot_lw = [2.3, 1.8]
        plot_color = ['#dd3be3', '#2d81cf']
        
        # case in which all data from all pvs will be plotted together 
        if (not self.ui.check_mutualPlotOnly.isChecked()):
            for i, pv in enumerate(self.loaded_pvs):
                    ax.plot(timestamp[pv], max_peak[pv], 'o--', markersize=4, label=pv, lw=0.8, color=plot_color[i])
            ax.set_ylabel('Oscilation period [h]', fontsize=18)
            ax.set_title("Period of max. peak on RF's FFT | 12h analysis", fontsize=20, y=1.04)
        # case in which only data presented in all pvs will be plotted
        else:
            # ****** considering only 2 pvs *******
            timestamp_filtered, max_peak_filtered = {self.loaded_pvs[0]: [], self.loaded_pvs[1]: []}, {self.loaded_pvs[0]: [], self.loaded_pvs[1]: []} 
            timestamp_filtered[self.loaded_pvs[0]] = list(filter(lambda i: i in timestamp[self.loaded_pvs[1]], timestamp[self.loaded_pvs[0]]))
            timestamp_filtered[self.loaded_pvs[1]] = list(filter(lambda i: i in timestamp[self.loaded_pvs[0]], timestamp[self.loaded_pvs[1]]))
            idx_pv0 = [timestamp[self.loaded_pvs[0]].index(i) for i in timestamp_filtered[self.loaded_pvs[0]]]
            idx_pv1 = [timestamp[self.loaded_pvs[1]].index(i) for i in timestamp_filtered[self.loaded_pvs[1]]]
            max_peak_filtered[self.loaded_pvs[0]] = np.array(max_peak[self.loaded_pvs[0]])[idx_pv0]
            max_peak_filtered[self.loaded_pvs[1]] = np.array(max_peak[self.loaded_pvs[1]])[idx_pv1]
            yf_filtered0 = np.array(yf[self.loaded_pvs[0]])[idx_pv0]
            yf_filtered1 = np.array(yf[self.loaded_pvs[1]])[idx_pv1]
            corr_fft = []
            # correlation calculation from fft curves
            for yf0, yf1 in zip(yf_filtered0, yf_filtered1):
                pearson_r = np.corrcoef(yf0, yf1)
                corr_fft.append(pearson_r[0,1])
            # correlation calculation from time filtered signals
            time_filt_filtered_s0 = np.array(trimmed_fft_data[self.loaded_pvs[0]]['filtered_data_list'])[idx_pv0]
            time_filt_filtered_s1 = np.array(trimmed_fft_data[self.loaded_pvs[1]]['filtered_data_list'])[idx_pv1]
            corr_time = []
            for time_filt_s0, time_filt_s1 in zip(time_filt_filtered_s0, time_filt_filtered_s1):
                s0 = (time_filt_s0 - np.mean(time_filt_s0))/(np.std(time_filt_s0)*len(time_filt_s0))
                s1 = (time_filt_s1 - np.mean(time_filt_s1))/(np.std(time_filt_s1))
                corr = np.correlate(s0, s1, mode='full')
                corr_time.append(max(corr))
            
            # plotting bars containing time and frequency correlation information
            ax.bar(timestamp_filtered[self.loaded_pvs[0]], corr_fft, color='#00995c', alpha=0.3, label='FFT (Pearson)')
            ax.bar(timestamp_filtered[self.loaded_pvs[0]], corr_time, color='#944dff', alpha=0.3, label='Time (cross-correlation normalized)')

            ax.set_ylabel('Correlation coefficient', fontsize=18)
            ax.set_title("Correlation HLS x RF | 12h analysis", fontsize=20, y=1.04)
            ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0., prop={'size': 12})

        # generic plot parameters and call
        ax.yaxis.labelpad = 10
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='both', labelsize=16)        
        ax.grid()
        plt.show()

        # plotting first filtered time data from first loaded pv if requested by user
        if (self.ui.check_plotFiltSignal.isChecked()):
            fig3 = plt.figure()
            ax3 = fig3.add_subplot()
            timeseries = trimmed_fft_data[self.loaded_pvs[0]]['filtered_data_list'][0]/np.linalg.norm(trimmed_fft_data[self.loaded_pvs[0]]['filtered_data_list'][0])

            color = ['#32a83a','#c78212']
            ax3.plot(timeseries, label=self.loaded_pvs[0], color=color[0])
            ax3.legend()
            plt.show()

        # saving fft curves if requested by user
        if (self.ui.check_saveFig.isChecked()):
            self.save_fft_figures(timestamp, xp, yf)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_2D_static(self, level, time):
        fig = plt.figure(figsize=(18,9))#(10,6)
        ax = fig.add_subplot()
        time = [datetime.strptime(record, "%d.%m.%y %H:%M") for record in time]
        ax.plot(time, level)
        plt.show()


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_2D(self, plot_data):
        # setting plot style
        sns.set()
        sns.set_style('ticks')
        sns.set_palette(sns.light_palette("green", reverse=False, n_colors=len(plot_data.columns.values)))

        fig = plt.figure(figsize=(18,9))#(10,6)
        ax = fig.add_subplot()

   
        ax.set_title("HLS @ 25/06/2021 (Artesian Well shutdown)", weight='semibold', fontsize=18, y=1.05) #fontsize=19
        for timestamp, values in plot_data.iterrows():
            ax.plot(plot_data.columns.values, values, label=timestamp)

            ax.tick_params(axis='both', labelsize=15) #size=13
            ax.grid(True)

            tickpos = np.linspace(1,20,20)
            ax.set_xticks(tickpos)
            ax.set_xticklabels(tickpos)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

            # ax.set_ylim(-0.02, 0.02)
            # ax.set_ylim(-0.06, 0.12)

            ax.set_ylabel(r'$\Delta \/ {Level} \/ [mm]$', fontsize=17) #14
            ax.set_xlabel('Storage Ring sector', fontsize=17, labelpad=6)

            # plt.savefig(f"./output/media_1h40/24h/6-6-21/hls-ax-{i}.png", dpi=150)
            # ax.cla()
        fig.tight_layout()
        plt.show()

        # saving images if needed
        if (self.ui.check_saveFig.isChecked()):
            # creating directory to save images
            dir_path = './output/' + self.ui.inputTxt_dirFig.text()
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            
            fig = plt.figure(figsize=(18,9))#(10,6)
            ax = fig.add_subplot()
            i = 0 
            for timestamp, values in plot_data.iterrows():
                ax.plot(plot_data.columns.values, values, label=timestamp, color='#39cc45')

                ax.tick_params(axis='both', labelsize=15) #size=13
                ax.grid(True)

                tickpos = np.linspace(1,20,20)
                ax.set_xticks(tickpos)
                ax.set_xticklabels(tickpos)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

                # ax.set_ylim(-0.21, 0.11) #25/6
                # ax.set_ylim(-0.13, 0.07) #24/6
                ax.set_ylim(-0.22, 0.13) #28/6

                # drawing box containing the datetime of the plotted serie
                # ob = offsetbox.AnchoredText(timestamp, loc=2, pad=0.25, borderpad=0.5, prop=dict(fontsize=18, weight=550))#3
                # text = ax.add_artist(ob)
                text = ax.text(0.5, ax.get_ylim()[1]*0.85, timestamp, fontsize=18, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
                dt = datetime.strptime(timestamp, "%d.%m.%y %H:%M")
                if (dt > datetime(2021, 6, 25, 9, 20) and dt < datetime(2021, 6, 25, 13, 6)):
                    text.set_color('red')

                # ax.set_title("HLS @ 25/06/2021 (Artesian Well shutdown)", weight='semibold', fontsize=18, y=1.05)
                ax.set_title("HLS @ 28/06/2021", weight='semibold', fontsize=18, y=1.05)
                ax.set_ylabel(r'$\Delta \/ {Level} \/ [mm]$', fontsize=17) #14
                ax.set_xlabel('Storage Ring sector', fontsize=17, labelpad=6)

                plt.savefig(f"{dir_path}/hls-ax-{i}.png", dpi=150)
                ax.cla()
                i += 1

    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: 
    Args:
    Output:
        -------------------------------------------------------------------------------------------------------------------- """
    def plot_data_3D(self, plot_data):
        fig = plt.figure(figsize=(9.8,7))
        ax = fig.add_subplot(111, projection='3d')

        # retrieving plot data
        # (level, legend_list, sensors) = plot_data

        # finding min e max values from the series
        # max_level, min_level = level.max(), level.min()
        
        density = len(plot_data.columns.values)
        repeat_times = 2
        R = np.linspace(70, 90, repeat_times*(density+1))
        u = np.linspace(0,  2*np.pi, repeat_times*(density+1))

        x = np.outer(R, np.cos(u))
        y = np.outer(R, np.sin(u))

        # finding x and y coords of cut and fill line
        x_cutfill = [x[-1][6*repeat_times], x[-1][17*repeat_times]]
        y_cutfill = [y[-1][6*repeat_times], y[-1][17*repeat_times]]
        i=0

        for timestamp, values in plot_data.iterrows():
            level = values.to_numpy()
            level = np.append(level, level[0])
            level = np.repeat(level, repeat_times)

            z = np.outer(level, np.ones(repeat_times*(density+1))).T

            surf = ax.plot_surface(x,y,z,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h
            ax.plot(x_cutfill,y_cutfill, [level[6*repeat_times], level[17*repeat_times]])
            colorbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.13)

            # ax.set_title("HLS @ 25/06/2021 (Artesian Well shutdown)", weight='semibold', fontsize=15, y=1.02, x=0.65)
            ax.set_title("HLS @ 28/06/2021", weight='semibold', fontsize=15, y=1.02, x=0.65)
            ax.set_xlabel('x [m]', fontsize=12)
            ax.set_zlabel(r'$\Delta \/ {Level} \/ [mm]$', fontsize=12, labelpad=7) #14
            ax.set_ylabel('y [m]', fontsize=12)
            ax.tick_params(axis='both', labelsize=10)

            ax.view_init(15, -55)

            ax.set_zlim(-0.22, 0.13)
            surf.set_clim(-0.22, 0.13)

            # drawing box containing the datetime of the plotted serie
            text = ax.text(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[1]*1.25, timestamp, fontsize=13, bbox=dict(boxstyle='round', facecolor='white'))
            dt = datetime.strptime(timestamp, "%d.%m.%y %H:%M")
            if (dt > datetime(2021, 6, 25, 9, 20) and dt < datetime(2021, 6, 25, 13, 6)):
                text.set_color('red')

            fig.tight_layout()
            plt.savefig(f"./output/hls_28-6-21_3d/hls-timeseries-{i}.png", dpi=150)
            ax.cla()
            colorbar.remove()

            i+=1


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
    def add_loaded_pv(self, pv):
        self.loaded_pvs.append(pv)
        self.ui.txt_loadedPvs.append('▶ '+ pv)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: function that takes current loaded data and divide into small time chunks
    Output: dict containing a list of multiple timeseries for each of loaded pvs
        -------------------------------------------------------------------------------------------------------------------- """
    def generate_dynamic_timeseries_data (self):
        output = {}
        for pv, data in zip(self.loaded_pvs, self.data):
            # trimming original df to get sequencial periods of x hours (1h40, 2h etc.)
            trimmed_data = []
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
            # trim data
            for i in range(total_periods):
                trimmed_data.append(data.iloc[i*num_period_in_records : (i+1)*num_period_in_records, :])
            output[pv] = trimmed_data
        
        return output


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: event function called when fetch button is pressed on ui; fetch data from Archiver according to user selections
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
        
        # user options
        if (self.ui.check_allPvs.isChecked()):
            pvs = self.generate_all_sensors_list()
            pv_internal_name = 'all-hls'
            column_names = self.hls_legend
        elif (self.ui.check_rfPv.isChecked()):
            pv_internal_name = 'RF-Gen:GeneralFreq-RB'
            pvs = [pv_internal_name]
            column_names = [pv_internal_name]
        elif (self.ui.check_selectPvs.isChecked()):
            pv_internal_name = self.ui.inputTxt_pvs.text()
            pvs = [pv_internal_name]
            column_names = [pvs[0]]
        else:
            self.logMessage('PV not recognized', 'danger')
            return
        optimize = self.ui.check_optimize.isChecked()
        time_in_minutes = self.ui.spin_timeOptimize.value()

        # forming data structures
        level_raw, level = [], []
        try:
            res_level = await self.retrieve_data(pvs, timespam[0], timespam[1], optimize, time_in_minutes)
            # treating raw json data
            # level
            for i in range (len(pvs)):
                level_raw.append(res_level[i][0]['data'])
            for serie in level_raw:
                level.append(np.array([record['val'] for record in serie]))
            # datetime
            time = [record['secs'] for record in level_raw[0]]
            time_fmt = [datetime.fromtimestamp(ts).strftime("%d.%m.%y %H:%M") for ts in time]

            # creating pandas dataframe object
            d = {'datetime': time_fmt}
            for l_data, name in zip(level, column_names):
                d[name] = l_data
            data = pd.DataFrame(data=d)
            # indexing by datetime
            data.reset_index(drop=True, inplace=True)
            data = data.set_index('datetime')
            # saving to app level variable
            self.data.append(data)

            # storing excel if needed
            if (False):
                data.to_excel('./data/hls_24h_06-06-21.xlsx')

        except IndexError:
            self.logMessage('No data retrieved for this timespam', 'danger')
        else:
            self.logMessage('Fetched!', 'success')
        finally:
            self.ui.btn_plot.setEnabled(True)
            self.ui.btn_makeVideo.setEnabled(True)
            self.ui.label_dataLoaded.setStyleSheet("background-color:rgb(163, 255, 138);color:rgb(0, 0, 0);padding:3;")
            self.ui.label_dataLoaded.setText("Data loaded")
            self.add_loaded_pv(pv_internal_name)


    """ --------------------------------------------------------------------------------------------------------------------
    Desc.: event function called when plot button is pressed on ui; plots data based on user selections
        -------------------------------------------------------------------------------------------------------------------- """ 
    def on_btn_plot_clicked(self):
        # preparing generic static data to plot
        if (self.ui.check_plotStatic.isChecked()):
            static_plot_data = []
            for pv, data in zip(self.loaded_pvs, self.data):
                static_plot_data.append(data.iloc[:, 0].values)
            ts = self.data[0].index

        # ploting fft series
        if (self.ui.check_plotFFT.isChecked()):
            if (self.ui.check_plotDynamic.isChecked()):
                trimmed_time_data = self.generate_dynamic_timeseries_data()
                trimmed_fft_data = {}
                for pv in self.loaded_pvs:
                    fft_data = self.calculate_fft(trimmed_time_data[pv])
                    trimmed_fft_data[pv] = fft_data
                self.plot_fft_dynamic(trimmed_fft_data)
            else:
                fft_data = self.calculate_fft(static_plot_data)
                self.plot_fft_static(fft_data)
            return
        
        # ploting time series
        if (self.ui.check_plotTime.isChecked()):
            if (self.ui.check_plotDynamic.isChecked()):
                # ploting
                if (self.ui.check_plot2D.isChecked()):
                    plot_data = self.generate_HLS_data_to_dynamic_time_plot()
                    self.plot_data_2D(plot_data)
                else:
                    plot_data = self.generate_HLS_data_to_dynamic_time_plot()
                    self.plot_data_3D(plot_data)
            else:
                self.plot_data_2D_static(static_plot_data, ts)
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
        getattr(app, 'aboutToQuit')\
            .connect(functools.partial(close_future, future, loop))

    mainWindow = MainWindow()
    mainWindow.show()

    await future

    return True


if __name__ == "__main__":
    try:
        qasync.run(main())
    except asyncio.CancelledError:
        sys.exit(0)