# %%

# https://stackoverflow.com/questions/61842432/pyqt5-and-asyncio

import time
import asyncio
from functools import partial
import sys
from datetime import datetime
import math
import os
from typing import Dict
import aiohttp
from itertools import combinations
import json

from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtGui import (QColor)
from matplotlib.pyplot import jet
from scipy.signal.windows.windows import blackman
from calculations.geometrical import calc_offset
from calculations.timeseries import filter_timeseries_in_df
from plot import plot_cross_correl, plot_data_2D, plot_data_2D_static, plot_data_3D, plot_directional, plot_fft_dynamic, plot_fft_static
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
        self.data = {}
        self.loaded_pvs = []

        # constants
        self.hls_pvs = self.generate_all_sensors_list()

    def make_video(self):
        """ create a video file from images in directory defined in ui's textbox """
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        image_folder='./output/' + self.ui.inputTxt_dirFig.text()
        fps_movie=12
        image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
        image_files.sort(key=lambda i: int(i.split('/')[-1].split('-')[-1].split('.')[0]))
        clip = ImageSequenceClip(image_files, fps=fps_movie)
        image_folder+'/'+image_folder.split('/')[-1]+'.mp4'
        clip.write_videofile(image_folder+'/'+image_folder.split('/')[-1]+'.mp4')

    @asyncSlot()
    async def fetch(self, session, pv, time_from, time_to, is_optimized, mean_minutes):
        """ Fetch data from Archiver """
        if is_optimized:
            pv_query = f'mean_{int(60*mean_minutes)}({pv})'
        else:
            pv_query = pv
        query = {'pv': pv_query, 'from': time_from, 'to': time_to}
        async with session.get(self.ARCHIVER_URL, params=query) as response:
            response_as_json = await response.json()
            return response_as_json

    @asyncSlot()
    async def retrieve_data(self, pvs, time_from, time_to, isOptimized=False, mean_minutes=0):
        """ mid function that fetches data from multiple pvs """
        async with aiohttp.ClientSession() as session:
            data = await asyncio.gather(*[self.fetch(session, pv, time_from, time_to, isOptimized, mean_minutes) for pv in pvs])
            return data

    def generate_all_sensors_list (self):
        """ utility function to create a list with all the current HLS PVs names """

        sectors = [17, 16, 15, 14, 13, 1, 1, 20, 19, 18, 6, 6, 5, 4, 3, 11, 11, 10, 9, 8]
        axis = [4, 1, 59, 57, 54, 18, 16, 14, 12, 9, 33, 31, 29, 27, 24, 48, 46, 44, 42, 39]
        quadrant = ['NE5', 'NE4', 'NE3', 'NE2', 'NE1', 'SE5', 'SE4', 'SE3', 'SE2', 'SE1', 'SW5', 'SW4', 'SW3', 'SW2', 'SW1', 'NW5', 'NW4', 'NW3', 'NW2', 'NW1']
        sensors_list = []
        for sector, ax, quad in zip(sectors, axis, quadrant):
            sensors_list.append(f'TU-{sector:02d}C:SS-HLS-Ax{ax:02d}{quad}:Level-Mon')
        return sensors_list

    def generate_HLS_data_to_dynamic_time_plot(self):
        try:
            data_df = self.data['hls_all'].copy()
        except KeyError:
            self.ui.logMessage('All HLS data needs to be fetched to run this analysis', severity='alert')
            return

        data_df.columns = data_df.columns.map(float)

        if (self.ui.filter_data):
            data_df = filter_timeseries_in_df(data_df, self.ui.filter_min, self.ui.filter_max)

        # referencing in one sensor
        sens_ref = 19
        data_df = data_df.sub(data_df.loc[:,sens_ref], axis=0)

        # --------- TEMPORARIO -------------
        # mapeando sensores do setor do anel para eixos do prédio
        # data_df.columns = [4,1,59,54,18,16,14,12,9,33,31,29,27,24,48,46,44,39]
        data_df.columns = [4,1,59,57,54,18,16,14,12,9,33,31,29,27,24,48,46,44,42,39]

        # sorting sensors
        data_df = data_df.sort_index(axis=1)

        # calculating bestfit between 2 epochs and applying shift in curve
        for i in range(1, len(data_df.iloc[:,0])):
            offset = calc_offset(data_df.iloc[i,:].values, data_df.iloc[i-1,:].values)
            data_df.iloc[i,:] += offset[0]

        # setting first record as time reference
        data_df = data_df - data_df.iloc[0,:]

        return data_df


    def clean_loaded_data(self):
        """ clean all the app level variables """

        # app level variables
        self.data = {}
        self.loaded_pvs = []
        # ui
        self.ui.txt_loadedPvs.clear()
        self.ui.btn_plot.setEnabled(False)
        self.ui.btn_makeVideo.setEnabled(False)
        self.ui.label_dataLoaded.setStyleSheet("background-color:rgb(255, 99, 101);color:rgb(0, 0, 0);padding:3;")
        self.ui.label_dataLoaded.setText("No data loaded")

    def add_loaded_pv(self, pvs):
        for pv in pvs:
            self.loaded_pvs.append(pv)
            self.ui.update_pvs_loaded(pv)

    def pvs_combinations(self):
        return {'hls_all': self.generate_all_sensors_list(),
                'rf': [self.RFFREQ_PV],
                'well': [self.WELLPRESSURE_PV],
                'tides': self.EARTHTIDES_PVS_LIST,
                'hls_oposite': self.HLS_OPOSITE_PVS_LIST,
                'select': [self.ui.ui.inputTxt_pvs.text()]}

    def treat_raw_data(self, data: pd.DataFrame, pv_option: str):
        if pv_option == 'hls_all':
            data.columns = self.HLS_LEGEND
        elif pv_option == 'hls_oposite':
            data['HLS Easth-West'] = data.iloc[:,0] - data.iloc[:,1]
            data['HLS North-South'] = data.iloc[:,2] - data.iloc[:,3]
            data.drop(columns=self.HLS_OPOSITE_PVS_LIST, inplace=True)

        return data

    @asyncSlot()
    async def on_btn_fetchFromArchiver_clicked(self):
        """ fetch data from Archiver according to user's selections """

        self.ui.logMessage('Fetching from Archiver...')

        # get datetime from ui
        timespam = self.ui.get_timespam_formatted()
        
        # define pvs list
        pv_option = self.ui.get_pv_option()
        pvs = self.pvs_combinations()[pv_option]

        try:
            # retrieving raw data from Archiver
            json_data = await self.retrieve_data(pvs, timespam['init'], timespam['end'], self.ui.optimize, self.ui.time_in_minutes)

            # mapping pv's values and timestamps
            data = [np.array(list(map(lambda i: i['val'], serie))) for serie in map(lambda j: j[0]['data'], json_data)]
            time_fmt = list(map(lambda data: datetime.fromtimestamp(data['secs']), json_data[0][0]['data']))
            # time_fmt = list(map(lambda data: datetime.fromtimestamp(data['secs']).strftime("%d.%m.%y %H:%M"), json_data[0][0]['data']))

            # creating pandas dataframe object
            d = {'datetime': time_fmt}
            for l_data, name in zip(data, pvs):
                d[name] = l_data
            data = DataFrame(data=d)

            # indexing by datetime
            data.reset_index(drop=True, inplace=True)
            data = data.set_index('datetime')

            # checking if special treatment is needed
            data = self.treat_raw_data(data, pv_option)

            # saving to app level variable
            self.data[pv_option] = data

        except IndexError:
            self.ui.logMessage('No data retrieved for this timespam', 'danger')
        else:
            self.ui.logMessage('Fetched!', 'success')
        finally:
            self.ui.enable_actions()
            self.add_loaded_pv(pvs)


    def static_plot_data(self) -> Dict[str, pd.DataFrame]:
        plot_data = {}

        for data_ref in self.data:
            plot_data[data_ref] = filter_timeseries_in_df(self.data[data_ref], self.ui.filter_min, self.ui.filter_max) if self.ui.filter_data else self.data[data_ref]
        
        return plot_data

    def on_btn_plot_clicked(self):
        """ plots data based on user's selections """

        plot_type = self.ui.get_plot_type()

        if plot_type['analysis'] == 'correlation':
            plot_cross_correl()
        elif plot_type['analysis'] == 'directional':
            plot_directional()
        elif plot_type['analysis'] == 'fft':
            if plot_type['is_static']:
                plot_fft_static()
            else:
                plot_fft_dynamic()
        else:
            if plot_type['is_static']:
                plot_data = self.static_plot_data()
                plot_data_2D_static(plot_data)
            else:
                # arranging data structure
                plot_data = self.generate_HLS_data_to_dynamic_time_plot()
                if plot_type['is_2d']:
                    plot_data_2D(plot_data, self.ui.save_fig, self.ui.figures_path)
                else:
                    plot_data_3D(plot_data)
                

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
