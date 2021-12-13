from functools import partial
import sys
from datetime import datetime
import os
from typing import Dict, List
import asyncio
from itertools import combinations

import qasync
from qasync import asyncSlot, QApplication

import numpy as np
import pandas as pd
from pandas import DataFrame
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from PyQt5.QtWidgets import QMainWindow, QWidget

from archiver import retrieve_data
from calculations.fft import calculate_fft
from calculations.geometrical import calc_offset
from calculations.timeseries import filter_timeseries_in_df, slice_timeseries_data
from plot import plot_cross_correl, plot_data_2D, plot_data_2D3D, plot_data_2D_static, plot_data_3D, plot_directional, plot_fft_dynamic, plot_fft_static
from ui import Ui_MainWindow
from ui.handler import Ui_handler
import config

class App(QWidget):  
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
        
        if not self.ui.figures_path:
            self.ui.logMessage(f'Video making failed: no directory indicated.', severity='danger')
            return

        image_folder='../data/output/' + self.ui.figures_path
        
        try:
            image_files = [image_folder+'/'+img for img in os.listdir(image_folder) if img.endswith(".png")]
        except FileNotFoundError:
            self.ui.logMessage(f'Video making failed: directory {image_folder} not found.', severity='danger')
            return

        image_files.sort(key=lambda i: int(i.split('/')[-1].split('-')[-1].split('.')[0]))

        fps_movie=12
        clip = ImageSequenceClip(image_files, fps=fps_movie)
        clip.write_videofile(image_folder+'/'+image_folder.split('/')[-1]+'.mp4')

    def generate_all_sensors_list (self):
        """ utility function to create a list with all the current HLS PVs names """

        sectors = [17, 16, 15, 14, 13, 1, 1, 20, 19, 18, 6, 6, 5, 4, 3, 11, 11, 10, 9, 8]
        axis = [4, 1, 59, 57, 54, 18, 16, 14, 12, 9, 33, 31, 29, 27, 24, 48, 46, 44, 42, 39]
        quadrant = ['NE5', 'NE4', 'NE3', 'NE2', 'NE1', 'SE5', 'SE4', 'SE3', 'SE2', 'SE1', 'SW5', 'SW4', 'SW3', 'SW2', 'SW1', 'NW5', 'NW4', 'NW3', 'NW2', 'NW1']
        sensors_list = []
        for sector, ax, quad in zip(sectors, axis, quadrant):
            sensors_list.append(f'TU-{sector:02d}C:SS-HLS-Ax{ax:02d}{quad}:Level-Mon')
        return sensors_list

    def find_axis_ref_in_hls_sensors_names(self, pvs_names: list) -> list:
        return pvs_names.map(lambda name: name[name.index('Ax') + 2: name.index('Ax') + 4])

    def generate_HLS_data_to_dynamic_time_plot(self, sensor_ref: str = 'sector'):
        try:
            data_df = self.data['hls_all'].copy()
        except KeyError:
            raise

        if (self.ui.filter_data):
            data_df = filter_timeseries_in_df(data_df, self.ui.filter_min, self.ui.filter_max)

        # # referencing in one arbitrary sensor
        # sens_ref = 19
        # data_df = data_df.sub(data_df.loc[:,sens_ref], axis=0)

        # mapping sensor positions to SR sectors or building axis
        if sensor_ref == 'sector':
            data_df.columns = self.HLS_MAPPING_SECTOR
        else:
            data_df.columns = self.find_axis_ref_in_hls_sensors_names(data_df.columns)

        data_df.columns = data_df.columns.map(float)

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
                'rf': [config.RFFREQ_PV],
                'well': [config.WELLPRESSURE_PV],
                'tides': config.EARTHTIDES_PVS_LIST,
                'hls_oposite': config.HLS_OPOSITE_PVS_LIST,
                'select': [self.ui.ui.inputTxt_pvs.text()]}

    def treat_hls_oposite_data(self, data: pd.DataFrame):

        # calculating the signal differences
        data['HLS Easth-West'] = data.iloc[:,0] - data.iloc[:,1]
        data['HLS North-South'] = data.iloc[:,2] - data.iloc[:,3]

        # ignoring each sensor's data
        data.drop(columns=self.HLS_OPOSITE_PVS_LIST, inplace=True)

        return data

    @asyncSlot()
    async def fetch_data_from_archiver(self):
        """ fetch data from Archiver according to user's selections """

        self.ui.logMessage('Fetching from Archiver...')

        # get datetime from ui
        timespam = self.ui.get_timespam_formatted()
        
        # define pvs list
        pv_option = self.ui.get_pv_option()
        pvs = self.pvs_combinations()[pv_option]

        try:
            # retrieving raw data from Archiver
            json_data = await retrieve_data(pvs, timespam['init'], timespam['end'], self.ui.optimize, self.ui.time_in_minutes)

            # mapping pv's values and timestamps
            data = [np.array(list(map(lambda i: i['val'], serie))) for serie in map(lambda j: j[0]['data'], json_data)]
            time_fmt = list(map(lambda data: datetime.fromtimestamp(data['secs']), json_data[0][0]['data']))

            # creating pandas dataframe object
            d = {'datetime': time_fmt}
            for l_data, name in zip(data, pvs):
                d[name] = l_data
            data = DataFrame(data=d)

            # indexing by datetime
            data.reset_index(drop=True, inplace=True)
            data = data.set_index('datetime')

            # "hls oposite" pvs choice leads to an early data treatment
            if pv_option == 'hls_oposite':
                data = self.treat_hls_oposite_data(data)

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
            if self.ui.filter_data:
                plot_data[data_ref] = filter_timeseries_in_df(self.data[data_ref], self.ui.filter_min, self.ui.filter_max)
            else:
                plot_data[data_ref] = self.data[data_ref]
        
        return plot_data

    def correlation_plot_data(self) -> List[dict]:
        # creating sliced (based on the input of 'time chuncks') dfs
        sliced_data_df = slice_timeseries_data(self.data, *self.ui.sliced_time_props)

        if len(sliced_data_df) <= 1:
            self.ui.logMessage('Correlation failed: more than one PV required.', 'danger')
            return None

        # creating timeserie lists from sliced dfs
        sliced_data = []
        for pv in sliced_data_df:
            sliced_data.append({'pv': pv,
                                'val': {'ts': [df.index[0] for df in sliced_data[pv]],
                                        'serie': np.array([df.to_numpy().reshape(1,df.size)[0] for df in sliced_data[pv]])}})

        # checking all possible combinations
        comb = list(combinations(np.arange(0,len(sliced_data)), 2))

        cross_corr_all = []
        # cross-correlation calculation upon time series
        for comb_idx in comb:
            cross_corr = []
            ts = []
            if (len(sliced_data[comb_idx[0]]['val']['serie']) != len(sliced_data[comb_idx[1]]['val']['serie'])):
                self.ui.logMessage(f'Not evaluating correlation between {sliced_data[comb_idx[0]]["pv"]} and {sliced_data[comb_idx[1]]["pv"]}: divergent array lenghts ({len(sliced_data[comb_idx[0]]["val"]["serie"])} and {len(sliced_data[comb_idx[1]]["val"]["serie"])})', 'alert')
                continue

            if (sliced_data[comb_idx[0]]['val']['ts'] != sliced_data[comb_idx[1]]['val']['ts']):
                self.ui.logMessage(f'Not evaluating correlation between {sliced_data[comb_idx[0]]["pv"]} and {sliced_data[comb_idx[1]]["pv"]}: datetimes not coincident', 'alert')
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

        return cross_corr_all

    def check_output_path(self):
        if not self.ui.figures_path:
            dir_name = datetime.now().strftime('%d%m%y%H%M')
            self.ui.logMessage(f'No directory name provided. Saving movie in dir {dir_name}')
        else:
            dir_name = self.ui.figures_path

    def plot_data(self):
        """ plots data based on user's selections """

        plot_type = self.ui.get_plot_type()

        if plot_type['analysis'] == 'correlation':
            cross_corr_data = self.correlation_plot_data()
            if cross_corr_data:
                plot_cross_correl(cross_corr_data)
        elif plot_type['analysis'] == 'directional':
            plot_directional(self.data, self.generate_all_sensors_list(), self.ui)
        elif plot_type['analysis'] == 'fft':
            if plot_type['is_static']:
                # treating data for plotting
                fft_data = calculate_fft(self.data, self.ui.filter_data, [self.ui.filter_min, self.ui.filter_max])

                plot_fft_static(fft_data)
            else:
                plot_fft_dynamic(self.data, self.ui.filter_data, [self.ui.filter_min, self.ui.filter_max])
        else:
            if plot_type['is_static']:
                # treating data for plotting
                plot_data = self.static_plot_data()

                plot_data_2D_static(plot_data)
            else:
                # defining sensors position reference (hard-coded, but an UI option would be better)
                sensor_pos_ref = 'sector'
                try:
                    # treating data for plotting
                    plot_data = self.generate_HLS_data_to_dynamic_time_plot(sensor_pos_ref)
                except KeyError:
                    self.ui.logMessage('All HLS data needs to be fetched to run this analysis', severity='alert')
                else:
                    if plot_type['is_2d']:
                        if self.ui.save_fig:
                            dir_name = self.check_output_path()
                        plot_data_2D(plot_data, self.ui.save_fig, dir_name, sensor_pos_ref)
                    else:
                        dir_name = self.check_output_path()
                        
                        # here both ...3D and ...2D3D can be called
                        plot_data_2D3D(plot_data, dir_name, sensor_pos_ref)
                

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
