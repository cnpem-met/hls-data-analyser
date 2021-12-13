from functools import partial
import os
import time
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.interpolate import make_interp_spline
from scipy.signal import  spectrogram
from scipy.stats import pearsonr, spearmanr

from calculations.correlations import calc_cross_corr
from calculations.metrics import print_progress_bar
from calculations.timeseries import filter_timeserie
from plot.utils import PlotPickable

def plot_data_2D_static(plot_data):
    plot = PlotPickable()
    fig, ax = plot.get_plot_props()
    axs = [ax]

    if len(plot_data) > 1:
        fig.subplots_adjust(right=(1 - len(plot_data)*0.05))
        for i in range(len(plot_data) - 1):
            new_ax = ax.twinx()
            new_ax.spines["right"].set_position(("axes", 1 + (i)*0.1))
            axs.append(new_ax)

    lines, legends = [], []
    for i, data in enumerate(plot_data.values()):
        line = axs[i].plot(data.index, data.iloc[:,:]) # colors are repeating from one df to another...
        [legends.append(var) for var in data.columns.values]
        [lines.append(l) for l in line]
        axs[i].set_ylabel(list(plot_data.keys())[i])
    
    leg = axs[0].legend(lines, legends)        
    plot.define_legend_items(leg, lines)
    plot.change_legend_alpha(leg)
    
    ax.yaxis.labelpad = 10
    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both')

    # for some reason, feature of picking in legend is not working when associated with multiple y axis...
    fig.canvas.mpl_connect('pick_event', partial(PlotPickable.on_pick, fig=fig, lined=plot.get_lined()))

    ax.grid()
    plt.show()

def plot_data_2D(plot_data: pd.DataFrame, save_fig: bool, dir_path: str, x_axis_ref: str = 'sector'):

    # setting plot style
    sns.set()
    sns.set_style('ticks')
    sns.set_palette(sns.light_palette("green", n_colors=len(plot_data.index.values)))

    # creating directory to save images
    dir_path = '../data/output/' + dir_path
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    
    # finding min and max limits for the y axis
    max_lim = 1.1 * max(plot_data.max())
    min_lim = 1.1 * min(plot_data.min())

    # smoothing curve (x axis)
    x = plot_data.columns.values
    x_smooth = np.linspace(min(x), max(x), 300)

    fig = plt.figure(figsize=(18,9))#(10,6)
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    i = 0
    num_data = len(plot_data.index)
    print_progress_bar(0, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for timestamp, values in plot_data.iterrows():

        # smoothing curve (y axis)
        a_BSpline = make_interp_spline(x, values, k=2)
        y_smooth = a_BSpline(x_smooth)

        ax.plot(x_smooth, y_smooth, color='limegreen')
        ax2.scatter(x, values, color='forestgreen', s=4)

        ax2.axes.get_yaxis().set_visible(False)
        ax.tick_params(axis='both', labelsize=15) #size=13
        ax.grid(True)

        if x_axis_ref == 'sector':
            ax.set_xlabel('Setor do Anel de Armazenamento', fontsize=17, labelpad=6)
            tickpos = np.linspace(1,20,20)
        else:
            ax.set_xlabel('Eixos do prédio', fontsize=17, labelpad=6) 
            tickpos = np.linspace(1,60,20)

        ax.set_xticks(tickpos)
        ax.set_xticklabels(tickpos)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax.set_ylim(min_lim, max_lim)
        ax2.set_ylim(min_lim, max_lim)

        # drawing box containing the datetime of the plotted serie
        text = ax.text(0.5, ax.get_ylim()[1]*0.85, timestamp, fontsize=18, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        ax.set_ylabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=17) #14
        
        
        # saving images if needed
        if (save_fig):
            plt.savefig(f"{dir_path}/hls-ax-{i}.png", dpi=150)
            ax.cla()
            
        if i % 50 == 0:
            print_progress_bar(i + 1, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)

        ax2.cla()
        i += 1

    if (not save_fig):
        plt.show()

def plot_data_3D(plot_data: pd.DataFrame, dir_path: str):
    fig = plt.figure(figsize=(9.8,7))
    ax = fig.add_subplot(111, projection='3d')
    
    # # old way
    # density = len(plot_data.columns.values)
    # repeat_times = 2
    # R = np.linspace(70, 90, repeat_times*(density+1))
    # u = np.linspace(0,  2*np.pi, repeat_times*(density+1))
    # x = np.outer(R, np.cos(u))
    # y = np.outer(R, np.sin(u))

    # # finding x and y coords of cut and fill line
    # x_cutfill = [x[-1][6*repeat_times], x[-1][17*repeat_times]]
    # y_cutfill = [y[-1][6*repeat_times], y[-1][17*repeat_times]]

    R_smooth = np.linspace(70, 90, 300)
    u_smooth = np.linspace(0,  2*np.pi, 300)
    x_smooth = np.outer(R_smooth, np.cos(u_smooth))
    y_smooth = np.outer(R_smooth, np.sin(u_smooth))

    # finding min and max limits for the y axis
    max_lim = 1.2 * max(plot_data.max())
    min_lim = 1.2 * min(plot_data.min())

    i=0
    size_plot_array = plot_data.index.size

    dir_path = '../data/output/' + dir_path
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

        # # old way: no smoothing, so more than one point is plotted for each value
        # # so the curve is composed of flat steps instead of triangles
        # z = np.outer(level, np.ones(repeat_times*(density+1))).T
        # surf = ax.plot_surface(x,y,z,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h
        # ax.plot(x_cutfill,y_cutfill, [level[6*repeat_times], level[17*repeat_times]])

        colorbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.13)

        ax.set_title("HLS @ Poço desligado", weight='semibold', fontsize=15, y=1.02, x=0.65)
        ax.set_xlabel('x [m]', fontsize=12)
        ax.set_zlabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=12, labelpad=7) #14
        ax.set_ylabel('y [m]', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        ax.view_init(15, 145)

        ax.set_zlim(min_lim, max_lim)
        surf.set_clim(min_lim, max_lim)

        # drawing box containing the datetime of the plotted serie
        ax.text(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[1]*1.25, timestamp, fontsize=13, bbox=dict(boxstyle='round', facecolor='white'))
    
        fig.tight_layout()
        plt.savefig(f"{dir_path}/hls-timeseries-{i}.png", dpi=150)
        ax.cla()
        colorbar.remove()

        i+=1

def plot_data_2D3D(plot_data: pd.DataFrame, dir_path: str, x_axis_ref: str = 'sector'):
    fig = plt.figure(figsize=(16,7))
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

    dir_path = '../data/output/' + dir_path
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    i=0
    num_data = len(plot_data.index)
    print_progress_bar(0, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)

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

        if x_axis_ref == 'sector':
            tickpos = np.linspace(1,20,20)
            ax_2d.set_xlabel('Setores do Anel de Armazenamento', fontsize=17, labelpad=6)
        else:
            tickpos = np.linspace(1,60,10)
            ax_2d.set_xlabel('Eixos do prédio', fontsize=17, labelpad=6)
        
        ax_2d.set_xticks(tickpos)
        ax_2d.set_xticklabels(tickpos)
        ax_2d.xaxis.set_major_formatter(FormatStrFormatter('%d'))

        ax_2d.set_ylim(min_lim_2d, max_lim_2d)
        ax_2d_2.set_ylim(min_lim_2d, max_lim_2d)

        # drawing box containing the datetime of the plotted serie
        _ = ax_2d.text(0.5, ax_2d.get_ylim()[1]*0.85, timestamp, fontsize=18, bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

        # ax_2d.set_title("HLS @ Poço ligado", weight='semibold', fontsize=18, y=1.05)
        ax_2d.set_ylabel(r'$\Delta \/ {Nível} \/ [mm]$', fontsize=17) #14

        ##### 3D stuff #####

        level = values.to_numpy()
        level = np.append(level, level[0])

        # smoothing curve
        a_BSpline = make_interp_spline(np.linspace(0, 20, len(level)), level)
        level_smooth = a_BSpline(np.linspace(0, 20, 300))
        z_smooth = np.outer(level_smooth, np.ones(300)).T
        surf = ax_3d.plot_surface(x_smooth,y_smooth,z_smooth,cmap='viridis', edgecolor='none') # z in case of disk which is parallel to XY plane is constant and you can directly use h

        colorbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.13)

        ax_3d.set_xlabel('x [m]', fontsize=12)
        ax_3d.set_ylabel('y [m]', fontsize=12)
        ax_3d.tick_params(axis='both', labelsize=10)
        

        ax_3d.view_init(15, 145)

        ax_3d.set_zlim(min_lim, max_lim)
        surf.set_clim(min_lim, max_lim)

        fig.tight_layout()
        plt.savefig(f"{dir_path}/hls-timeseries-{i}.png", dpi=150)
        ax_3d.cla()
        colorbar.remove()

        ax_2d.cla()
        ax_2d_2.cla()

        print_progress_bar(i + 1, num_data, prefix = 'Progress:', suffix = 'Complete', length = 50)
        i+=1

def plot_fft_static(fft_data: List[dict]):
    fig, ax = plt.subplots()
    for data in fft_data:
        ax.plot(data['xp'], data['yp'], label=data['var'])
    ax.legend()
    ax.set_xlabel('Period [h]')
    ax.set_ylabel('Power Density') #16
    plt.show()

def plot_cross_correl (data: List[dict]):
    # plotting bars containing time and frequency correlation information
    plot = PlotPickable()
    fig, ax = plot.get_plot_props()

    lines, pts = [], []
    for cross_corr in data:
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
    fig.canvas.mpl_connect('pick_event', partial(PlotPickable.on_pick, fig=fig, lined=plot.get_lined()))
    fig.tight_layout()
    plt.show()

def plot_fft_dynamic(data: Dict[str, pd.DataFrame], apply_filter: bool, filter_limits: list):

        for df in data.values():
            # only the first column of df will be used
            timeserie = df.iloc[:,0].values

            # applying filter if needed
            if apply_filter:
                timeserie = filter_timeserie(df, *filter_limits)

            # extracting frequency of acquisition
            ts1 = time.mktime(pd.to_datetime(df.index.values[0]).timetuple())
            ts2 = time.mktime(pd.to_datetime(df.index.values[1]).timetuple())
            fs = 1/(ts2 - ts1)

            # plotting spectogram
            f, t, Sxx = spectrogram(timeserie, fs)
            plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.title(df.columns[0])
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')

        plt.show()

def plot_directional(data: Dict[str, pd.DataFrame], hls_pvs: list, ui):
    # extracting data from HLS sensors and earth tides
    try:
        hls_df = data['hls_all']
        tides_df = data['tides']
        well_df = data['well']
        rf_df = data['rf']
    except KeyError:
        ui.logMessage('Directional analysis failed: it needs hls, tides, well and rf data.', severity='danger')
        return
    
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
            corr_cc = calc_cross_corr(level['val'], tide)
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
        corr_cc = calc_cross_corr(level['val'], tides)
        print(f'{level["var"]}\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
        hls_pairs_df = hls_pairs_df.append({'tide': 'SUM OF 3 TIDES', 'var': level['var'],\
                                            'pearson': "{:.2f}".format(abs(corr_p)),\
                                            'spearman': "{:.2f}".format(abs(corr_s)),\
                                            'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
    print('\n')
    for i, hls_pv in enumerate(hls_pvs):
        corr_p, _ = pearsonr(hls_df.iloc[:,i], tides)
        corr_s, _ = spearmanr(hls_df.iloc[:,i], tides)
        corr_cc = calc_cross_corr(hls_df.iloc[:,i], tides)
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
        corr_cc = calc_cross_corr(well, tide)
        print(f'Well\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}')
        well_rf_df = well_rf_df.append({'tide': tide_name, 'var': 'Well',\
                                        'pearson': "{:.2f}".format(abs(corr_p)),\
                                        'spearman': "{:.2f}".format(abs(corr_s)),\
                                        'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
        corr_p, _ = pearsonr(rf, tide)
        corr_s, _ = spearmanr(rf, tide)
        corr_cc = calc_cross_corr(rf, tide)
        print(f'RF\t{"{:.2f}".format(abs(corr_p))}\t{"{:.2f}".format(abs(corr_s))}\t{"{:.2f}".format(corr_cc)}\n')
        well_rf_df = well_rf_df.append({'tide': tide_name, 'var': 'RF',\
                                            'pearson': "{:.2f}".format(abs(corr_p)),\
                                            'spearman': "{:.2f}".format(abs(corr_s)),\
                                            'cross': "{:.2f}".format(abs(corr_cc))}, ignore_index=True)
        
    hls_pairs_df = hls_pairs_df.append(well_rf_df)
    hls_pairs_df.to_excel('../data/output/directional_analysis_results.xlsx')
        