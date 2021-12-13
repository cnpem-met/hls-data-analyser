import pandas as pd
import config

def treat_hls_oposite_data(data: pd.DataFrame) -> pd.DataFrame:
    # calculating the signal differences
    data['HLS Easth-West'] = data.iloc[:,0] - data.iloc[:,1]
    data['HLS North-South'] = data.iloc[:,2] - data.iloc[:,3]

    # ignoring each sensor's data
    data.drop(columns=config.HLS_OPOSITE_PVS_LIST, inplace=True)

    return data

def generate_hls_pvs () -> list:
    """ utility function to create a list with all the current HLS PVs names """

    sectors = [17, 16, 15, 14, 13, 1, 1, 20, 19, 18, 6, 6, 5, 4, 3, 11, 11, 10, 9, 8]
    axis = [4, 1, 59, 57, 54, 18, 16, 14, 12, 9, 33, 31, 29, 27, 24, 48, 46, 44, 42, 39]
    quadrant = ['NE5', 'NE4', 'NE3', 'NE2', 'NE1', 'SE5', 'SE4', 'SE3', 'SE2', 'SE1', 'SW5', 'SW4', 'SW3', 'SW2', 'SW1', 'NW5', 'NW4', 'NW3', 'NW2', 'NW1']
    sensors_list = []
    for sector, ax, quad in zip(sectors, axis, quadrant):
        sensors_list.append(f'TU-{sector:02d}C:SS-HLS-Ax{ax:02d}{quad}:Level-Mon')
    return sensors_list

def find_axis_ref_in_hls_sensors_names(pvs_names: list) -> list:
    return pvs_names.map(lambda name: name[name.index('Ax') + 2: name.index('Ax') + 4])