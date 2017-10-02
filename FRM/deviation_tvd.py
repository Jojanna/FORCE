
# This module is shamelessly bastardised from Agile Geoscience's welly\location module.
# from welly.location import compute_position_log: https://github.com/agile-geoscience
# I claim no copyright; any credit must go to Agile Geoscience. All mistakes, I own up to!

#copyright: 2016 Agile Geoscience
#license: Apache 2.0

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import lasio
import sys

# Load positioning info
position_filepath = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\Position_data.txt'
# Tell the module where to find the XY co-ords (in metres), the water depth, KB and TD
# (these should be in the same units as the MD in your las file)
x_col = "Well X (m)"
y_col = "Well Y (m)"
wd_col = "WD (ft)" # water depth from MSL to Seabed
kb_col = "KB Elev. (ft)" #kb from the Kelly Bushing to MSL
td_col = "TD (ft)"

#load deviation
# module expects a 3 column tab delimited file containing md/incl/azimuth, sorted by depth
root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\dev'
wells = ["21_24-1"]#,"21_24-4","21_24-6", "21_24-7", "21_25-10", "21_25-8","21_25-9"]
suffix = "_dev.txt"
header = 1
#md_col = 0
#inc_col = 1
#azi_col = 2
depth_units = "feet"
output_sampling = 0.5
start = 0 #start depth

## Use only if well is vertical - uses well position, TD and datums to create depth logs
vert = False
vert_td = 9590.00

# loading well positions IN METRES, EASTING AND NORTHING
positions = pd.read_csv(position_filepath, delimiter = "\t", header = 0, index_col = 0)
positions.index.name = "Well_Name"
positions.index = positions.index.str.replace('/', '_')
#print (positions)

def compute_position_log(deviation, method = 'mc', td = None):
    """
    Args:
        deviation (ndarray): A deviation survey with rows like MD, INC, AZI
        td (Number): The TD of the well, if not the end of the deviation
            survey you're passing.
        method (str):
            'aa': average angle
            'bt': balanced tangential
            'mc': minimum curvature
        update_deviation: This function makes some adjustments to the dev-
            iation survey, to account for the surface and TD. If you do not
            want to change the stored deviation survey, set to False.

    Returns:
        ndarray. A position log with rows like X-offset, Y-offset, Z-offset
    """

    # Adjust to TD.
    if td is not None:
        last_row = np.copy(deviation[-1, :])
        last_row[0] = td
        deviation = np.vstack([deviation, last_row])

    # Adjust to surface if necessary.
    if deviation[0, 0] > 0:
        deviation = np.vstack([np.array([0, 0, 0]), deviation])

    last = deviation[:-1]
    this = deviation[1:]

    diff = this[:, 0] - last[:, 0]

    Ia, Aa = np.radians(last[:, 1]), np.radians(last[:, 2])
    Ib, Ab = np.radians(this[:, 1]), np.radians(this[:, 2])

    if method == 'aa':
        Iavg = (Ia + Ib) / 2
        Aavg = (Aa + Ab) / 2
        delta_N = diff * np.sin(Iavg) * np.cos(Aavg)
        delta_E = diff * np.sin(Iavg) * np.sin(Aavg)
        delta_V = diff * np.cos(Iavg)
    elif method in ('bt', 'mc'):
        delta_N = 0.5 * diff * np.sin(Ia) * np.cos(Aa)
        delta_N += 0.5 * diff * np.sin(Ib) * np.cos(Ab)
        delta_E = 0.5 * diff * np.sin(Ia) * np.sin(Aa)
        delta_E += 0.5 * diff * np.sin(Ib) * np.sin(Ab)
        delta_V = 0.5 * diff * np.cos(Ia)
        delta_V += 0.5 * diff * np.cos(Ib)
    else:
        raise Exception("Method must be one of 'aa', 'bt', 'mc'")

    if method == 'mc':
        _x = np.sin(Ib) * (1 - np.cos(Ab - Aa))
        dogleg = np.arccos(np.cos(Ib - Ia) - np.sin(Ia) * _x)
        dogleg[dogleg == 0] = 1e-9
        rf = 2 / dogleg * np.tan(dogleg / 2)  # ratio factor
        rf[np.isnan(rf)] = 1  # Adjust for NaN.
        delta_N *= rf
        delta_E *= rf
        delta_V *= rf

    # Prepare the output array.
    result = np.zeros_like(deviation, dtype=np.float)

    # Stack the results, add the surface.
    _offsets = np.squeeze(np.dstack([delta_N, delta_E, delta_V]))
    _offsets = np.vstack([np.array([0, 0, 0]), _offsets])
    result += _offsets.cumsum(axis=0)

    #print (_offsets)


    return deviation, result, _offsets

def interpolate_dev(wells, positions, x_col, y_col, wd_col, kb_col, td_col, root, suffix, depth_units, header):

    for well in wells:
        td = positions[td_col].loc[well].astype(float)

        if vert == False:
            dev_filepath = root + "\\" + well + suffix
            dev = np.genfromtxt(dev_filepath, delimiter="\t") #, autostrip = True, usecols=[md_col, inc_col, azi_col])

            dev = dev[header:]
            if dev[-1, 0] < td:
                dev = np.vstack([dev, [td, 0, 0]])

            #print(dev)

            if depth_units == "feet":
                dev[:,0] = dev[:,0] * 0.3048


        if vert == True:
            dev = np.array([[start,0,0],[vert_td,0 ,0]])
            if depth_units == "feet":
                dev[:, 0] = dev[:, 0] * 0.3048

        #print (dev)
        deviation, result, _offsets = compute_position_log(dev, method = 'mc', td = None)
        deviation_df = pd.DataFrame(deviation, columns = ["Md m", "Inc deg", "Azimuth deg"])#, columns = )
        deviation_df["Md ft"] = deviation_df["Md m"] / 0.3048
        #print (deviation["Md"].tolist()) #= 0.3048 * deviation["Md"].to_list()
        result_df = pd.DataFrame(result, columns = ["delta_N m", "delta_E m", "delta_V m"])
        result_df["delta_V ft"] = result_df["delta_V m"] / 0.3048

        offsets_df = pd.DataFrame(_offsets, columns = ["delta_N_offset", "delta_E_offset", "delta_V_offset"])

        result_df = pd.concat([deviation_df, result_df, offsets_df], axis = 1)
        well_x = positions[x_col].loc[well]
        well_y = positions[y_col].loc[well]

        result_df["X Co-ord"] = well_x + result_df["delta_E m"]
        result_df["Y Co-ord"] = well_y + result_df["delta_N m"]

        depth_log = pd.DataFrame()
        wd = positions[wd_col].loc[well]
        kb = positions[kb_col].loc[well]

        if depth_units == "feet":
            depth_log["Md ft"] = (np.arange(start * pow(10, 6), (td + 1) * pow(10, 6), int(output_sampling * pow(10, 6))))/ pow(10, 6)
            depth_log["Md m"] = depth_log["Md ft"] * 0.3048
            wd = wd * 0.3048
            kb = kb * 0.3048
        elif depth_units == "metres":
            depth_log["Md m"] = (np.arange(start * pow(10, 6), (td + 1) * pow(10, 6), int(output_sampling * pow(10, 6)))) / pow(10, 6)
            depth_log["Md ft"] = depth_log["Md m"] / 0.3048

        f = interp1d(result_df["Md m"], result_df["delta_V m"], kind = "linear", fill_value = "extrapolate", assume_sorted=True)
        depth_log["TVDKB m"] = f(depth_log["Md m"])
        depth_log["TVDKB ft"] = depth_log["TVDKB m"] / 0.3048

        depth_log["TVDSS m"] =  depth_log["TVDKB m"] - kb
        depth_log["TVDML m"] = depth_log["TVDSS m"] - wd
        depth_log["TVDSS ft"] = depth_log["TVDSS m"] / 0.3048
        depth_log["TVDML ft"] = depth_log["TVDML m"] / 0.3048

        log_list = depth_log.columns.tolist() #[:, 1]
        log_units = list(map(lambda x: x.split(" ")[-1], log_list))

        #print (result_df)
        #print (depth_log) # seems to be delta_N, delta_E, delta_V, these deltas seem to be CUMULATIVE from the top of the well
        output_deviation = root + "\\" + well + "_dev_FULL" + ".txt"
        result_df.to_csv(output_deviation, sep = "\t", index = False)


        output_md2tvd = root + "\\" + well + "_md2tvd" + ".las"
        with open(output_md2tvd, mode = "w") as lasfile:
            las = lasio.LASFile()
            if depth_units == "feet":
                las.depth = ["Md ft"]
            elif depth_units == "metres":
                las.depth = ["Md m"]
            las.well["WELL"].value = str(well)

            for log, units in zip(log_list, log_units):
                las.add_curve(log, depth_log[log].tolist(), unit = units)
            las.write(lasfile, version = 2, fmt = "%10.9g")

    return (result_df, depth_log, output_md2tvd)

#interpolate_dev(wells, positions, x_col, y_col, wd_col, kb_col, root, suffix, depth_units, header)