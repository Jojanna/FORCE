import lasio
import pandas as pd
import numpy as np

# root for main las file
root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'
wells = ["WELL"] #
scenarios = ["_100WTR","_05OIL", "_70OIL", "_95OIL"]#, "_05GAS", "_70GAS", "_95GAS"]
depth_column = "DEPTH" #this will be the column that the files are merged on. These files should be sampled at the same intervals.

# root for las file to be merged with main las
merge_root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'
merge_suffix = "_md2tvd"
merge_depth_col = "Md ft" #this will be the column that the files are merged on. These files should be sampled at the same intervals.

#root for output las file
out_root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'

# Null value
null = -999.25


def merge_las(root, wells, scenarios, depth_column, merge_root, merge_suffix, merge_depth_col, out_root, null):

    filepaths = []
    output_filenames = []

    for well in wells:

        # load merge file

        logs_dict = {}
        merge_path = merge_root + "\\" + well + merge_suffix + ".las"

        data = lasio.read(merge_path)
        logs_merge = []
        units_merge = []
        for curve in data.curves:
            logs_merge.append(curve.mnemonic)
            units_merge.append(curve.unit)
            logs_dict[curve.mnemonic] = curve.unit

        merge_df = pd.DataFrame()

        for log in logs_merge:
            merge_df[str(log)] = data[str(log)]
            merge_df[str(log)] = np.where(merge_df[str(log)] == null, np.nan, merge_df[str(log)])

        merge_df.rename(columns = {merge_depth_col: depth_column}, inplace=True)

        # load main las file
        for scenario in scenarios:
            path = root + "\\" + well + scenario + ".las"
            filepaths.append(path)

            data_main = lasio.read(path)
            logs_main = []
            units_main = []
            for curve in data_main.curves:
                logs_main.append(curve.mnemonic)
                units_main.append(curve.unit)
                logs_dict[curve.mnemonic] = curve.unit

            main_df = pd.DataFrame()

            for log in logs_main:
                main_df[str(log)] = data_main[str(log)]
                main_df[str(log)] = np.where(main_df[str(log)] == null, np.nan, main_df[str(log)])

            output = pd.merge(main_df, merge_df, on = depth_column, how = "outer")
            output_file = out_root + "\\" + well + scenario + ".las"
            output_filenames.append(output_file)

            output.dropna(how = "all", axis = 0, inplace = True)
            output.sort_values(by = depth_column, inplace = True)

            with open(output_file, mode = 'w') as lasfile:
                las = lasio.LASFile()
                las.depth = [depth_column]
                las.well["WELL"].value = str(well)
                las.well["NULL"].value = null
                for log in list(output.columns.values):
                    las.add_curve(log, output[log], unit = logs_dict.get(log))
                las.write(lasfile, version=2, fmt = "%10.10g")

    return output_filenames

merge_las(root, wells, scenarios, depth_column, merge_root, merge_suffix, merge_depth_col, out_root, null)







