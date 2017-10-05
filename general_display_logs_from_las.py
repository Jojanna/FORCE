import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## takes las files with filename in following format: root\well_scenario.las and displays as a chart for quick viewing
# if chart = True, the output graph will be saved with filename: root\well_scenario.png
# use the null value to prevent these values being plotted

root = r"C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\INI"
scenarios = [""]
wells = ["21_24-1"]
null = -999.25
depth_column = "Md"
depth_range = [[6775, 6975]]

chart = True

filepaths = []

for well in wells:
    for scenario in scenarios:
        path = root + "\\" + well + scenario + ".las"
        filepaths.append(path)

        data = lasio.read(path)
        logs = []
        units = []
        for curve in data.curves:
            logs.append(curve.mnemonic)
            units.append(curve.unit)


        logs_df = pd.DataFrame()

        for log in logs:
            logs_df[str(log)] = data[str(log)]
            logs_df[str(log)] = np.where(logs_df[str(log)] == null, np.nan, logs_df[str(log)])
        print(logs)
        logs.remove(str(depth_column))
        md_unit = units[0]
        del units[0]

        md = logs_df[str(depth_column)].tolist()
        ax = np.arange(1, (len(logs)), 1)
        ax_ref = list(map(lambda x: "ax" + "_" + str(x), ax))

        idx = wells.index(well)
        range = depth_range[idx]

        fig1, ((ax_ref)) = plt.subplots(1, len(logs))
        fig1.set_size_inches(len(logs) * 3, 12.3)
        size = 8
        plt.title(str(well), ha="center")
        for log, ax_n, unit in zip(logs, ax_ref, units):

            ax_n.plot(logs_df[log].tolist(), md)
            ax_n.set_xlabel(str(log) + " " + str(unit), fontsize = size)
            ax_n.set_ylabel("Depth" + " " + md_unit, fontsize = size)
            ax_n.set_axisbelow(True)
            ax_n.grid()
            ax_n.invert_yaxis()
            ax_n.tick_params(axis='both', which='major', labelsize=6)
            ax_n.set_ylim(range[0], range[1])


        plt.tight_layout()

        if chart == True:
            chart_name = root + "\\" + well + scenario + ".png"
            fig1.savefig(chart_name, dpi = 400)
        plt.show()

