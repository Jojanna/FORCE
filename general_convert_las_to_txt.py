import lasio
import pandas as pd
import numpy as np

## takes las files with filename in following format: root\well_scenario.las
#outputs a tab delimited file with same name to same directory
#requires pandas, numpy and lasio libraries
#multiple wells and scenarios can be specified and converted in a single run

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\FRM\TVD'
scenarios = ["_100WTR"]
wells = ["21_24-1", "21_24-4","21_24-5","21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]
null = -999.25

filepaths = []

for well in wells:
    for scenario in scenarios:
        path = root + "\\" + well + scenario + ".las"
        output = root + "\\" + well + scenario + ".txt"
        filepaths.append(path)

        data = lasio.read(path)
        logs = []
        units = []
        for curve in data.curves:
            logs.append(curve.mnemonic)
            units.append(curve.unit)
        log_header = "\t".join(logs)
        units_header = "\t".join(units)
        header = [log_header, units_header]
        header = "\n".join(header)


        logs_df = pd.DataFrame()

        for log in logs:
            logs_df[str(log)] = data[str(log)]
            logs_df[str(log)] = np.where(logs_df[str(log)] == null, np.nan, logs_df[str(log)])

        np.savetxt(output, logs_df.values, delimiter = '\t', header = header)


