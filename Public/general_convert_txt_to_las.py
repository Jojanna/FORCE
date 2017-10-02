import lasio
import pandas as pd

## takes tab delimited text files with filename in following format: root\well_scenario.txt
#outputs a las file with same name to same directory
#column names and units must be specified in the order they appear in the text file
#header lines must be specified
#requires pandas and lasio libraries

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\INI'
scenarios = [""]
wells = ["21_24-4","21_24-5","21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]
header_lines = 1

column_names = ["Md", "Vp", "Vs", "RhoB", "PhiE", "PhiT", "Vsh", "Sw"]
column_units = ["ft", "m/s", "m/s", "g/cc", "frac", "frac", "frac", "frac"]

def data_load(filepath,header_lines=None, column_names=None, null_value=None):
    input = pd.read_table(filepath, index_col=False, sep=('\t'), delim_whitespace=True, header=header_lines,
                          names=column_names, lineterminator='\n', na_values=null_value)
    return input

filepaths = []
for well in wells:
    for scenario in scenarios:
        path = root + "\\" + well + scenario + ".txt"
        output = root + "\\" + well + scenario + ".las"
        filepaths.append(path)
        logs = data_load(path, header_lines = header_lines, column_names = column_names)

        with open(output, mode = 'w') as lasfile:
            las = lasio.LASFile()
            las.depth = ["Md"]
            las.well["WELL"].value = str(well)
            for log, name, unit in zip(logs, column_names, column_units):
                las.add_curve(name, logs[name].tolist(), unit = unit)
            las.write(lasfile, version=2, fmt='%10.5g')


