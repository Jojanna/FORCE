import numpy as np
import pandas as pd
import lasio

from Catcher_markers import markers

root_deviation = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\dev'
suffix_deviation = "_md2tvd.las"
sample_interval = 0.5
depth_column = "Md ft"

wells = ["21_24-1"]#, "21_24-4", "21_24-5", "21_24-6", "21_24-7", "21_25-10", "21_25-8","21_25-9"]

markers = pd.DataFrame.from_dict(markers).transpose()
markers.index.name = "Well_Name"
markers_tvdml = pd.DataFrame(index = wells, columns=list(markers.columns))
markers_tvdml.index.name = "Well_Name"
print (markers)

for well in wells:
    deviation = lasio.read(root_deviation + "\\" + well + suffix_deviation)
    deviation = deviation.df()
    #print (deviation)



    for marker_name in list(markers.columns):
        markers[marker_name] = list(map(lambda x: round(x * (1/sample_interval)) / (1/sample_interval), markers[marker_name]))
        if np.isnan(markers[marker_name].loc[well]):
            pass
        else:
            md = markers[marker_name].loc[well]
            tvdml = (deviation["TVDML ft"].loc[deviation.index == md])#.tolist()
            markers_tvdml[marker_name].loc[well] = tvdml.iloc[0]#.item()

print (markers_tvdml)