
import lasio
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\INI'

wells = ["21_24-1"]
scenarios = ["05GAS", "70GAS", "95GAS", "100WTR"]
null = -999.25

"""
# import list of las files
# create dataframe
--> ideally convert md to vertical depth using deviation... scope to do this with agile's code?
# well / depth  / Vp / Vs / RhoB / PhiE / Vsh / Sw

--> clustering, unsupervised?

"""

def data_load(filename):
    data = lasio.read(filename)
    logs = []
    units = []

    for curve in data.curves:
        logs.append(curve.mnemonic)
        units.append(curve.unit)

    units_dict = dict(zip(logs, units))

    logs_df = pd.DataFrame()
    for log in logs:
        logs_df[str(log)] = np.where(data[str(log)] == null, np.nan, data[str(log)])

    return logs_df, units_dict

filepaths = []

well_data = pd.DataFrame()

for well in wells:
    for scenario in scenarios:
        path = root + "\\" + well + "_" + scenario + ".las"
        filepaths.append(path)
        data, units_dict = data_load(path)
        data["Well_Name"] = well
        data["Scenario"] = scenario
        if well_data.empty == True:
            #print ("True")
            well_data = data
            #print (well_data)
        else:
            well_data = pd.merge(well_data, data, how = "outer")

print (well_data.head)

well_data["Mu"] = pow(well_data["Vs"], 2) * well_data["RhoB"] * 1000
well_data["K"] = pow(well_data["Vp"], 2) * well_data["RhoB"] * 1000 - 4/3 * well_data["Mu"]

data_subset = well_data[["Vp", "RhoB"]]

kmeans = KMeans(n_clusters=2).fit(data_subset)

labels = kmeans.labels_
print (labels)
data_subset["labels"] = labels
print (data_subset)
#results = pd.DataFrame([dataset.index,labels]).T

fig1 = plt.scatter(data_subset["Vp"], data_subset["RhoB"], c = labels)

plt.show()




