import lasio
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandas.plotting import scatter_matrix

plt.rcParams["font.size"] = 11
plt.rcParams["figure.figsize"] = (8., 6.)
plt.rcParams["figure.dpi"] = 400
plt.rcParams["lines.markersize"] = 6
plt.rcParams["lines.markeredgewidth"] = 0.1
plt.rcParams['patch.edgecolor'] = 'k'
plt.rcParams['patch.linewidth'] = 0.1

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\FRM'

wells = ["21_24-1","21_24-4","21_24-5","21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]
water_scenarios = ["100WTR"]
hc_scenarios = ["05GAS", "70GAS", "95GAS","05OIL", "70OIL", "95OIL"]
phase = ["gas", "gas", "gas", "oil", "oil", "oil"]
hc_scenario_num = np.arange(len(hc_scenarios)) + 1


null = -999.25

rebuild_database = True
construct_graph = False

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

if rebuild_database == True:

    hc_filepaths = []

    well_data_wtr = pd.read_csv(root + "\\" + "All_wells_100WTR_labels.txt", sep = "\t")
    well_data_wtr["Phase"] = "water"
    well_data = well_data_wtr.copy()
    clusters = np.unique(well_data_wtr["labels"])
    num_clusters = len(clusters)
    label_add = num_clusters * hc_scenario_num

    for well in wells:

        for scenario, label_add_n, phase_n in zip(hc_scenarios, label_add, phase):
            path = root + "\\" + well + "_" + scenario + ".las"
            hc_filepaths.append(path)
            data, hc_units_dict = data_load(path)

            well_data_wtr_well = well_data_wtr.loc[well_data_wtr["Well_Name"] == well]
            common_cols = ["Md", "Bulk Modulus", "Shear Modulus", "Vp/Vs", "AI", "SI", "Poisson's Ratio", "Lambda-Rho", "Mu-Rho", "Vp", "Vs", "RhoB", "PhiE", "PhiT", "Vsh"]
            data_exc = pd.merge(well_data_wtr_well, data, on = common_cols, how = "inner").copy()
            data_inc = data[~data["Md"].isin(data_exc["Md"])].copy()
            well_data_label = well_data_wtr_well[["Md", "labels"]]
            data_inc = data_inc.merge(well_data_label, on = ["Md"],  how = "left")
            data_inc["Well_Name"] = well
            data_inc["Scenario"] = scenario
            data_inc["labels"] = data_inc["labels"].astype(float) + label_add_n
            data_inc["Phase"] = phase_n

            well_data = pd.merge(well_data, data_inc, how="outer")

        print ("Complete: %s" % well)

    well_data.to_csv(root + "\\" + "All_wells_labels.txt", sep = "\t", index = False)

if rebuild_database == False:
    well_data = pd.read_csv(root + "\\" + "All_wells_labels.txt", sep = "\t")

if construct_graph == True:

    data_subset = well_data[["Vp", "RhoB","Vsh", "PhiE"]]
    cmap = plt.cm.get_cmap('gist_ncar')
    clusters = np.unique(well_data["labels"])
    norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))
    fig2 = scatter_matrix(data_subset, c = well_data["labels"], cmap = cmap, norm = norm, figsize=[10, 10], diagonal='hist', hist_kwds = {"edgecolor":'k', "linewidth":0.5}, edgecolor = 'k')#, edgewidth = 0.05)
    #fig2.set_size_inches(18.5, 10.5)
    handles = [plt.plot([],[],color=cmap(i/(len(clusters)-1)), ls="", marker=".", markersize=np.sqrt(10))[0] for i in np.arange(len(clusters))]
    plt.legend(handles, clusters, loc=(1.02, 0), title = "Classes", fontsize = 'small', ncol = 2 )
    plt.savefig(root + "\\" + "Scatter_Matrix_all_FRM_Clusters", dpi = 400)
    plt.show()