import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix



plt.rcParams["font.size"] = 11
#plt.rcParams["figure.figsize"] = (8., 6.)
#plt.rcParams["figure.dpi"] = 400
plt.rcParams["lines.markersize"] = 6
plt.rcParams["lines.markeredgewidth"] = 0.1
plt.rcParams['patch.edgecolor'] = 'k'
plt.rcParams['patch.linewidth'] = 0.1

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'

wells = ["WELL"]
water_scenarios = ["100WTR"]
hc_scenarios = ["05OIL", "70OIL", "95OIL"]
phase = ["oil", "oil", "oil"]
depth_col = "DEPTH"

# These columns are used to exclude duplicate datapoints i.e. where there is no fluid sub. Requires depth (md) as a minimum, ideally also Vp, Vs, RhoB, Vsh and PhiE,
common_cols = ["DEPTH", "Vp", "Vs", "RhoB", "Sw"]

clustered_cols = ["Vp", "RhoB","Vsh", "PhiE"]

# option to have a different class for points of the same facies cluster, but different fluid saturation/phase
# if True, the difference between classes of same facies but different fluid/saturation will be equal to the number of clusters initially identified
# e.g. 7 classes; 100% water class = 1, 95% oil = 8, 70% oil = 15
#True recommended to ensure equal sampling of different fluid saturations by the models
vary_class_w_sw = True

null = -999.25

rebuild_database = True
construct_graph = True

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

def build_modelling_database(rebuild_database, root, wells, hc_scenarios, phase, common_cols, depth_col, clustered_cols, vary_class_w_sw):

    hc_scenario_num = np.arange(len(hc_scenarios)) + 1
    if rebuild_database == True:

        hc_filepaths = []

        well_data_wtr = pd.read_csv(root + "\\" + "All_wells_100WTR_labels.txt", sep = "\t")
        well_data_wtr["Phase"] = "water"
        well_data = well_data_wtr.copy()
        clusters = np.unique(well_data_wtr["labels"])
        decimals = pd.Series([3] * len(common_cols), index = common_cols)
        well_data = well_data.round(decimals)
        num_clusters = len(clusters)
        if vary_class_w_sw == True:
            label_add = num_clusters * hc_scenario_num
        else:
            label_add = [0] * hc_scenario_num

        for well in wells:

            for scenario, label_add_n, phase_n in zip(hc_scenarios, label_add, phase):
                path = root + "\\" + well + "_" + scenario + ".las"
                hc_filepaths.append(path)
                data, hc_units_dict = data_load(path)
                data.dropna(subset = [clustered_cols], inplace = True, how = "any", axis = 0)
                data = data.round(decimals)

                well_data_wtr_well = well_data_wtr.loc[well_data_wtr["Well_Name"] == well]
                well_data_label = well_data_wtr_well[[depth_col, "labels"]]
                data = data.merge(well_data_label, on=[depth_col], how="left")
                data["Well_Name"] = well
                data["Scenario"] = scenario
                data["labels"] = data["labels"].astype(float) + label_add_n
                data["Phase"] = phase_n

                well_data = pd.merge(well_data, data, how="outer")
                well_data.drop_duplicates(subset=common_cols, inplace=True, keep="first")

            print ("Complete: %s" % well)


        well_data.sort_values(by=depth_col, inplace=True)

        well_data.to_csv(root + "\\" + "All_wells_labels.txt", sep = "\t", index = False)

    if rebuild_database == False:
        well_data = pd.read_csv(root + "\\" + "All_wells_labels.txt", sep = "\t")

    if construct_graph == True:

        data_subset = well_data[clustered_cols]
        cmap = plt.cm.get_cmap('gist_ncar')
        clusters = np.unique(well_data["labels"])
        norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))
        fig2 = scatter_matrix(data_subset, c = well_data["labels"], cmap = cmap, norm = norm, figsize=[10, 10], diagonal='hist', hist_kwds = {"edgecolor":'k', "linewidth":0.5}, edgecolor = 'k')#, edgewidth = 0.05)
        #fig2.set_size_inches(18.5, 10.5)
        handles = [plt.plot([],[],color=cmap(i/(len(clusters)-1)), ls="", marker=".", markersize=np.sqrt(10))[0] for i in np.arange(len(clusters))]
        plt.legend(handles, clusters, loc=(1.02, 0), title = "Classes", fontsize = 'small', ncol = 2 )
        plt.savefig(root + "\\" + "Scatter_Matrix_all_FRM_Clusters", dpi = 400)
        plt.show()

    return well_data, str(root + "\\" + "All_wells_labels.txt")

build_modelling_database(rebuild_database, root, wells, hc_scenarios, phase, common_cols, depth_col, clustered_cols, vary_class_w_sw)