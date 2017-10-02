import lasio
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from pandas.plotting import scatter_matrix
import random
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.size"] = 10

rebuild_model_database = True

# define parameters

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\FRM'

wells = ["21_24-1", "21_24-4","21_24-5","21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]

scenarios = ["100WTR","05OIL", "70OIL", "95OIL", "05GAS", "70GAS", "95GAS"]

well_data = pd.read_csv(root + "\\" "All_wells_labels.txt", sep = "\t")

class_counts = well_data.groupby("labels").size().to_frame(name = "counts")
summary = well_data.groupby("labels").agg({"Vp": np.mean, "Vs": np.mean, "RhoB": np.mean, "PhiE": np.mean, "Vsh": np.mean, "Sw": np.mean, "Phase": stats.mode}).join(class_counts)

no_samples = well_data["Md"].count()
clusters = (np.unique(well_data["labels"])).astype(int)
num_clusters = len(clusters)
model_factor = 20 #no samples per class * no class = no models

#rules?

# reservoir/layer 2
max_vsh2 = 0.8
min_vsh2 = 0

max_sw2 = 1
min_sw2 = 0

max_phie2 = 1
min_phie2 = 0.1

# overburden/layer 1

max_vsh1 = 1
min_vsh1 = 0.8

max_sw1 = 1
min_sw1 = 0.9

max_phie1 = 0.2
min_phie1 = 0



def sampling(database, model_factor):
    output_samples = pd.DataFrame()
    filt_clusters = np.unique(database["labels"]).astype(int)
    database["labels"].astype(int, inplace = True)
    print (len(filt_clusters))


    class_counts_filt = database.groupby("labels").size().to_frame(name="counts")
    summary_filt = database.groupby("labels").agg(
        {"Vp": np.mean, "Vs": np.mean, "RhoB": np.mean, "PhiE": np.mean, "Vsh": np.mean, "Sw": np.mean,
         "Phase": stats.mode}).join(class_counts_filt)

    for x in filt_clusters:
        if summary_filt.loc[x,"counts"] < model_factor:
            database = database.where(database["labels"] != x).dropna()
            summary_filt.drop(x, inplace = True)
            #filt_clusters = np.delete(filt_clusters, x)

    filt_clusters = np.unique(database["labels"]).astype(int)
    print(len(filt_clusters))
    num_filt_clusters = len(filt_clusters)
    num_models = int(num_filt_clusters * model_factor)
    print("num_models = %f" % num_models)
    summary_filt["samp_per_class"] = (num_models / num_filt_clusters)

    for x in filt_clusters:
        num_samples = summary_filt["samp_per_class"].loc[x].astype(int)
        class_data = database.where(database["labels"] == x).dropna()
        samp = class_data.sample(n = num_samples)

        if output_samples.empty == True:
            output_samples = samp
        else:
            output_samples = pd.merge(output_samples, samp, how="outer")

    return output_samples


def shuey_three (vp1, vp2, vs1, vs2, rho1, rho2):
    vp_mean = ((vp2+vp1)/2)
    vs_mean = ((vs2+vs1)/2)
    rho_mean = ((rho2+rho1)/2)
    g = 0.5 * (vp2 - vp1)/vp_mean - 2 * np.power(vs_mean,2)/np.power(vp_mean,2) * ((rho2-rho1)/rho_mean + 2 * (vs2-vs1)/vs_mean)
    r0 = 0.5 * ((vp2-vp1)/vp_mean + (rho2-rho1)/rho_mean)
    f = 0.5 * (vp2 - vp1)/vp_mean

    return r0, g, f

if rebuild_model_database == True:

    reservoir_database = well_data.loc[(well_data["Vsh"] < max_vsh2) & (well_data["Vsh"] >= min_vsh2) & (well_data["Sw"] <= max_sw2) & (well_data["Sw"] >= min_sw2) & (well_data["PhiE"] <= max_phie2)& (well_data["PhiE"] > min_phie2)].copy().dropna(how = "any", axis = 0)
    overburden_database = well_data.loc[(well_data["Vsh"] <= max_vsh1) & (well_data["Vsh"] > min_vsh1) & (well_data["Sw"] <= max_sw1) & (well_data["Sw"] >= min_sw1) & (well_data["PhiE"] <= max_phie1)& (well_data["PhiE"] > min_phie1)].copy().dropna(how = "any", axis = 0)


    reservoir_sample = sampling(reservoir_database, model_factor).rename(columns = lambda x: x + "_Layer2")
    overburden_sample = sampling(overburden_database, model_factor).rename(columns = lambda x: x + "_Layer1")


    if reservoir_sample["Md_Layer2"].count() > overburden_sample["Md_Layer1"].count():
        reservoir_sample = reservoir_sample.sample(axis = 0, weights = None, n = overburden_sample["Md_Layer1"].count(), replace = False).dropna(how = "any", axis = 0)
    elif overburden_sample["Md_Layer1"].count() > reservoir_sample["Md_Layer2"].count():
        overburden_sample = overburden_sample.sample(axis=0, weights=None, n=reservoir_sample["Md_Layer2"].count(), replace=False).dropna(how = "any", axis = 0)
    else:
        pass

    models = overburden_sample.join(reservoir_sample).dropna(how = "any", axis = 0)

    r0, g, f = shuey_three(models["Vp_Layer1"], models["Vp_Layer2"],models["Vs_Layer1"],models["Vs_Layer2"],models["RhoB_Layer1"],models["RhoB_Layer2"])

    models["Shuey R0"] = r0
    models["Shuey G"] = g
    models["Shuey F"] = f

    models.to_csv(root + "\\" + "Half_Space_Models.txt", sep="\t", index=False)

if rebuild_model_database == False:
    models = pd.read_csv(root + "\\" + "Half_Space_Models.txt", sep = "\t")

fig1 = plt.figure(1, (4., 4.))
ax = AxesGrid(fig1, 111,
               nrows_ncols = (2, 2),
               axes_pad = (1.5, 0.5),
               cbar_mode="None",
              label_mode='all') #share_all=False,


fig1.set_size_inches(14.3, 14.3)
cmap = plt.cm.get_cmap('hot_r')
norm = plt.Normalize(vmin=0, vmax=1)
area = np.pi * 2 ** 2

map1 = ax[0].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models["Vsh_Layer2"], cmap=cmap, norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)

norm = plt.Normalize(vmin=0, vmax=0.3)
map2 = ax[1].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models["PhiE_Layer2"], cmap='viridis', norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)


gas_cm = (1, 0, 0, 1), (0.8, 0, 0, 1), (0.6, 0, 0, 1)
oil_cm = (0, 1, 0, 1), (0, 0.6 ,0, 1), (0, 0.4, 0, 1)
water_cm = (0, 0, 1, 1)



fluids_cm = (0, 0, 1, 1), (0, 1, 0, 1), (0, 0.6 ,0, 1), (0, 0.4, 0, 1), (1, 0, 0, 1), (0.8, 0, 0, 1), (0.6, 0, 0, 1)
fluids_cm = (0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 0, 0, 1), (1, 0, 0, 1)

cm_dict = {}
for x, y in zip(scenarios, fluids_cm):
    cm_dict[x] = y

fluids_cmap = colors.ListedColormap(fluids_cm)

map3 = ax[2].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=[cm_dict[i] for i in models["Scenario_Layer2"]],norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)

cmap = plt.cm.get_cmap('gist_ncar')
norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))
ax[3].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models["labels_Layer2"], cmap=cmap, norm=norm, alpha = 1, edgecolors='k', linewidths = 0.15)

# colour by phase...


for x in ax:
    x.set_xlabel("Intercept")
    x.set_ylabel("Gradient")
    x.set_xlim(-0.5, 0.5)
    x.set_ylim(-0.5, 0.5)
    x.grid()
    x.set_axisbelow(True)
    x.axhline(y = 0, color = 'k')
    x.axvline(x = 0, color = 'k')


cax1 = fig1.add_axes([0.47, 0.53, 0.01, 0.33])
cax2 = fig1.add_axes([0.82, 0.53, 0.01, 0.33])

cbar1 = fig1.colorbar(map1, cax= cax1).set_label(label = "Layer 2 Vsh(frac)", size = "x-small")
cbar2 = fig1.colorbar(map2, cax= cax2).set_label(label = "Layer 2 PhiE(frac)", size = "x-small")



handles_x = [plt.plot([], [], color = fluids_cmap(i / len(scenarios)), ls = "", marker = ".", markersize=np.sqrt(10))[0] for i in
           np.arange(len(scenarios))]

ax[2].legend(handles_x, scenarios, loc = (1.02, 0), fontsize = 'x-small')
cmap = plt.cm.get_cmap('gist_ncar')
handles = [plt.plot([], [], color=cmap(i / (num_clusters- 1)), ls="", marker=".", markersize=np.sqrt(10))[0] for i in
           np.arange(num_clusters)]
ax[3].legend(handles, clusters, loc=(1.02, 0.0), title="Classes", fontsize='x-small', ncol=3)
"""

plt.show()
