import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import gc
from itertools import chain
import scipy.interpolate
from matplotlib.colors import LinearSegmentedColormap

from mpl_toolkits.axes_grid1 import AxesGrid


plt.rcParams["font.size"] = 10

rebuild_model_database = True

# define parameters

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'

wells = ["WELL"]
scenarios = ["100WTR","05OIL", "70OIL", "95OIL"]
database = "All_wells_labels.txt"
depth_log = "DEPTH"
vp_log = "Vp"
vs_log = "Vs"
rhob_log = "RhoB"
vsh_log = "Vsh"
sw_log = "Sw"
phie_log = "PhiE"

model_factor = 5 #no samples per class * no class = no models
num_models = 2000
min_class_size = 10
#rules
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

colour_layer = "Layer2"


def sampling(database, num_models, min_class_size): #model_factor
    output_samples = pd.DataFrame()

    #phases = np.unique(database["phase"])
    database["labels"].astype(int, inplace = True)

    class_counts_filt = database.groupby("labels").size().to_frame(name="counts")
    summary_filt = database.groupby("labels").agg(
        {vp_log: np.mean, vs_log: np.mean, rhob_log: np.mean, phie_log: np.mean, vsh_log: np.mean, sw_log: np.mean,
         "Phase": stats.mode}).join(class_counts_filt)

    filt_clusters = np.unique(database["labels"]).astype(int)
    num_filt_clusters = len(filt_clusters)


    # filter for minimum class size
    for x in filt_clusters:
        if summary_filt.loc[x,"counts"] < min_class_size:
            database = database.where(database["labels"] != x).dropna()
            summary_filt.drop(x, inplace = True)
            #filt_clusters = np.delete(filt_clusters, x)

    summary_filt["samp_per_class"] = round((summary_filt["counts"] / sum(summary_filt["counts"]) * num_models), 0)
    filt_clusters = np.unique(database["labels"]).astype(int)

    for x in filt_clusters:
        num_samples = summary_filt["samp_per_class"].loc[x].astype(int)
        class_data = database.where(database["labels"] == x).dropna()
        samp = class_data.sample(n = num_samples, replace = True)

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

def build_models(rebuild_model_database, root, database, depth_log, vp_log, vs_log, rhob_log, vsh_log, sw_log, phie_log, num_models, min_class_size, max_vsh2, min_vsh2, max_sw2, min_sw2, max_phie2, min_phie2, max_vsh1, min_vsh1, max_sw1, min_sw1, max_phie1, min_phie1):

    well_data = pd.read_csv(root + "\\" + database, sep="\t").dropna(subset = [vp_log,vs_log, rhob_log, phie_log, vsh_log, sw_log, "Phase"], how = "any", axis = 0)
    well_data = pd.DataFrame(well_data)
    classes_list = (np.unique(well_data["labels"])).astype(int)
    #print (well_data.head())


    class_counts = well_data.groupby("labels").size().to_frame(name="counts")
    summary = well_data.groupby("labels").agg(
        {vp_log: np.mean, vs_log: np.mean, rhob_log: np.mean, phie_log: np.mean, vsh_log: np.mean, sw_log: np.mean,
         "Phase": stats.mode}).join(class_counts)

    phase = (list(chain.from_iterable(list(map(lambda x: x[0].tolist(), summary["Phase"])))))
    summary["Phase"] = phase
    print("Summary of Property Means by Cluster Label:")
    print (summary)
    summary.to_csv(path = root + "\\" + "Summary_of_Properties_by_Class.txt", sep = "\t", index = False)



    if rebuild_model_database == True:

        reservoir_database = well_data.loc[(well_data[vsh_log] < max_vsh2) & (well_data[vsh_log] >= min_vsh2) & (well_data[sw_log] <= max_sw2) & (well_data[sw_log] >= min_sw2) & (well_data[phie_log] <= max_phie2)& (well_data[phie_log] > min_phie2)].copy().dropna(how = "any", axis = 0)
        print("Reservoir Database Entries:")
        print (len(reservoir_database[depth_log]))
        reservoir_counts = reservoir_database.groupby("labels").size().to_frame(name="Layer 2 Counts")
        #print(reservoir_counts)
        #summary_reservoir = reservoir_database.groupby("labels")
        overburden_database = well_data.loc[(well_data[vsh_log] <= max_vsh1) & (well_data[vsh_log] > min_vsh1) & (well_data[sw_log] <= max_sw1) & (well_data[sw_log] >= min_sw1) & (well_data[phie_log] <= max_phie1)& (well_data[phie_log] > min_phie1)].copy().dropna(how = "any", axis = 0)
        print("Overburden Database Entries:")
        print(len(overburden_database[depth_log]))
        overburden_counts = overburden_database.groupby("labels").size().to_frame(name="Layer 1 Counts")

        split_counts = reservoir_counts.join(overburden_counts)
        print (split_counts)

        reservoir_sample = sampling(reservoir_database, num_models, min_class_size).rename(columns = lambda x: x + "_Layer2")
        overburden_sample = sampling(overburden_database, num_models, min_class_size).rename(columns = lambda x: x + "_Layer1")


        if reservoir_sample[str(depth_log) + "_Layer2"].count() > overburden_sample[str(depth_log) + "_Layer1"].count():
            reservoir_sample = reservoir_sample.sample(axis = 0, weights = None, n = overburden_sample[str(depth_log) + "_Layer1"].count(), replace = False).dropna(how = "any", axis = 0)
        elif overburden_sample[str(depth_log) + "_Layer1"].count() > reservoir_sample[str(depth_log) + "_Layer2"].count():
            overburden_sample = overburden_sample.sample(axis=0, weights=None, n=reservoir_sample[str(depth_log) + "_Layer2"].count(), replace=False).dropna(how = "any", axis = 0)
        else:
            pass

        models = overburden_sample.join(reservoir_sample).dropna(how = "any", axis = 0)

        r0, g, f = shuey_three(models[str(vp_log) + "_Layer1"], models[str(vp_log) + "_Layer2"],models[str(vs_log) + "_Layer1"],models[str(vs_log) + "_Layer2"],models[str(rhob_log)+ "_Layer1"],models[str(rhob_log)+ "_Layer2"])

        models["Shuey R0"] = r0
        models["Shuey G"] = g
        models["Shuey F"] = f

        models.to_csv(root + "\\" + "Half_Space_Models.txt", sep="\t", index=False)

    if rebuild_model_database == False:
        models = pd.read_csv(root + "\\" + "Half_Space_Models.txt", sep = "\t")

    print ("models database created and loaded")
    num_models = int(len(models.index))
    print("num_models = %d" % num_models)

    return root + "\\" + "Half_Space_Models.txt", models, classes_list, summary

def plot_models(models, classes_list, colour_layer):

    num_clusters = len(classes_list)
    print ("Class List =" + str(classes_list))
    print ("Number of Classes = %d" % num_clusters)

    fig1 = plt.figure(1, (4., 4.))
    ax = AxesGrid(fig1, 111,
                   nrows_ncols = (2, 2),
                   axes_pad = (1.7, 0.5),
                   cbar_mode="None",
                  label_mode='all') #share_all=False,


    fig1.set_size_inches(16, 9)
    cmap = plt.cm.get_cmap('hot_r')
    norm = plt.Normalize(vmin=0, vmax=1)
    area = np.pi * 2 ** 2

    map1 = ax[0].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models[str(vsh_log) + "_" + colour_layer], cmap=cmap, norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)

    norm = plt.Normalize(vmin=0, vmax=0.3)
    map2 = ax[1].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models[str(phie_log) + "_" + colour_layer], cmap='viridis', norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)
    #print (models.head())
    """
    fluids_cm = []
    if np.any(models["Phase_Layer2"].isin(["gas"])) == True:
        red = [(1, 0, 0, 1), (1, 0, 0, 1)]
        fluids_cm = fluids_cm + red
        print (fluids_cm)
        print ("TRUE!")
    if np.any(models["Phase_Layer2"].isin(["oil"])) == True:
        green = [(0, 1, 0, 1), (0, 1, 0, 1)]
        fluids_cm = fluids_cm + green
    if np.any(models["Phase_Layer2"].isin(["water"])) == True:
        blue =

    scenarios = ["100WTR", "05OIL", "70OIL", "95OIL"]
    gas = (0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1)
    fluids_cm = (0, 0, 1, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 0, 0, 1), (1, 0, 0, 1)

    cm_dict = {}
    for x, y in zip(scenarios, fluids_cm):
        cm_dict[x] = y

    fluids_cmap = colors.ListedColormap(fluids_cm)
    """
    red = (1, 0, 0, 1)
    green = (0, 1, 0, 1)
    blue = (0, 0, 1, 1)

    for x, colour in zip(["gas","oil","water"], [red, green, blue]):

        plot_data = models.where(models["Phase_" + colour_layer] == x).dropna(how = "any")
        ax[2].scatter(plot_data["Shuey R0"], plot_data["Shuey G"],s=area, c= colour,norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)

    #map3 = ax[2].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=[cm_dict[i] for i in models["Scenario_Layer2"]],norm=norm, alpha=1, edgecolors='k', linewidths = 0.15)

    cmap = plt.cm.get_cmap('gist_ncar')
    norm = plt.Normalize(vmin=min(classes_list), vmax=max(classes_list))
    ax[3].scatter(models["Shuey R0"], models["Shuey G"],s=area, c=models["labels_" + colour_layer], cmap=cmap, norm=norm, alpha = 1, edgecolors='k', linewidths = 0.15)

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


    cax1 = fig1.add_axes([0.46, 0.55, 0.01, 0.33])
    cax2 = fig1.add_axes([0.79, 0.55, 0.01, 0.33])

    cbar1 = fig1.colorbar(map1, cax= cax1).set_label(label = colour_layer + "Vsh(frac)", size = "x-small")
    cbar2 = fig1.colorbar(map2, cax= cax2).set_label(label = colour_layer + "PhiE(frac)", size = "x-small")



    handles_x = [plt.plot([], [], color = i, ls = "", marker = ".", markersize=area, markeredgewidth = 0.1, markeredgecolor = 'k')[0] for i in [red, green, blue]]

    ax[2].legend(handles_x, ["gas", "oil", "water"], loc = (1.02, 0), fontsize = 'x-small', title = colour_layer + " Fluid")

    cmap = plt.cm.get_cmap('gist_ncar')
    handles = [plt.plot([], [], color=cmap(i / (num_clusters- 1)), ls="", marker=".", markersize=area,  markeredgewidth = 0.1, markeredgecolor = 'k')[0] for i in
               np.arange(num_clusters)]

    ncol = int(np.ceil(num_clusters/16))

    ax[3].legend(handles, classes_list, loc=(1.02, 0.0), title="Classes" + colour_layer, fontsize='x-small', ncol=ncol)



    #print("saving fig...")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Intercept-Gradient for Half Space Models, coloured by %s properties" % colour_layer)

    fig_name = "Intercept_Gradient_all_models_" + colour_layer + ".png"
    fig1.savefig(root + "\\" + fig_name, dpi = 400, bbox = "tight", pad_inches = 1)


    plt.show()


    return (root + "\\" + fig_name)

# add back in to run from script

models_database_file, models_database, classes_list, summary = build_models(rebuild_model_database, root, database, depth_log, vp_log, vs_log, rhob_log, vsh_log, sw_log, phie_log, num_models, min_class_size, max_vsh2, min_vsh2, max_sw2, min_sw2, max_phie2, min_phie2, max_vsh1, min_vsh1, max_sw1, min_sw1, max_phie1, min_phie1)
plot_filename = plot_models(models_database, classes_list, "Layer1")
plot_filename = plot_models(models_database, classes_list,  "Layer2")

def contour_plot(models, classes_list, colour_layer):

    fig2 = plt.figure(1, (4., 4.))
    ax = AxesGrid(fig2, 111,
                  nrows_ncols=(2, 2),
                  axes_pad=(1.7, 0.5),
                  cbar_mode="None",
                  label_mode='all')  # share_all=False,

    for x in ax:
        x.set_xlabel("Intercept")
        x.set_ylabel("Gradient")
        x.set_xlim(-0.5, 0.5)
        x.set_ylim(-0.5, 0.5)
        x.grid()
        x.set_axisbelow(True)
        x.axhline(y = 0, color = 'k')
        x.axvline(x = 0, color = 'k')


    fig2.set_size_inches(16, 9)
    cmap = plt.cm.get_cmap('hot_r')
    norm = plt.Normalize(vmin=0, vmax=1)
    area = np.pi * 2 ** 2

    #ax[0].hexbin()

    #map1 = ax[0].scatter(models["Shuey R0"], models["Shuey G"], s=area, c=models[str(vsh_log) + "_" + colour_layer],
    #                     cmap=cmap, norm=norm, alpha=1, edgecolors='k', linewidths=0.15)

    cmap = plt.cm.get_cmap('hot_r')
    #X, Y = np.meshgrid(models["Shuey R0"], models["Shuey G"])
    #rbf = scipy.interpolate.Rbf(models["Shuey R0"], models["Shuey G"], models[str(vsh_log) + "_" + colour_layer],  function = "linear")
    #Z = rbf(X, Y)
    xi, yi = np.linspace(-0.5, 0.5, 200), np.linspace(-0.5, 0.5, 200)
    X, Y = np.meshgrid(xi, yi)
    #x = (models[["Shuey R0", "Shuey G"]].as_matrix(columns = None)) #, models[].as_matrix(columns = None))
    x = models["Shuey R0"].tolist()
    y = models["Shuey G"].tolist()
    points = list(zip(x, y))
    vsh = scipy.interpolate.griddata(points, models[str(vsh_log) + "_" + colour_layer] , (X, Y), method = "linear")
    phie = scipy.interpolate.griddata(points, models[str(phie_log) + "_" + colour_layer], (X, Y), method="linear")
    sw = scipy.interpolate.griddata(points, models[str(sw_log) + "_" + colour_layer], (X, Y), method="linear")
    #X, Y, Z = np.mgrid(models["Shuey R0"], models["Shuey G"], models[str(vsh_log) + "_" + colour_layer])
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.get_cmap('hot_r')
    map1 = ax[0].contourf(X, Y, vsh, cmap = cmap, norm = norm)
    cax1 = fig2.add_axes([0.46, 0.55, 0.01, 0.33])
    cbar1 = fig2.colorbar(map1, cax=cax1).set_label(label=colour_layer + "Vsh(frac)", size="x-small")

    norm = plt.Normalize(vmin=0, vmax=0.3)
    area = 2 * np.pi ** 2
    map2 = ax[1].scatter(X, Y, c = phie, cmap='viridis', s = area, marker = "h", norm = norm)
    cax2 = fig2.add_axes([0.79, 0.55, 0.01, 0.33])
    cbar2 = fig2.colorbar(map2, cax= cax2).set_label(label = colour_layer + "PhiE(frac)", size = "x-small")

    norm = plt.Normalize(vmin=0, vmax=1)
    area = 2 * np.pi ** 2
    map2 = ax[2].scatter(X, Y, c = sw, cmap='winter_r', s = area, marker = "h", norm = norm)
    cax3 = fig2.add_axes([0.46, 0.1, 0.01, 0.33])
    cbar3 = fig2.colorbar(map2, cax= cax3).set_label(label = colour_layer + "Sw(frac)", size = "x-small")

    red = (1, 0, 0, 1)
    green = (0, 1, 0, 1)
    blue = (0, 0, 1, 1)
    norm = plt.Normalize(vmin=0, vmax=1)
    #oil_cmap = colors.ListedColormap([green, blue])
    oil_cmap = LinearSegmentedColormap.from_list(
        "oil_cmap", [green, blue], N=100)
    gas_cmap = colors.ListedColormap([red, blue])
    water_cmap = colors.ListedColormap([blue, blue])

    #for fluid, cmap in zip(["gas","oil","water"], [gas_cmap, oil_cmap, water_cmap]):
    for fluid, cmap, grid in zip(["water", "oil", "gas",],[water_cmap, oil_cmap, gas_cmap], [0.78, 0.81, 0.84]):
        plot_data = models.where(models["Phase_" + colour_layer] == fluid).dropna(how="any")
        if (len(plot_data[depth_log +"_" + colour_layer])) != 0:

            norm = plt.Normalize(vmin=0, vmax=1)
            area = 2 * np.pi ** 2
            #ax[2].scatter(plot_data["Shuey R0"], plot_data["Shuey G"],s=area, c= plot_data["Sw" + "_" + colour_layer], norm=norm, cmap = cmap, alpha=1, marker = "h")

            xi, yi = np.linspace(-0.5, 0.5, 200), np.linspace(-0.5, 0.5, 200)
            X, Y = np.meshgrid(xi, yi)
            x = plot_data["Shuey R0"].tolist()
            y = plot_data["Shuey G"].tolist()
            sw_in = plot_data[str(sw_log) + "_" + colour_layer].tolist()
            points = list(zip(x, y))
            sw = scipy.interpolate.griddata(points, sw_in, (X, Y), method="linear")
            map4 = ax[3].scatter(X, Y, c=sw, cmap=cmap, s=area, marker="h", norm=norm, alpha = 0.5)

            cax4 = fig2.add_axes([grid, 0.1, 0.01, 0.33])
            fig2.colorbar(map4, cax=cax4).set_label(label=colour_layer + "Sw(frac)", size="x-small")

    plt.suptitle("Intercept-Gradient for Half Space Models, coloured by %s properties" % colour_layer)

    plt.tight_layout(rect=[0, 0, 1, 0.95])


    fig2_name = "Intercept_Gradient_density_all_models_" + colour_layer + ".png"
    fig2.savefig(root + "\\" + fig2_name, dpi = 400, bbox = "tight", pad_inches = 1)


    #phase =  scipy.interpolate.griddata(points, models["Phase" + "_" + colour_layer], (X, Y), method="nearest")
    #ax[2].scatter(X, Y, phase)

    #ax[3].scatter(X, Y, c = Z, cmap = cmap)
    #cbar2 = fig2.colorbar(mappable, ax[1])


    #



    plt.show()

    return

#contour_plot(models_database, classes_list, colour_layer)
#contour_plot(models_database, classes_list, "Layer1")