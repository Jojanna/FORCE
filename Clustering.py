
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
plt.rcParams["figure.figsize"] = (16., 9.)
plt.rcParams["lines.markersize"] = 6
plt.rcParams["lines.markeredgewidth"] =0.0


root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'

wells = ["WELL"]
water_scenarios = ["100WTR"]
cols_to_cluster = ["Vp", "RhoB","Vsh", "PhiE"]
vsh_log = "Vsh"

set_num_classes = True
num_classes = 3


null = -999.25
#max_facies = 3

"""
# import list of las files
# create dataframe
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

def create_clustering_database(root, wells, water_scenarios, cols_to_cluster):

    filepaths = []

    well_data = pd.DataFrame()

    for well in wells:
        for scenario in water_scenarios:
            path = root + "\\" + well + "_" + scenario + ".las"
            filepaths.append(path)
            data, units_dict = data_load(path)
            data["Well_Name"] = well
            data["Scenario"] = scenario
            data.dropna(how = "any", subset = cols_to_cluster, axis = 0, inplace = True)
            if well_data.empty == True:
                well_data = data
            else:
                well_data = pd.merge(well_data, data, how = "outer")

    return well_data


def facies_clustering(well_data, cols_to_cluster, vsh_log, set_num_classes, num_classes):
    cmap = "hot_r"
    norm = plt.Normalize(vmin=0, vmax=1)

    data_subset = well_data[cols_to_cluster]#, "Md"]]
    scatter_matrix(data_subset, c = well_data[vsh_log], cmap = cmap, norm = norm, hist_kwds = {"edgecolor":'k', "linewidth":0.5}, edgecolor = 'k')
    X = StandardScaler().fit_transform(data_subset)

    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    a = np.array([[0, 1]])
    cbar = plt.scatter(a, a, c = a, cmap = "hot_r", norm = norm)
    #plt.gca().set_visible(False)
    plt.colorbar(cbar, cax = cax, label = "Vsh, frac")


    plt.savefig(root + "\\" + "Scatter_Matrix_100WTR_Vsh", dpi = 400)
    #db = KMeans(n_clusters=6).fit(data_subset)
    #db = DBSCAN(eps=0.5, min_samples=100, p = 2).fit(X)



    if set_num_classes == False:
        db = KMeans(algorithm = "elkan").fit(X) #n_clusters=3
    elif set_num_classes == True:
        db = KMeans(algorithm="elkan", n_clusters = num_classes).fit(X)
    #db = KMeans(n_clusters = 4, n_init = 10).fit(X)
    #db = MeanShift(cluster_all = True).fit(X) #cluster_all = True
    #db = SpectralClustering(affinity = "rbf", gamma = 0.5, n_clusters = 6).fit(X) #, = max_facies,
    #labels = kmeans.labels_
    well_data["labels"] = db.labels_
    well_data.reset_index()



    clusters = np.unique(well_data["labels"])
    no_clusters = len(clusters)

    cmap = plt.cm.get_cmap('rainbow')
    norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))
    fig2 = scatter_matrix(data_subset, c = well_data["labels"], cmap = cmap, norm = norm, hist_kwds = {"edgecolor":'k', "linewidth":0.5}, edgecolor = 'k')
    # create legend for fig 2 - colouring facies
    handles = [plt.plot([],[],color=cmap(i/(len(clusters)-1)), ls="", marker=".", markersize=np.sqrt(10))[0] for i in np.arange(len(clusters))]
    plt.legend(handles, clusters, loc=(1.02, 0), title = "classes")

    plt.savefig(root + "\\" + "Scatter_Matrix_100WTR_Clusters", dpi = 400)

    well_data.to_csv(root + "\\" + "All_wells_100WTR_labels.txt", sep = "\t", index = False)

    database_filename = str(root + "\\" + "All_wells_100WTR_labels.txt")
    #scatter.legend()
    #gm = GaussianMixture.fit()

    plt.show()

    return well_data, database_filename

well_data = create_clustering_database(root, wells, water_scenarios, cols_to_cluster)
clustered_data, database_filename = facies_clustering(well_data, cols_to_cluster, vsh_log, set_num_classes, num_classes)






