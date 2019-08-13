
import lasio
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\FRM'

wells = ["21_24-1", "21_24-4","21_24-5"] #,"21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]
#scenarios = ["05GAS", "70GAS", "95GAS", "100WTR"]
scenarios = ["100WTR"]
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

#print (well_data.head)

well_data["Mu"] = pow(well_data["Vs"], 2) * well_data["RhoB"] * 1000
well_data["K"] = pow(well_data["Vp"], 2) * well_data["RhoB"] * 1000 - 4/3 * well_data["Mu"]
well_data["KM"] =  pow(well_data["Vp"], 2) * well_data["RhoB"] * 1000



data_subset = well_data[["Vp", "RhoB","Vsh", "PhiE"]]
scatter = pd.scatter_matrix(data_subset, c = well_data["Vsh"])
X = StandardScaler().fit_transform(data_subset)

#kmeans = KMeans(n_clusters=2).fit(data_subset)
#db = DBSCAN(eps=0.3, min_samples=100).fit(X)
#db = KMeans().fit(X) #n_clusters=3
#db = KMeans(n_clusters = 4, n_init = 10).fit(X)
db = MeanShift().fit(X) #cluster_all = True
#db = SpectralClustering(affinity = "rbf", gamma = 0.5).fit(X) #,n_clusters = max_facies,
#labels = kmeans.labels_
well_data["labels"] = db.labels_

clusters = np.unique(well_data["labels"])
print (clusters)
no_clusters = len(clusters)

#fig1, (ax1) = plt.subplots(1, 1)
cmap = plt.cm.get_cmap('rainbow')
norm = plt.Normalize(vmin=min(clusters), vmax=max(clusters))
pd.scatter_matrix(data_subset, c = well_data["labels"], cmap = cmap, norm = norm)

handles = [plt.plot([],[],color=cmap(i/(len(clusters)-1)), ls="", marker=".", markersize=np.sqrt(10))[0] for i in np.arange(len(clusters))]

print (np.arange(len(clusters)))

plt.legend(handles, clusters, loc=(1.02, 0), title = "classes")
#scatter.legend()
#gm = GaussianMixture.fit()


plt.show()



