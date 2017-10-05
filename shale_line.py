import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from markers_dict import markers
import sys

# Las file location. This las file should include the TVDml - otherwise run "depth logs" and "Merge las files" first.
root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\example_well_log'
wells = ["WELL"]
# recommended to run this on 100% water only
scenarios = ["100WTR"] #,"05OIL", "70OIL", "95OIL", "05GAS", "70GAS", "95GAS"]
null = -999.25

# assumes that files are named root\[well]_[scenario]
suffixes = list(map(lambda x: "_" + x, scenarios))

md_log = "Md"
tvd_log = "TVDML ft"

sample_interval = 0.5

rebuild_database = False
plot_markers = False



upper_bound = None #[7200]
lower_bound = None

null = -999.25

# matrix parameters
# matrix velocity/density
vp_ma = 6000
vs_ma = 4500
rhob_ma = 2.8 # mean density of illite = 2.75g/cc

#mudline velocity/density
vp_ml = 1500 # ~P-velocity of water
vs_ml = 800 # velocity of shear waves in water
rhob_ml = 1 #i.e. density of water

plot_other_trends = True
#b for other_trend
dt_b_extra = [0.0001658]
dts_b_extra = [0.00012681]
rho_b_extra = [0.00021094]

# ma for other trend
vp_ma_extra = [6000]
vs_ma_extra = [4500]
rhob_ma_extra = [2.8] # mean density of illite = 2.75g/cc

# ml for other trend
vp_ml_extra = [1500] # ~P-velocity of water
vs_ml_extra = [800] # velocity of shear waves in water
rhob_ml_extra = [1] #i.e. density of water

markers = pd.DataFrame.from_dict(markers).transpose()
reservoir_marker = "320_Fulmar"
td_marker = "999_TD"
markers.index.name = "Well_Name"



#exclude zones





# load logs
if rebuild_database == True:
    filepaths = []
    logs_df = pd.DataFrame()

    for well in wells:
        for scenario, suffix in zip(scenarios, suffixes):
            path = root + "\\" + well + suffix + ".las"
            logs = []
            units = []

            filepaths.append(path)

            data = lasio.read(path)
            for curve in data.curves:
                logs.append(curve.mnemonic)
                units.append(curve.unit)

            well_data = pd.DataFrame()

            for log in logs:
                well_data[str(log)] = data[str(log)]
                well_data[str(log)] = np.where(well_data[str(log)] == null, np.nan,  well_data[str(log)])
            well_data.dropna(how = "all", axis = 0, inplace = True)

            well_data["Well_Name"] = well
            well_data["Scenario"] = scenario

            if logs_df.empty == True:
                logs_df = well_data
            else:
                logs_df = pd.merge(logs_df, well_data, how = "outer")

    # separate out shales

    shale_df = logs_df.dropna(subset = ["Vsh"], inplace = False)
    shale_df = shale_df.where(logs_df["Vsh"] > 0.95).dropna(how = "all", axis = 0)

    shale_df["dt"] = (1 / (shale_df["Vp"] * 3.281)) * pow(10, 6)
    shale_df["dts"] = (1 / (shale_df["Vs"] * 3.281)) * pow(10, 6)
    shale_df.replace([np.inf, -np.inf], np.nan).dropna(subset = ["Vp", "Vs", "RhoB", "dt", "dts", tvd_log], how = "any")
    logs_df.to_csv(root + "\\" + "100WTR_Database.txt", sep = "\t", index = False)
    shale_df.to_csv(root + "\\" + "Shale_Line_Database.txt", sep = "\t", index = False)

# load database from tabbed csv
if rebuild_database == False:

    shale_df = pd.read_csv(root + "\\" + "Shale_Line_Database.txt", sep = "\t")
    logs_df = pd.read_csv(root + "\\" + "100WTR_Database.txt", sep = "\t")



shale_df.sort_values(by = tvd_log, inplace = True, ascending = True)

# find tvdml for markers
marker_names = markers.columns
for name in marker_names:
    # round md depths to sample interval of las files
    markers[name] = list(map(lambda x: round(x * (1/sample_interval)) / (1/sample_interval), markers[name]))

# create dataframe for TVDML of markers
marker_tvdml = pd.DataFrame(index = wells, columns=marker_names)
marker_tvdml.index.name = "Well_Name"
#print (marker_tvdml)

# converting markers from md to tvd
for well in wells:
    max_md = logs_df[md_log].loc[logs_df["Well_Name"] == well].max()
    min_md = logs_df[md_log].loc[logs_df["Well_Name"] == well].min()

    max_tvd = logs_df[tvd_log].max()

    for name in marker_names:
        md = markers[name].loc[well]
        if (md <= max_md and md >= min_md):
            tvdml = logs_df[tvd_log].loc[(logs_df["Well_Name"] == well) & (logs_df[md_log] == md)].tolist()
            marker_tvdml.ix[well, name] = tvdml[0]


dt_ma = (1/(vp_ma *  3.281)) * pow(10, 6)
dt_ml = (1/(vp_ml *  3.281)) * pow(10, 6)

dts_ma = (1/(vs_ma *  3.281)) * pow(10, 6)

if vs_ml != 0:
    dts_ml = (1/(vs_ml *  3.281)) * pow(10, 6)
else:
    dts_ml = 0

def dt_func_matrix (x,  b):

    return dt_ma + (dt_ml - dt_ma) * np.exp(-b * x)

def dts_func_matrix (x,  b):

    return dt_ma + (dts_ml - dts_ma) * np.exp(-b * x)

def rhob_func_matrix (x,  b):

    return rhob_ma + (rhob_ml - rhob_ma) * np.exp(-b * x)

def func (x, a, b, c):
    #return a + (b - a) * np.exp(-c * x)
    #return a * np.exp(-b * x) + c
    return a + b * x + c * np.power(x, 2)

def func_exp (x, a, b, c):
    return a * np.exp(-b * x) + c

#z = np.polyfit (x, y, 3)
#f = np.poly1d(z)

#y_poly = f(x)

#print (shale_df[["dt", "Vp"]])#, shale_df["Vp"])
if upper_bound == None:
    upper_bound = shale_df[tvd_log].min()
if lower_bound == None:
    lower_bound = shale_df[tvd_log].max()

bounded_df = shale_df.where((shale_df[tvd_log] < lower_bound) & (shale_df[tvd_log] > upper_bound)).dropna(how = "all", axis = 0)

x = bounded_df[tvd_log].tolist()
z = shale_df[tvd_log].tolist()
dtp = bounded_df["dt"].tolist()
dts = bounded_df["dts"].tolist()
rho = bounded_df["RhoB"].tolist()

x = np.array(x, dtype=float) #transform your data in a numpy array of floats
z = np.array(z, dtype=float)
dtp = np.array(dtp, dtype=float)
dts = np.array(dts, dtype=float)
rho = np.array(rho, dtype=float)

# fit to P-wave sonic
#dt_popt, dt_pcov = curve_fit(func, x, y, p0=[0.5, 0.5, 0.5])
#dt_popt_exp, dt_pcov_exp = curve_fit(func_exp, x, y, p0=[1500, 0.001, 0.001])
dt_popt_matrix, dt_pcov_matrix = curve_fit(dt_func_matrix, x, dtp, p0 = [0.001])

print("P-wave optimisation complete")

# fit to S-wave sonic
#dts_popt, dts_pcov = curve_fit(func, x, dts, p0=[0.5, 0.5, 0.5])
#dts_popt_exp, dts_pcov_exp = curve_fit(func_exp, x, dts, p0=[800, 0.001, 0.001])
dts_popt_matrix, dt_pcov_matrix = curve_fit(dts_func_matrix, x, dts, p0 = [0.0001])
print("S-wave optimisation complete")

# fit to RhoB

# rhob_popt, rhob_pcov = curve_fit(func, x, r, p0=[0.5, 0.5, 0.5])
#rhob_popt_exp, rhob_pcov_exp = curve_fit(func_exp, x, r, p0=[1, 0.0001, 0.0001])
rhob_popt_matrix, dt_pcov_matrix = curve_fit(rhob_func_matrix, x, rho, p0=[0.001])


#dt_y_popt = func(x, *dt_popt)
#dt_y_popt_exp = func_exp(x, *dt_popt_exp)
dt_y_popt_matrix = dt_func_matrix(z, *dt_popt_matrix)

#vp_y_popt = 0.3048 / (dt_y_popt * pow(10, -6))
#vp_y_popt_exp = 0.3048 / (dt_y_popt_exp * pow(10, -6))
vp_y_popt_matrix = 0.3048 / (dt_y_popt_matrix * pow(10, -6))


#dts_y_popt = func(x, *dts_popt)
#dts_y_popt_exp = func_exp(x, *dts_popt_exp)
dts_y_popt_matrix = dts_func_matrix(z, *dts_popt_matrix)

#vs_y_popt = (0.3048) / (dts_y_popt * pow(10, -6))
#vs_y_popt_exp = (0.3048) / (dts_y_popt_exp * pow(10, -6))
vs_y_popt_matrix = (0.3048) / (dts_y_popt_matrix * pow(10, -6))


#rhob_y_popt = func(x, *rhob_popt)
rhob_y_popt_matrix = rhob_func_matrix(z, *rhob_popt_matrix)
print ("RhoB optimisation complete")

print (dt_popt_matrix)
print (dts_popt_matrix)
print (rhob_popt_matrix)


### plot data
fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
fig1.set_size_inches(18.6, 12.3)
cmap = "hot_r"
norm = plt.Normalize(vmin=0.95, vmax=1)
area = np.pi * 1 ** 2
mappable = ax1.scatter(shale_df["dt"], shale_df[tvd_log], c = shale_df["Vsh"], norm = norm, s = area, cmap = cmap)
ax1.plot(dt_y_popt_matrix, z, c = 'blue', label =  "Matrix lobf")
ax2.scatter(shale_df["dts"], shale_df[tvd_log], c = shale_df["Vsh"], norm = norm, s = area, cmap = cmap)
ax2.plot(dts_y_popt_matrix, z, c = 'blue', label =  "Matrix lobf")
ax3.scatter(shale_df["RhoB"], shale_df[tvd_log], c = shale_df["Vsh"], norm = norm, s = area, cmap = cmap)
ax3.plot(rhob_y_popt_matrix, z, c = 'blue', label =  "Matrix lobf")
#temp = (shale_df[tvd_log] * 0.3048) * 0.0144788 + 91.4228
#ax3.plot(temp, shale_df[tvd_log])


#ax2.plot(func_exp(x, *dt_popt_exp), x, c = 'green', label = "Exponential lobf")
#print (func(shale_df["TVDML"], *popt))

#ax2.plot(y_poly, x, c = 'r', label = "Polynomial lobf")
ax1.legend(loc='upper right', shadow=True, fontsize='x-small')
ax2.legend(loc='upper right', shadow=True, fontsize='x-small')
ax3.legend(loc='upper right', shadow=True, fontsize='x-small')
#print (popt)

ax4.plot(vp_y_popt_matrix, z, c = 'green', label = "Exponential lobf")
ax5.plot(vs_y_popt_matrix, z, c = 'green', label = "Exponential lobf")
ax6.plot(rhob_y_popt_matrix, z, c = 'green', label = "Exponential lobf")


def trends(x, ma, ml, b):
    return ma + (ml - ma) * np.exp(-b * x)

def m_s2us_ft(velocity):
    sonic = (1 / (velocity * 3.281)) * pow(10, 6)
    return sonic

def us_ft2m_s(sonic):
    velocity = 0.3048 / (sonic * pow(10, -6))
    return velocity

if plot_other_trends == True:

    for dt_x, dts_x, rho_x, vp_ma_x, vs_ma_x, rho_ma_x, vp_ml_x, vs_ml_x, rho_ml_x in zip(dt_b_extra, dts_b_extra, rho_b_extra, vp_ma_extra, vs_ma_extra, rhob_ma_extra, vp_ml_extra, vs_ml_extra, rhob_ml_extra):
        #dt_ma_x = m_s2us_ft(vp_ma_x)
        #dts_ma_x = m_s2us_ft(vp_ms_x)
        dt_ma_x, dts_ma_x, dt_ml_x, dts_ml_x = list(map(lambda x: m_s2us_ft(x), [vp_ma_x, vs_ma_x, vp_ml_x, vs_ml_x]))
        dt_trend = trends(z, dt_ma_x, dt_ml_x, dt_x)
        dts_trend = trends(z, dts_ma_x, dts_ml_x, dts_x)
        rho_trend = trends(z, rho_ma_x, rho_ml_x, rho_x)
        vp_trend = us_ft2m_s(dt_trend)
        vs_trend = us_ft2m_s(dts_trend)

        ax1.plot(dt_trend, z, label = "b = %f" % dt_x)
        ax2.plot(dts_trend, z, label = "b = %f" % dts_x)
        ax3.plot(rho_trend, z, label="b = %f" % rho_x)
        ax4.plot(vp_trend, z, label = "b = %f" % dt_x)
        ax5.plot(vs_trend, z, label="b = %f" % dts_x)
        ax6.plot(rho_trend, z, label="b = %f" % rho_x)




colormap = plt.cm.nipy_spectral
colorst = [colormap(i) for i in np.linspace(0, 0.9, len(wells))]
color_dict = {}
for well, col in zip(wells, colorst):
    color_dict[well] = col
#print (colorst)

for well in wells:
    colour = color_dict[well]
    n = ax4.scatter(shale_df["Vp"].where(shale_df["Well_Name"] == well), shale_df[tvd_log].where(shale_df["Well_Name"] == well), s = area, label = well, c = colour)
    ax5.scatter(shale_df["Vs"].where(shale_df["Well_Name"] == well), shale_df[tvd_log].where(shale_df["Well_Name"] == well), s = area, label = well, c = colour)
    ax6.scatter(shale_df["RhoB"].where(shale_df["Well_Name"] == well),shale_df[tvd_log].where(shale_df["Well_Name"] == well), s=area, label=well, c=colour)
    #line_col = n.get_color()
    if plot_markers == True:
        res_depth = marker_tvdml[reservoir_marker].loc[well]
        if res_depth != np.nan:
            ax4.axhline(y = res_depth, label = well, c = colour, linestyle = "dashed")
        well_depth = marker_tvdml[td_marker].loc[well]
        #print (well_depth)
        if well_depth != np.nan:
            ax4.axhline(y = well_depth, label = well, c = colour, linestyle = "solid")


ax4.legend(loc='upper right', shadow=True, fontsize='x-small')
ax5.legend(loc='upper right', shadow=True, fontsize='x-small')
ax6.legend(loc='upper right', shadow=True, fontsize='x-small')

ax1.set_xlabel = "dt, us/ft"
ax2.set_xlabel = "dts, us/ft"
ax3.set_xlabel = "RhoB, g/cc"
ax4.set_xlabel = "Vp, m/s"
ax5.set_xlabel = "Vs, m/s"
ax6.set_xlabel = "RhoB, g/cc"


cax = fig1.add_axes([0.05, 0.1, 0.02, 0.8])
cbar1 = fig1.colorbar(mappable, cax = cax, label = "Vsh, frac")



for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_ylabel = str(tvd_log)
    ax.set_axisbelow(True)
    ax.grid()
    ax.invert_yaxis()
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.axhline(y = upper_bound, c = "grey")
    ax.axhline(y = lower_bound,  c = "grey")


#print (shale_df)

plt.show()

