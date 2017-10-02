#   Copyright (c) 2017, Joanna L. Wallis
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   1. Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
#   2. Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
#   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#   OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
#   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
#   OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#   ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import pandas as pd
import lasio
import matplotlib.pyplot as plt
from sys import exit

show_charts = True
save_charts = True
save_txt = True

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\INI'
wells = ["21_24-1", "21_24-4", "21_24-5", "21_24-6", "21_24-7", "21_25-8", "21_25-9","21_25-10"]
null = -999.25

md_min = 5000
md_max = 12000

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


df = pd.DataFrame()
means = pd.DataFrame()

filepaths = []
for well in wells:

    path = root + "\\" + well + ".las"
    filepaths.append(path)
    data, units_dict = data_load(path)

    data.dropna(subset = ["Vp"], inplace=True)
    data.dropna(subset=["Vs"], inplace=True)
    data.dropna(subset=["RhoB"], inplace=True)
    #print (data)
    data["PhiE"] = data["PhiE"].where(data["PhiE"].notnull(), 0)
    data["PhiT"] = data["PhiT"].where(data["PhiT"].notnull(), 0)
    data["Vsh"] = data["Vsh"].where(data["Vsh"].notnull(), 1)
    data["Sw"] = data["Sw"].where(data["Sw"].notnull(), 1)

    data_clay = data.loc[(data["Vsh"] > 0.95) & (data["Md"] < md_max) & (data["Md"] > md_min), :].copy()

    data_clay["Mu"] = np.power(data_clay["Vs"], 2) * data_clay["RhoB"] * 1000
    data_clay["K"] = np.power(data_clay["Vp"], 2) * data_clay["RhoB"] * 1000 - data_clay["Mu"] * 4/3

    data_clay["Mu"] = data_clay["Mu"] * pow(10, -9)
    data_clay["K"] = data_clay["K"] * pow(10, -9)

    well_means = {}
    well_means["Well"] = well
    well_means["Vp mean"] = data_clay["Vp"].mean(axis = 0)
    well_means["Vs mean"] = data_clay["Vs"].mean(axis = 0)
    well_means["RhoB mean"] = data_clay["RhoB"].mean(axis = 0)
    well_means["K mean"] = data_clay["K"].mean(axis = 0)
    well_means["Mu mean"] = data_clay["Mu"].mean(axis = 0)
    well_means["# Samples"] = len(data_clay["Md"])

    data_clay["Well"] = well
    df = pd.concat([df, data_clay])
    means = means.append(well_means, ignore_index = True)

all_means = {}
all_means["Well"] = "all wells"
all_means["Vp mean"] = df["Vp"].mean(axis=0)
all_means["Vs mean"] = df["Vs"].mean(axis=0)
all_means["RhoB mean"] = df["RhoB"].mean(axis=0)
all_means["K mean"] = df["K"].mean(axis=0)
all_means["Mu mean"] = df["Mu"].mean(axis=0)
all_means["# Samples"] = len(df["Md"])
means = means.append(all_means, ignore_index = True)

bins1 = np.linspace(df["Vp"].min(axis = 0), df["Vp"].max(axis = 0), 50)
bins2 = np.linspace(df["Vs"].min(axis = 0), df["Vs"].max(axis = 0), 50)
bins3 = np.linspace(df["RhoB"].min(axis = 0), df["RhoB"].max(axis = 0), 50)
bins6 = np.linspace(df["K"].min(axis = 0), df["K"].max(axis = 0), 50)
bins5 = np.linspace(df["Mu"].min(axis = 0), df["Mu"].max(axis = 0), 50)
bins4 = np.linspace(df["Md"].min(axis = 0), df["Md"].max(axis = 0), 50)

transparency = 0.75
edge = 'k'

fig1, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
fig1.set_size_inches(18.6, 12.3)

fig2, ((ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(2, 3)
fig2.set_size_inches(18.6, 12.3)

for well in wells:
    df_plot = df.loc[(df["Well"] == well), :].copy()

    ax1.hist(df_plot["Vp"], bins=bins1, label = well, alpha=transparency, edgecolor=edge)
    ax2.hist(df_plot["Vs"], bins=bins2, label=well, alpha=transparency, edgecolor=edge)
    ax3.hist(df_plot["RhoB"], bins=bins3, label=well, alpha=transparency, edgecolor=edge)
    ax4.hist(df_plot["Md"], bins=bins4, label=well, alpha=transparency, edgecolor=edge)
    ax5.hist(df_plot["K"], bins=bins5, label=well, alpha=transparency, edgecolor=edge)
    ax6.hist(df_plot["Mu"], bins=bins6, label=well, alpha=transparency, edgecolor=edge)

ax7.hist(df["Vp"], bins=bins1, label = "all wells", alpha=transparency, edgecolor=edge)
ax8.hist(df["Vs"], bins=bins2, label="all wells", alpha=transparency, edgecolor=edge)
ax9.hist(df["RhoB"], bins=bins3, label="all wells", alpha=transparency, edgecolor=edge)
ax10.hist(df["Md"], bins=bins4, label="all wells", alpha=transparency, edgecolor=edge)
ax11.hist(df["K"], bins=bins5, label="all wells", alpha=transparency, edgecolor=edge)
ax12.hist(df["Mu"], bins=bins6, label="all wells", alpha=transparency, edgecolor=edge)

ax7.axvline(x = df["Vp"].mean(axis=0), c = "black", label = "Mean Vp = %.2f" % df["Vp"].mean(axis=0))
ax8.axvline(x = df["Vs"].mean(axis=0), c = "black", label = "Mean Vs = %.2f" % df["Vs"].mean(axis=0))
ax9.axvline(x = df["RhoB"].mean(axis=0), c = "black", label = "Mean RhoB = %.2f" % df["RhoB"].mean(axis=0))
ax10.axvline(x = df["Md"].mean(axis=0), c = "black", label = "Mean Md = %.2f" % df["Md"].mean(axis=0))
ax11.axvline(x = df["K"].mean(axis=0), c = "black", label = "Mean K = %.2f" % df["K"].mean(axis=0))
ax12.axvline(x = df["Mu"].mean(axis=0), c = "black", label = "Mean Mu = %.2f" % df["Mu"].mean(axis=0))

for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12):
    ax.legend(loc='upper right', shadow=True, fontsize = 'medium')
    ax.set_ylabel("freq")
    ax.grid()

ax1.set_xlabel("Vp (m/s)")
ax2.set_xlabel("Vs (m/s)")
ax3.set_xlabel("RhoB (g/cc)")
ax4.set_xlabel("Md (ft)")
ax5.set_xlabel("K (GPa)")
ax6.set_xlabel("Mu (GPa)")
ax7.set_xlabel("Vp (m/s)")
ax8.set_xlabel("Vs (m/s)")
ax9.set_xlabel("RhoB (g/cc)")
ax10.set_xlabel("Md (ft)")
ax11.set_xlabel("K (GPa)")
ax12.set_xlabel("Mu (GPa)")

#ax1.hist(df["Vp"], bins=bins1, label=df["Well"], alpha=transparency, edgecolor=edge)
#ax2.hist(df["Vs"], bins=bins2, label=df["Well"], alpha=transparency, edgecolor=edge)
#ax3.hist(df["RhoB"], bins=bins3, label=df["Well"], alpha=transparency, edgecolor=edge)
#ax4.hist(df["Md"], bins=bins4, label=df["Well"], alpha=transparency, edgecolor=edge)
#ax5.hist(df["K"], bins=bins5, label=df["Well"], alpha=transparency, edgecolor=edge)
#ax6.hist(df["Mu"], bins=bins6, label=df["Well"], alpha=transparency, edgecolor=edge)

if show_charts == True:
    plt.show()

if save_charts == True:
    fig1_name = (root + "\\" + "clay_properties_by_well.png")
    fig1.savefig(fig1_name, dpi = 400, bbox_inches = "tight")

    fig2_name = (root + "\\" + "clay_properties_all_wells.png")
    fig2.savefig(fig1_name, dpi = 400, bbox_inches = "tight")

if save_txt == True:
    means.to_csv(root + "\\" + "mean_clay_parameters.txt", index = False, sep = "\t")
#print (means)









#return vp_mean, vs_mean, rho_mean, k_mean, mu_mean, num_samples