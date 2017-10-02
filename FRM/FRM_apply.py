

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
from fluid import batzle_wang

from FRM_function import  HS_2_phase_L, fluids_calc, reuss_fluids, sat, dry_rock, fluid_sub_k, rho_dry_calc, multiple_FRM, calc_vp_vs

check_bounds_only = False
show_charts = False
save_las = True
save_fig = True


"""
This module 
"""

root = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\INI'
parameters_file = r'C:\Users\joanna.wallis\Documents\FORCE_presentation\FORCE_catcher\FRM_parameters.txt'

wells = ["21_24-1", "21_24-4","21_24-5","21_24-6","21_24-7","21_25-8","21_25-9","21_25-10"]
null = -999.25

# Clay parameters
factor = 1
mu_clay = 4.72 * pow(10, 9) * factor # Shear modulus of clay in Pa
k_clay = 14.66 * pow(10, 9) * factor # Bulk modulus of clay in Pa
rho_clay = 2.48 * pow(10, 3) * factor #Density of clay in kg/m^3

# Quartz parameters
mu_qtz = 45.0 * pow(10, 9)  # Shear modulus of quartz in Pa
k_qtz = 37.0 * pow(10, 9)  # Bulk modulus of quartz in Pa
rho_qtz = 2.65 * pow(10, 3)  #Density of quartz in kg/m^3

output_sw = [0.05, 0.3, 0.95, 1]
output_fluid = 'oil'
scenarios = ["95OIL", "70OIL", "05OIL", "100WTR"]

#fluid sub cutoffs
frm_max_vsh = 1#0.3 #max Vsh to apply FRM to
frm_min_phie = 0 #0.08 # min phie to apply FRM to
frm_max_phie = 1 #max phie to apply FRM to
frm_min_pr = 0.1

#dry rock model bounds
k_phi_const_max = 0.25 #Upper bound on the dry rock model (picked from graph)
k_phi_const_min = 0.05 #Lower bound on the dry rock model (picked from graph)

# read las file
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

# load frm parameters from tab delimited txt file
def parameters_load(parameters_file, well):
    parameters_all = pd.read_table(parameters_file, sep = "\t", skip_blank_lines = True)
    parameters_all.set_index(["Well"], inplace = True)
    parameters = parameters_all.loc[well].copy()

    return parameters

filepaths = []

for well in wells:
    plt.close()

    path = root + "\\" + well + ".las"
    fig1_name = (root +"\\" + well + "_dry_rock_bounds.png")
    filepaths.append(path)
    data, units_dict = data_load(path)


    data.dropna(subset = ["Vp"], inplace=True)#, how = 'all', axis = 0)
    data.dropna(subset=["Vs"], inplace=True)
    data.dropna(subset=["RhoB"], inplace=True)
    #print (data)
    data["PhiE"] = data["PhiE"].where(data["PhiE"].notnull(), 0)
    data["PhiT"] = data["PhiT"].where(data["PhiT"].notnull(), 0)
    data["Vsh"] = data["Vsh"].where(data["Vsh"].notnull(), 1)
    data["Sw"] = data["Sw"].where(data["Sw"].notnull(), 1)

    data["Vp/Vs"] = data["Vp"]/data["Vs"]
    data["AI"] = data["Vp"] * data["RhoB"]
    data["SI"] = data["Vs"] * data["RhoB"]



    #print(data)

    parameters = parameters_load(parameters_file, well)
    if parameters["In Situ Fluid"] == "Oil":
        parameters["In Situ Fluid"] = "oil"
    elif parameters["In Situ Fluid"] == "Gas":
        parameters["In Situ Fluid"] = "gas"
    #min_md = parameters["Top ZOI (ft MD KB)"]
    #print (min_md)
    #data = data.loc[data["Md ft"] > min_md]

    data["K_Matrix"], data["Mu_Matrix"] =  HS_2_phase_L(k_qtz, k_clay, mu_qtz, mu_clay, data["Vsh"])

    rho_brine, k_brine, rho_h, k_h = fluids_calc(parameters["In Situ Pressure (psi)"]* 0.00689476, parameters["In Situ Temperature C"], parameters["In Situ Fluid"], parameters["Salinity (l/l)"], parameters["Gas Gravity"], parameters["Oil API"], parameters["GOR (scf/bbl)"])
    data["K_Fluid"], data["RhoB_Fluid"] = reuss_fluids(rho_brine, k_brine, rho_h, k_h, data["Sw"])

    data["K_sat"], data["Mu_sat"] = (sat(data["Vp"], data["Vs"], data["RhoB"]))
    data["K_sat GPa"] = data["K_sat"] / np.power(10, 9)
    data["Mu_sat GPa"] = data["Mu_sat"] / np.power(10, 9)

    data["Poisson's Ratio"] = (3 * data["K_sat"] - 2 * data["Mu_sat"]) / (2 * (3 * data["K_sat"] + data["Mu_sat"]))
    data["Lambda"] = (data["K_sat GPa"] - 2/3 * data["Mu_sat GPa"])
    data["Lambda-Rho"] = data["Lambda"] * data["RhoB"]
    data["Mu-Rho"] = data["Mu_sat"]* data["RhoB"]

    units_dict["K"] = "GPa"
    units_dict["Mu"] = "GPa"
    units_dict["Mu_sat"] = "GPa"
    units_dict["AI"] = "m/s*g/cc"
    units_dict["SI"] = "m/s*g/cc"
    units_dict["Lambda"] = "GPa"
    units_dict["Lambda-Rho"] = "GPa*g/cc"
    units_dict["Mu-Rho"] = "GPa*g/cc"

    data["K_pore"], data["K_dry"] = dry_rock(data["PhiE"], data["K_sat"], data["K_Fluid"], data["K_Matrix"])
    data["K_pore_norm"] =  data["K_pore"] / data["K_Matrix"]
    data["K_dry_norm"] = data["K_dry"]/data["K_Matrix"]
    data["K_out"] = fluid_sub_k(data["K_Matrix"], data["PhiE"], data["K_pore"], data["K_Fluid"])
    data["K_out_norm"] = data["K_out"] / data["K_Matrix"]

    porosity = np.arange(0, 1.01, 0.01)
    pore_stiffness_0 = np.arange(0.05, 0.6, 0.05)
    dry_bounds = pd.DataFrame(columns = porosity, index = pore_stiffness_0)
    for phi in porosity:
        dry_bounds[phi] = 1/(1/k_qtz + phi/(k_qtz * dry_bounds.index))/k_qtz
    dry_bounds = dry_bounds.transpose()

    cmap = plt.cm.get_cmap('hot_r')
    norm = plt.Normalize(vmin=0, vmax=1)
    area = np.pi * 2 ** 2

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    fig1.set_size_inches(14.6, 10.3)

    mappable1 = ax1.scatter(data["PhiE"], data["K_dry_norm"], s=area, c=data["Vsh"], cmap=cmap, norm=norm, alpha=0.5,
                            edgecolors='none')
    ax2.scatter(data["PhiE"], data["K_out_norm"], s=area, c=data["Vsh"], cmap=cmap, norm=norm, alpha=0.5, edgecolors='none')

    for ax in (ax1, ax2):
        for phi in list(dry_bounds.columns.values):
            ax.plot(dry_bounds.index.tolist(), dry_bounds[phi].tolist(),label = 'Kphi/K0 = %.2f' % phi)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('phiE')
        ax.set_ylabel('K/K0')
        cbar1 = fig1.colorbar(mappable1, ax=ax)
        cbar1.set_label('Vsh')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
        ax.set_axisbelow(True)
        ax.grid()

    ax1.set_title('dry rock')
    ax2.set_title('wet rock')
    if show_charts == True:
        plt.show()

    if save_fig == True:
        fig1.savefig(fig1_name, dpi=400, bbox_inches="tight")

    if check_bounds_only == True:
        exit("Check Bounds Only = True")

    if save_las == True:

        with open(root + "\\" + str(well) + "_ini_params" + '.las', mode="w") as las1:
            ini = lasio.LASFile()
            ini.depth = ["Md"]
            ini.well["WELL"].value = str(well)
            ini.add_curve("Md", data["Md"], unit=units_dict.get("Md", "ft"))
            #ini.add_curve("DEPTH", md_data, unit="ft")
            ini.add_curve("Bulk Modulus", data["K_sat"], unit="GPa")
            ini.add_curve("Shear Modulus", data["Mu_sat"], unit="GPa")
            ini.add_curve("Vp/Vs", data["Vp/Vs"], unit="ratio")
            ini.add_curve("AI", data["AI"], unit="m/s*g/cc")
            ini.add_curve("SI", data["SI"], unit="m/s*g/cc")
            ini.add_curve("Poisson's Ratio", data["Poisson's Ratio"], unit="ratio")
            ini.add_curve("Lambda-Rho", data["Lambda-Rho"], unit="GPa")
            ini.add_curve("Mu-Rho", data["Mu-Rho"], unit="GPa")
            ini.write(las1, version=2, fmt='%10.5g')

    data["Rhob_dry"] = rho_dry_calc (data["RhoB"], data["RhoB_Fluid"], data["PhiE"])
    #print("kphi/k0 max = %f, kphi/k0 min = %f" % (k_phi_const_max, k_phi_const_min))

    data["K_pore_max"] = k_phi_const_max * data["K_Matrix"]
    data["K_dry_max"] = (1/(1/data["K_Matrix"] + data["PhiE"]/(data["K_pore_max"])))/data["K_Matrix"]
    data["K_dry_max_norm"] = data["K_dry_max"] / data["K_Matrix"]

    data["K_pore_min"] = k_phi_const_min * data["K_Matrix"]
    data["K_dry_min"] = (1/(1/data["K_Matrix"] + data["PhiE"]/(data["K_pore_min"])))/data["K_Matrix"]
    data["K_dry_min_norm"] = data["K_dry_min"] / data["K_Matrix"]


    data["K_pore_norm_bounded"] = np.where(data["K_pore_norm"] > k_phi_const_max, np.array(k_phi_const_max), data["K_pore_norm"])
    data["K_pore_norm_bounded"] = np.where(data["K_pore_norm_bounded"] < k_phi_const_min, np.array(k_phi_const_min), data["K_pore_norm_bounded"])
    data["K_pore_bounded"] = data["K_pore_norm_bounded"] * data["K_Matrix"]

    data["K_dry_bounded"] = k_d_n = 1/(data["PhiE"] /data["K_pore_bounded"] + 1/data["K_Matrix"])
    data["K_dry_bounded_norm"] = data["K_dry_bounded"] / data["K_Matrix"]


#def filter ():
    #if [data["Vsh frac"] > frm_max_vsh, data["Md ft"] < parameters["Top ZOI (ft MD KB)"], data["Md ft"] > parameters["Base ZOI (ft MD KB)"], data["PhiE frac"] > frm_max_phie, data["PhiE frac"] < frm_min_phie] == True

    fig2, (ax1, ax2) = plt.subplots(1, 2)
    fig2.set_size_inches(14.6, 10.3)

    for ax in (ax1, ax2):
        for phi in list(dry_bounds.columns.values):
            ax.plot(dry_bounds.index.tolist(), dry_bounds[phi].tolist(), label='Kphi/K0 = %.2f' % phi)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('phiE')
        ax.set_ylabel('K/K0')
        cbar1 = fig1.colorbar(mappable1, ax=ax)
        cbar1.set_label('Vsh')
        legend = ax.legend(loc='upper right', shadow=True, fontsize='medium')
        ax.set_axisbelow(True)
        ax.grid()
        ax.tick_params(axis='both', which='major', labelsize=6)

    mappable1 = ax1.scatter(data["PhiE"], data["K_dry_norm"], s=area, c=data["Vsh"], cmap=cmap, norm=norm,
                            alpha=0.5,
                            edgecolors='none')
    ax2.scatter(data["PhiE"], data["K_dry_bounded_norm"], s=area, c=data["Vsh"], cmap=cmap, norm=norm,
                alpha=0.5, edgecolors='none')

    plt.tight_layout()
    if show_charts == True:
        plt.show(fig2)

    if save_fig == True:
        fig2_name = (root +"\\" + well + "fig2_applying_rock_model.png")
        fig2.savefig(fig2_name, dpi=400)

    fig3, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6)
    fig3.set_size_inches(18.6, 12.3)

    fig4, (ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(1, 6)
    fig4.set_size_inches(18.6, 12.3)

    fig5, (ax13, ax14, ax15, ax16, ax17, ax18) = plt.subplots(1, 6)
    fig5.set_size_inches(18.6, 12.3)

    fig6, (ax19, ax20, ax21, ax22, ax23) = plt.subplots(1, 5)
    fig6.set_size_inches(15.6, 12.3)




    for scenario, output_sw_n in zip(scenarios, output_sw):
        data[str("Sw_" + scenario)] = [output_sw_n] * len(data)

        #print(list(zip(data[str("K_Matrix")], data["Md ft"])))
        #print(list(zip(data[str("Sw_" + scenario)], data["Md ft"])))
        #print(list(zip(data["K_pore_bounded"], data["Md ft"])))
        #print(list(zip(data["Rhob_dry"], data["Md ft"])))

        data[str("K_fluid_" + scenario)], data[str("RhoB_fluid_" + scenario)], data[str("K_" + scenario)], data[str("RhoB_" + scenario)], data[str("K_norm_" + scenario)] = multiple_FRM(data["PhiE"].tolist(), data[str("Sw_" + scenario)].tolist(), data["K_Matrix"].tolist(), data["K_pore_bounded"].tolist(), data["Rhob_dry"].tolist(), parameters["Modelled Pressure (psi)"] * 0.00689476, parameters["Modelled Temperature C"], output_fluid, parameters["Salinity (l/l)"], parameters["Gas Gravity"], parameters["Oil API"], parameters["GOR (scf/bbl)"])

        #print (list(zip(data[str("K_fluid_" + scenario)], data["Md ft"])))

        filter = (data["Vsh"] > frm_max_vsh) | (data["Md"] < parameters["Top ZOI (ft MD KB)"]) | (data["Md"] > parameters["Base ZOI (ft MD KB)"]) | (data["PhiE"] > frm_max_phie) | (data["PhiE"] < frm_min_phie)
        #print (filter)
        data[str("K_" + scenario)] = data["K_sat"].where(filter, data[str("K_" + scenario)])
        data[str("K GPa_" + scenario)] =  data[str("K_" + scenario)] / np.power(10, 9)
        data[str("RhoB_" + scenario)] = data["RhoB"].where(filter, data[str("RhoB_" + scenario)])
        #print (data[str("Sw_" + scenario)])
        data[str("Sw_" + scenario)] = data["Sw"].where(filter, data[str("Sw_" + scenario)])
        #print(data[str("Sw_" + scenario)])
        data[str("Vp_" + scenario)], data[str("Vs_" + scenario)] = calc_vp_vs(data[str("K_" + scenario)], data["Mu_sat"], data[str("RhoB_" + scenario)])

        data[str("Vp/Vs_" + scenario)] = data[str("Vp_" + scenario)] / data[str("Vs_" + scenario)]
        data[str("AI_" + scenario)] = data[str("Vp_" + scenario)] * data[str("RhoB_" + scenario)]
        data[str("SI_" + scenario)] = data[str("Vs_" + scenario)] * data[str("RhoB_" + scenario)]

        data["Poisson's Ratio_" + scenario] = (3 * data["K_" + scenario] - 2 * data["Mu_sat"]) / (2 * (3 * data["K_" + scenario] + data["Mu_sat"]))
        data["Lambda_" + scenario] = ((data["K_"  + scenario] - 2 / 3 * data["Mu_sat"])) / np.power(10, 9)
        data["Lambda-Rho_" + scenario] = data["Lambda_" + scenario] * data["RhoB_" + scenario]
        data["Mu-Rho_" + scenario] = data["Mu_sat GPa"] * data["RhoB_" + scenario]

        if save_las == True:
            with open(root + "\\" + str(well) + "_" + str(scenario) + '.las', mode="w") as las2:
                # las_2 = open(las2_out, mode="w")
                frm = lasio.LASFile()
                frm.depth = data["Md"]
                frm.well["WELL"].value = str(well)
                frm.add_curve("Md", data["Md"], unit=units_dict.get("Md", "ft"))
                frm.add_curve("Bulk Modulus", data["K GPa_" + scenario], unit="GPa")
                frm.add_curve("Shear Modulus", data["Mu_sat GPa"], unit="GPa")
                frm.add_curve("Vp/Vs", data["Vp/Vs_" + scenario], unit="ratio")
                frm.add_curve("AI", data["AI_" + scenario], unit="m/s*g/cc")
                frm.add_curve("SI", data["SI_" + scenario], unit="m/s*g/cc")
                frm.add_curve("Poisson's Ratio", data["Poisson's Ratio_" + scenario], unit="ratio")
                frm.add_curve("Lambda-Rho", data["Lambda-Rho_" + scenario], unit="GPa")
                frm.add_curve("Mu-Rho", data["Mu-Rho_" + scenario], unit="GPa")
                frm.add_curve("Vp", data["Vp_"+ scenario], unit = "m/s")
                frm.add_curve("Vs", data["Vs_" + scenario], unit = "m/s")
                frm.add_curve("RhoB", data["RhoB_"+ scenario], unit = "g/cc")
                frm.add_curve("PhiE", data["PhiE"], unit = "frac")
                frm.add_curve("PhiT", data["PhiT"], unit = "frac")
                frm.add_curve("Vsh", data["Vsh"], unit = "frac")
                frm.add_curve("Sw", data["Sw_" + scenario], unit = "frac")
                frm.write(las2, version=2, fmt='%10.5g')

        #print((data["Vp_" + scenario]))
        #print(data)
        #print (type(data["Md ft"]))
        #print


        for ax, log in zip( [ax1, ax2, ax3], ["Vp", "Vs", "RhoB"]):
            ax.plot((data[str(log + "_" + scenario)]), data["Md"], lw = 1, label = str(log + "_" + scenario))  # , label = scenario, lw = 1)
            ax.set_title(log)
            ax.set_xlabel(log + " " + units_dict.get(log,""), fontsize=8)

        for ax, log in zip( [ax7, ax9], ["K GPa", "RhoB"]):
            ax.plot((data[str(log + "_" + scenario)]), data["Md"], lw = 1, label = str(log + "_" + scenario))  # , label = scenario, lw = 1)
            ax.set_title(log)
            ax.set_xlabel(log + " " + units_dict.get(log,""), fontsize=8)



        for ax, log in zip([ax13, ax14, ax15], ["Vp/Vs", "AI", "SI"]):
            ax.plot((data[str(log + "_" + scenario)]), data["Md"], lw=1, label = str(log + "_" + scenario) )  # , label = scenario, lw = 1)
            ax.set_title(log)
            ax.set_xlabel(log + " " + units_dict.get(log, ""), fontsize=8)

        for ax, log in zip([ax19, ax20], ["Lambda-Rho", "Mu-Rho"]):
            ax.plot((data[str(log + "_" + scenario)]), data["Md"], lw=1, label = str(log + "_" + scenario))  # , label = scenario, lw = 1)
            ax.set_title(log)
            ax.set_xlabel(log + " " + units_dict.get(log, ""), fontsize=8)

        for ax in [ax4, ax10, ax16, ax21]:
            ax.set_xlim(0, 1)
        for ax in [ax5, ax11, ax17, ax22]:
            ax.set_xlim(0, 1)
        for ax in [ax6, ax12, ax18, ax23]:
            ax.set_xlim(0, 0.5)

    for ax, log in zip([ax4, ax5, ax6, ax10, ax11, ax12, ax16, ax17, ax18, ax21, ax22, ax23, ax8],
                       ["Vsh", "Sw", "PhiE", "Vsh", "Sw", "PhiE", "Vsh", "Sw", "PhiE", "Vsh", "Sw", "PhiE",
                        "Mu_sat"]):
        ax.plot((data[log]), data["Md"], lw=1,
                label=str(log))  # , label = scenario, lw = 1)
        ax.set_title(log)
        ax.set_xlabel(log + " " + units_dict.get(log, ""), fontsize=8)


    for ax in (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19, ax20, ax21, ax22, ax23):
        ax.grid()
        ax.invert_yaxis()
        ax.set_ylim(parameters["Base ZOI (ft MD KB)"] + 300, parameters["Top ZOI (ft MD KB)"] - 300)
        ax.set_ylabel("MD " + units_dict.get("Md", "ft"), fontsize=8)
        ax.legend(loc='lower left', shadow=True, fontsize='x-small')
        ax.tick_params(axis='both', which='major', labelsize=8)

    fig3_name = (root + "\\" + well + '_fig3_Vp_Vs_RhoB_FRM.png')
    fig4_name = (root + "\\" + well + '_fig4_K_Mu_RhoB_FRM.png')
    fig5_name = (root + "\\" + well + '_fig5_VpVs_AI_SI_FRM.png')
    fig6_name = (root + "\\" + well + '_fig6_LMR.png')
    for fig in [fig3, fig4, fig5, fig6]:
        fig.suptitle(well)
        fig.subplots_adjust(wspace = 0.3, hspace = 0.3, top = 0.90, left = 0.05, right = 0.95, bottom = 0.05)
        #fig.tight_layout()

    if save_fig == True:
        for fig, name in zip([fig3, fig4, fig5, fig6], [fig3_name, fig4_name, fig5_name,fig6_name]):
            fig.savefig(name, dpi=400, bbox_inches="tight")

    if show_charts == True:
        plt.show()






