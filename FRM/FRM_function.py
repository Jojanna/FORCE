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

' testing the branching...'


import numpy as np
import pandas as pd
from fluid import batzle_wang


# calculate density of dry rock
def rho_dry_calc (rho_data, rho_f_ini, phie_data):
    rho_dry = []
    for rho_rock, rho_f, p in zip(rho_data, rho_f_ini, phie_data):
        rho_dry.append(rho_rock - p * rho_f)

    return rho_dry

# calculate density of wet rock
def rho_out_calc(rho_dry, phie_data, rho_f_out):
    rho_out = []
    for rhod, phie, rhof in zip(rho_dry, phie_data, rho_f_out):
        rho_out.append(rhod + phie * rhof)

    return rho_out

# 2 phase hashin schtrickman
def HS_2_phase_L(k_qtz, k_clay, mu_qtz, mu_clay, vsh):
    k_max = k_qtz
    mu_max = mu_qtz
    k_min = k_clay
    mu_min = mu_clay

    k_ma = []
    mu_ma = []

    for n in vsh:
        f_min = n
        f_max = 1 - n

        k_ma_n = k_min + (f_max / (pow((k_max - k_min), -1) + f_min * pow((k_min + 4 / 3 * mu_min), -1)))
        mu_ma_n = mu_min + (
        f_max / pow((mu_max - mu_min), -1) + f_min * 2 * (k_min + 2 * mu_min) / (5 * mu_min * (k_min + 4 / 3 * mu_min)))

        k_ma.append(k_ma_n)
        mu_ma.append(mu_ma_n)

    return k_ma, mu_ma

#calculate fluid parameters
def fluids_calc(pressure, temp, fl, s, g, api_oil, ratio):

    rho_brine, k_brine = batzle_wang(pressure, temp, 'brine', S=s, G=g, api=api_oil, Rg=ratio)
    # print(rho_brine)
    if fl == 'oil':
        rho_h, k_h = batzle_wang(pressure, temp, 'oil', S=s, G=g, api=api_oil, Rg = ratio)
    elif fl == 'gas':
        rho_h, k_h = batzle_wang(pressure, temp, 'gas', S=s, G=g, api=api_oil,Rg = ratio)
    else:
        print ('check fluids!')
        rho_h, k_h = 0, 0
        exit(0)

    return rho_brine, k_brine, rho_h, k_h

# calculate mixed fluid logs
def reuss_fluids(rho_brine, k_brine, rho_h, k_h, sw_data):
    k_f = []
    rho_f = []
    for s in sw_data:
            #s_we = 2 * np.exp(-11 * p)
            #print (s_we)
            k_f_s = 1/(s/k_brine + (1 - s)/k_h) # Reuss average
            k_f.append(k_f_s)
            rho_f_p = s* rho_brine + (1 - s) * rho_h
            rho_f.append(rho_f_p)

    return k_f, rho_f

#calculate saturated rock properties

def sat(vp_data, vs_data, rho_data,):
    mu_sat = []
    k_sat = []

    for vp_n, vs_n, rho_n in zip(vp_data, vs_data, rho_data):
        mu_n = np.power(vs_n, 2) * (rho_n * 1000)
        k_n = np.power(vp_n, 2) * (rho_n * 1000) - (mu_n * 4 / 3)


        mu_sat.append(mu_n)
        k_sat.append(k_n)
    return k_sat, mu_sat

# Invert for dry rock parameters
def dry_rock(phie_data,k_sat,k_f, k_ma):
    k_pore_data = []
    k_d_data = []

    pore_params = zip(phie_data,k_sat,k_f, k_ma)

    for phie_n, k_sat_n, k_f_n, k_ma_n in pore_params:
        k_pore_n = phie_n /(1/k_sat_n - 1/k_ma_n) - 1 / (1/(k_f_n) - 1/k_ma_n)
        k_pore_data.append(k_pore_n)

    dry_params = zip(phie_data, k_pore_data, k_ma)
    for phie_n, k_pore_n, k_ma_n in dry_params:
        k_d_n = 1/(phie_n/k_pore_n + 1/k_ma_n)
        k_d_data.append(k_d_n)

    return k_pore_data, k_d_data

# fluid substitution

def fluid_sub_k(k_ma, phie_data, k_pore_data, k_f):

    sub_params = zip(k_ma, phie_data, k_pore_data, k_f)
    k_out = []
    #phie_exc_0 = []
    for k_ma_n, phie_n, k_pore_n, k_f_n in sub_params:
        if phie_n > 0:
            k_out_n = 1 / (1/k_ma_n + phie_n/(k_pore_n + (k_ma_n * k_f_n)/(k_ma_n - k_f_n)))
            #k_out.append(k_out_n)
            #phie_exc_0.append(phie_n)
        else:
            k_out_n = 1 / (1/k_ma_n)
        k_out.append(k_out_n)

    return k_out

# calculate rock model bounds
def find_kd_min_max(k_phi_const_max,k_phi_const_min, phie_data, k_ma ):
    kphi_max = []
    k_d_max = []
    k_d_0_max = []

    kphi_min = []
    k_d_min = []
    k_d_0_min = []

    for phi, k_ma_n in zip(phie_data, k_ma):
        kphi_max_n = (k_phi_const_max * k_ma_n) # find kphi from kphi/kma
        kphi_max.append(kphi_max_n)
        k_d_max_n = (1/(1/k_ma_n + phi/(kphi_max_n)))/k_ma_n # find Kd from k_ma, phie and k_phi
        k_d_max.append(k_d_max_n)
        k_d_0_max.append(k_d_max_n / k_ma_n) #find kd/k0 aka kd/ma aka normalised kd

        kphi_min_n = (k_phi_const_min * k_ma_n) # find kphi from kphi/kma
        kphi_min.append(kphi_min_n)
        k_d_min_n = (1/(1/k_ma_n + phi/(kphi_min_n)))/k_ma_n # find Kd from k_ma, phie and k_phi
        k_d_min.append(k_d_max_n)
        k_d_0_min.append(k_d_min_n / k_ma_n)#find kd/k0 aka kd/ma aka normalised kd

    return kphi_max, k_d_max, k_d_0_max, kphi_min, k_d_min, k_d_0_min


def multiple_FRM(phie_data, sw_out, k_ma, kphi_set, rho_dry, pressure_out, temp, fl_out, s, g, api_oil, ratio):

    rho_brine, k_brine, rho_h_out, k_h_out = fluids_calc(pressure_out, temp, fl_out, s, g, api_oil, ratio)
#k_f_out, rho_f_out = reuss_fluids(rho_brine, k_brine, rho_h_out, k_h_out, sw_out)
#k_pore_data, k_d_data = dry_rock(phie_data,k_sat,k_f_ini, k_ma)


    k_f_out = list(map(lambda sw_out_n: 1 / (sw_out_n / k_brine + (1 - sw_out_n) / k_h_out), sw_out))
    rho_f_out = list(map(lambda sw_out_n: sw_out_n * rho_brine + (1 - sw_out_n) * rho_h_out, sw_out))

    k_out = []
    for k_ma_n, phie_n, k_pore_n, k_f_n in zip(k_ma, phie_data, kphi_set, k_f_out):
        if phie_n > 0:
            k_out_n = 1 / (1/k_ma_n + phie_n/(k_pore_n + (k_ma_n * k_f_n)/(k_ma_n - k_f_n)))
            #k_out.append(k_out_n)
            #phie_exc_0.append(phie_n)
        else:
            k_out_n = 1 / (1/k_ma_n)
        k_out.append(k_out_n)

    rho_out = []
    for rhod, phie, rhof in zip(rho_dry, phie_data, rho_f_out):
        rho_out.append(rhod + phie * rhof)

    k_out_norm = [(k / k_ma_n) for k, k_ma_n in zip(k_out, k_ma)]

    #df_frm = pd.DataFrame()
    #df_frm["phie"], df_frm["k_ma"], df_frm["kphi_set"],df_frm["k_f_out"], df_frm["rho_f_out"], df_frm["k_out"], df_frm["rho_out"], df_frm["k_out_norm"] = phie_data, k_ma, kphi_set, k_f_out, rho_f_out, k_out, rho_out, k_out_norm
    #print(df_frm)


    return k_f_out, rho_f_out, k_out, rho_out, k_out_norm

def FRM_cutoffs(md, vsh, phie, k_out, k_sat, rho_data, rho_out, sw_out, sw_data, frm_max_vsh, frm_min_md, frm_max_md, frm_max_phie, frm_min_phie):
    k_out_cutoff = []
    rho_out_cutoff = []
    #mu_out_cutoff = []
    sw_out_cutoff = []
    for md_n, vsh_n, phie_n, k_out_n, k_sat_n, rho_data_n, rho_out_n, sw_out_n, sw_data_n in zip(md, vsh, phie, k_out, k_sat, rho_data, rho_out, sw_out, sw_data):
        if (vsh_n > frm_max_vsh or md_n < frm_min_md or md_n > frm_max_md or phie_n > frm_max_phie or phie_n < frm_min_phie):
            k_out_cutoff.append(k_sat_n)
            rho_out_cutoff.append(rho_data_n)
            sw_out_cutoff.append(sw_data_n)
        else:
            k_out_cutoff.append(k_out_n)
            rho_out_cutoff.append(rho_out_n)
            sw_out_cutoff.append(sw_out_n)

    return k_out_cutoff, rho_out_cutoff, sw_out_cutoff

def calc_vp_vs(k, mu, rho):
    vp = []
    vs = []
    for k_n, mu_n, rho_n in zip(k, mu, rho):
        vs.append(pow(mu_n/(rho_n * 1000), 0.5))
        vp.append(pow((mu_n * 4/3 + k_n)/(rho_n * 1000),0.5))
    return vp, vs