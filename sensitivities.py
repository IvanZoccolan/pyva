
import numpy as np
import pandas as pd
from processes import CGMYProcess, CirMortality, Gompertz, GBMProcess, VGProcess, MJDProcess
from glwb_contract import GLWB
import mu_ini_pars


cgmy = CGMYProcess(c=0.02, g=5, m=15, y=1.2)
cir = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.01, mu0=0.009954829)  # IPS55M
# Sensitivies on mortality

# IPS55M - Sensitivity on sigma

cir1 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.002, mu0=0.009954829)
cir2 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.004, mu0=0.009954829)
cir3 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.006, mu0=0.009954829)
cir4 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.008, mu0=0.009954829)
cir5 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.012, mu0=0.009954829)
cir6 = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.014, mu0=0.009954829)
gompertz = Gompertz(theta=0.09782020, mu0=0.01076324)  # IPS55M
####

# A62M
cir7 = CirMortality(alpha=0.000100000, theta=0.1142278, sigma=0.01, mu0=0.006189726)

# HMD Italy M 1x10 2000-2009

cir8 = CirMortality(alpha=0.000100000, theta=0.1105087, sigma=0.01, mu0=0.01212986)

# Sensitivity on financial processes

gbm = GBMProcess()
vg = VGProcess()
mjd = MJDProcess()

df = pd.DataFrame({"Sens":
                       ["IPS55M - 0.2%",
                        "IPS55M - 0.4%",
                        "IPS55M - 0.6%",
                        "IPS55M - 0.8%",
                        "IPS55M - 1.2%",
                        "IPS55M - 1.4%",
                        "A62M",
                        "HMD",
                        "GBM",
                        "VG",
                        "MJD"
                        ],
                   "V0": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]})

# Initial Parameters

# IPS55M sigma = 0.2%
glwb = GLWB(cgmy, cir1)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu1]

df.iloc[0, 1] = glwb.price("dynamic")

# IPS55M sigma = 0.4%
glwb = GLWB(cgmy, cir2)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu2]

df.iloc[1, 1] = glwb.price("dynamic")

# IPS55M sigma = 0.6%
glwb = GLWB(cgmy, cir3)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu3]

df.iloc[2, 1] = glwb.price("dynamic")

# IPS55M sigma = 0.8%
glwb = GLWB(cgmy, cir4)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu4]

df.iloc[3, 1] = glwb.price("dynamic")

# IPS55M sigma = 1.2%
glwb = GLWB(cgmy, cir5)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu5]

df.iloc[4, 1] = glwb.price("dynamic")

# IPS55M sigma = 1.4%
glwb = GLWB(cgmy, cir6)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu6]

df.iloc[5, 1] = glwb.price("dynamic")

# A62M
glwb = GLWB(cgmy, cir7)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu7]

df.iloc[6, 1] = glwb.price("dynamic")

# HMD
glwb = GLWB(cgmy, cir8)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu8]

df.iloc[7, 1] = glwb.price("dynamic")

# Sens on Financial process - GBM
glwb = GLWB(gbm, cir)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu]

df.iloc[8, 1] = glwb.price("dynamic")

# Sens on Financial process - VG
glwb = GLWB(vg, cir)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu]

df.iloc[9, 1] = glwb.price("dynamic")

# Sens on Financial process - MJD
glwb = GLWB(mjd, cir)

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.005)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu_ini_pars.mu]

df.iloc[10, 1] = glwb.price("dynamic")

df.to_csv("sensitivities.csv", index=False)