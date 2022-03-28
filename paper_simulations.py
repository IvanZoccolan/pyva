
# Script to calculate the tables for paper  BMZ 2022 - 24-2-2022

import numpy as np
from processes import CGMYProcess, CirMortality, Gompertz
from glwb_contract import GLWB
from glwb_dm import GLWB_DM
import pandas as pd

cgmy = CGMYProcess(c=0.02, g=5, m=15, y=1.2)
cir = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.01, mu0=0.009954829)  # IPS55M
gompertz = Gompertz(theta=0.09782020, mu0=0.01076324)  # IPS55M
glwb = GLWB(cgmy, cir)
# glwb_dm = GLWB_DM(cgmy, gompertz)
# Initial Parameters

mu_space = np.exp(gompertz.theta * np.arange(0, 54)) * gompertz.mu0
glwb.set_mu_space(mu_space)

glwb.set_maturity(53)
glwb.set_spot(0.02)
glwb.set_rollup(0.08)
glwb.set_g_rate(0.03)
glwb.penalty = 0.05
glwb.set_fee(0.05)

#  rollup = 6%, 7%, 8%, 9%, 10% with  penalty=5%, g_rate=3%, spot=2% e fee=10%

glwb.set_fee(0.1)

df = pd.DataFrame({"Rup": [0.06, 0.07, 0.08, 0.09, 0.1], "Static": [0, 0, 0, 0, 0], "Mixed": [0, 0, 0, 0, 0], "Dynamic": [0, 0, 0, 0, 0]})

for rup in [0.06, 0.07, 0.08, 0.09, 0.1]:
    glwb.set_rollup(rup)
    print(f'Rup = {rup}, Method: Static\n')
    df.loc[df["Rup"] == rup, "Static"] = glwb.price()
    print(f'Rup = {rup}, Method: Static - DONE\n')
    print(f'Rup = {rup}, Method: Mixed\n')
    df.loc[df["Rup"] == rup, "Mixed"] = glwb.price("mixed")
    print(f'Rup = {rup}, Method: Mixed - DONE\n')
    print(f'Rup = {rup}, Method: Dynamic\n')
    df.loc[df["Rup"] == rup, "Dynamic"] = glwb.price("dynamic")
    print(f'Rup = {rup}, Method: Dynamic - DONE\n')

df.to_csv("rup_paper.csv", index=False)


# Penalty = 0%, 3%, 5%, 7%, 9%  rollup=8%, g_rate=3%, spot=2% e fee=10%

glwb.set_rollup(0.08)

df = pd.DataFrame({"Penalty:": [0.0, 0.03, 0.05, 0.07, 0.09], "Static": [0, 0, 0, 0, 0], "Mixed": [0, 0, 0, 0, 0], "Dynamic": [0, 0, 0, 0, 0]})

for pe in [0.0, 0.03, 0.05, 0.07, 0.09]:
    glwb.penalty = pe
    print(f'Penalty = {pe}, Method: Static\n')
    df.loc[df["Penalty"] == pe, "Static"] = glwb.price()
    print(f'Penalty = {pe}, Method: Static - DONE\n')
    print(f'Penalty = {pe}, Method: Mixed\n')
    df.loc[df["Penalty"] == pe, "Mixed"] = glwb.price("mixed")
    print(f'Penalty = {pe}, Method: Mixed - DONE\n')
    print(f'Penalty = {pe}, Method: Dynamic\n')
    df.loc[df["Penalty"] == pe, "Dynamic"] = glwb.price("dynamic")
    print(f'Penalty = {pe}, Method: Dynamic - DONE\n')

df.to_csv("penalty_paper.csv", index=False)

#  g_rate = 1%, 2%, 3%, 4%, 5%  rollup=8%, penalty=5%, spot=2%,  fee=10%

rates = [0.01, 0.02, 0.03, 0.04, 0.05]
df = pd.DataFrame({"g_rate": rates, "Static": np.zeros(len(rates)), "Mixed": np.zeros(len(rates)), "Dynamic": np.zeros(len(rates))})

glwb.penalty = 0.05

for rate in rates:
    glwb.set_g_rate(rate)
    print(f'G-Rate = {rate}, Method: Static\n')
    df.loc[df["g_rate"] == rate, "Static"] = glwb.price()
    print(f'G-Rate = {rate}, Method: Static - DONE. Value: {glwb._price}\n')
    print(f'G-Rate = {rate}, Method: Mixed\n')
    df.loc[df["g_rate"] == rate, "Mixed"] = glwb.price("mixed")
    print(f'G-Rate = {rate}, Method: Mixed - DONE. Value: {glwb._price}\n')
    print(f'G-Rate = {rate}, Method: Dynamic\n')
    df.loc[df["g_rate"] == rate, "Dynamic"] = glwb.price("dynamic")
    print(f'G-Rate = {rate}, Method: Dynamic - DONE. Value: {glwb._price}\n')

df.to_csv("g_rate_paper.csv", index=False)

# Spot= 1%, 2%, 3.5%, 5%  and rollup=8%, penalty=5%, g_rate=3% e fee=10%

glwb.set_g_rate(0.03)

spots = [0.01, 0.02, 0.035, 0.05]
df = pd.DataFrame({"spot": spots, "Static": np.zeros(len(spots)), "Mixed": np.zeros(len(spots)), "Dynamic": np.zeros(len(spots))})

glwb.penalty = 0.05

for spot in spots:
    glwb.set_spot(spot)
    print(f'Spot = {spot}, Method: Static\n')
    df.loc[df["spot"] == spot, "Static"] = glwb.price()
    print(f'Spot = {spot}, Method: Static - DONE. Value: {glwb._price}\n')
    print(f'Spot = {spot}, Method: Mixed\n')
    df.loc[df["spot"] == spot, "Mixed"] = glwb.price("mixed")
    print(f'Spot = {spot}, Method: Mixed - DONE. Value: {glwb._price}\n')
    print(f'Spot = {spot}, Method: Dynamic\n')
    df.loc[df["spot"] == spot, "Dynamic"] = glwb.price("dynamic")
    print(f'Spot = {spot}, Method: Dynamic - DONE. Value: {glwb._price}\n')

df.to_csv("spot_paper.csv", index=False)
