
# Script to calculate the tables for paper  BMZ 2022 - 3-7-2022

import numpy as np
from processes import CGMYProcess, CirMortality, Gompertz
from glwb_contract import GLWB
import pandas as pd

cgmy = CGMYProcess(c=0.02, g=5, m=15, y=1.2)
cir = CirMortality(alpha=0.000100000, theta=0.100687520, sigma=0.01, mu0=0.009954829)  # IPS55M
gompertz = Gompertz(theta=0.09782020, mu0=0.01076324)  # IPS55M
glwb = GLWB(cgmy, cir)

# Initial Parameters

mu_space = np.exp(gompertz.theta * np.arange(0, 54)) * gompertz.mu0
glwb.set_mu_space(mu_space)

glwb.set_maturity(53)
glwb.set_spot(0.02)
glwb.set_rollup(0.08)
glwb.set_g_rate(0.03)
glwb.penalty = 0.05
glwb.set_fee(0.1)


# Table with only "dynamic" varying roll-up and withdrawal rate

rollups = np.arange(0.01, 0.11, 0.01)
g_rates = np.arange(0.01, 0.11, 0.01)

df = pd.DataFrame(np.zeros((len(rollups), len(g_rates))), index=rollups, columns=g_rates)

for k, rup in enumerate(rollups):
    for j, grate in enumerate(g_rates):
        glwb.set_rollup(rup)
        glwb.set_g_rate(grate)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("rollups_grates.csv")

# Increase penalty until static = mixed;

glwb.set_rollup(0.08)
glwb.set_g_rate(0.03)

penalties = np.arange(0.6, 0.9, 0.1)
df = pd.DataFrame({"Penalty": penalties, "Static": np.zeros(len(penalties)), "Mixed": np.zeros(len(penalties)), "Dynamic": np.zeros(len(penalties))})

for pe in penalties:
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

df.to_csv("penalty_paper_3.csv", index=False)

#### Increase spot rate until dynamic is fair with the fee = 10%.

glwb.penalty = 0.05

spots = np.arange(0.11, 0.22, 0.02)
df = pd.DataFrame({"spot": spots, "Static": np.zeros(len(spots)), "Mixed": np.zeros(len(spots)), "Dynamic": np.zeros(len(spots))})

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

df.to_csv("spot_paper_3.csv", index=False)


# Graph with fee varying from 0% to 60%

glwb.set_spot(0.02)

fees = np.arange(0.00, 0.65, 0.05)

df = pd.DataFrame({"fee": fees, "Static": np.zeros(len(fees)), "Mixed": np.zeros(len(fees)), "Dynamic": np.zeros(len(fees))})

for fee in fees:
    glwb.set_fee(fee.item())
    print(f'fee = {fee}, Method: Static\n')
    df.loc[df["fee"] == fee, "Static"] = glwb.price()
    print(f'fee = {fee}, Method: Static - DONE. Value: {glwb._price}\n')
    print(f'fee = {fee}, Method: Mixed\n')
    df.loc[df["fee"] == fee, "Mixed"] = glwb.price("mixed")
    print(f'fee = {fee}, Method: Mixed - DONE. Value: {glwb._price}\n')
    print(f'fee = {fee}, Method: Dynamic\n')
    df.loc[df["fee"] == fee, "Dynamic"] = glwb.price("dynamic")
    print(f'fee = {fee}, Method: Dynamic - DONE. Value: {glwb._price}\n')

df.to_csv("fees_paper.csv", index=False)

# Graph with only dynamic, fees = 0% - 60%, 3 penalties.

penalties = [0.03, 0.05, 0.07]

df = pd.DataFrame(np.zeros((len(fees), len(penalties))), index=fees, columns=penalties)

for k, fee in enumerate(fees):
    for j, penalty in enumerate(penalties):
        glwb.set_fee(fee.item())
        glwb.penalty = penalty
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("fees_penalties.csv")

# Graph with only dynamic, fees = 0% - 60%, 3 rollups.

glwb.penalty = 0.05

rollups = [0.07, 0.08, 0.09]

df = pd.DataFrame(np.zeros((len(fees), len(rollups))), index=fees, columns=rollups)

for k, fee in enumerate(fees):
    for j, rup in enumerate(rollups):
        glwb.set_fee(fee.item())
        glwb.set_rollup(rup)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("fees_rollups.csv")
