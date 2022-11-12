
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


mu = (
    (0.009954829, 0.009954829),
    (0.007930135, 0.014254560),
    (0.00727352, 0.01654696),
    (0.00695167, 0.02191768),
    (0.00753759, 0.02312999),
    (0.007187134, 0.026452391),
    (0.008147259, 0.030864219),
    (0.008448751, 0.034537717),
    (0.00906032, 0.03772911),
    (0.009686219, 0.043404986),
     (0.01224217, 0.04801214),
    (0.01243797, 0.05643224),
     (0.01492384, 0.06423226),
    (0.01441666, 0.07429595),
    (0.01736866, 0.07783029),
     (0.01734821, 0.08698642),
    (0.01888331, 0.09701530),
     (0.01887998, 0.11247654),
     (0.02305054, 0.11793394),
     (0.02747526, 0.13237303),
     (0.03021457, 0.15116043),
    (0.03182803, 0.16904134),
    (0.03618297, 0.18726290),
     (0.03663573, 0.20279493),
     (0.04251158, 0.22851837),
     (0.04330336, 0.26179542),
    (0.04931387, 0.27740327),
    (0.05121207, 0.30675494),
    (0.05409595, 0.33304990),
     (0.0566399, 0.3724727),
     (0.0599393, 0.4077251),
     (0.0649323, 0.4473628),
     (0.0713030, 0.4995644),
    (0.08102182, 0.54098441),
     (0.08744914, 0.59660920),
    (0.09528977, 0.66074613),
     (0.1062206, 0.7289570),
     (0.1167617, 0.7929744),
     (0.1258137, 0.8721379),
     (0.1356165, 0.9544155),
     (0.1533646, 1.0438879),
     (0.1696114, 1.1475031),
     (0.1881631, 1.2730895),
    (0.2111575, 1.3955704),
     (0.2296044, 1.5412792),
    (0.2474077, 1.7069442),
    (0.2724226, 1.8784335),
     (0.298322, 2.063321),
     (0.3165176, 2.2726621),
     (0.3439983, 2.5088967),
     (0.3811667, 2.7765041),
    (0.417627, 3.075570),
     (0.4555933, 3.3956352),
     (0.5069924, 3.7515802)
)

glwb._mu_space = [np.linspace(begin, end, glwb.num_mu_pts) for begin, end in mu]

glwb.set_maturity(53)
glwb.set_spot(0.03)
glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)
glwb.penalty = 0.02
glwb.set_fee(0.0050)


# Table with only "dynamic" varying roll-up and withdrawal rate

rollups = np.arange(0.02, 0.12, 0.02)
g_rates = [0.06]

df = pd.DataFrame(np.zeros((len(rollups), len(g_rates))), index=rollups, columns=g_rates)

for k, rup in enumerate(rollups):
    for j, grate in enumerate(g_rates):
        glwb.set_rollup(rup)
        glwb.set_g_rate(grate)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("rollups_grates_2.csv")


# Graph with fee varying from 0% to 1%

glwb.set_rollup(0.06)
glwb.set_g_rate(0.05)

fees = np.arange(0.00, 0.0105, 0.001)

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

# Graph with only dynamic, fees = 0% - 2%, 3 penalties.

fees = np.arange(0.00, 0.0205, 0.001)
penalties = [0.0, 0.02, 0.04]

df = pd.DataFrame(np.zeros((len(fees), len(penalties))), index=fees, columns=penalties)

for k, fee in enumerate(fees):
    for j, penalty in enumerate(penalties):
        glwb.set_fee(fee.item())
        glwb.penalty = penalty
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("fees_penalties.csv")

# Graph with only dynamic, fees = 0% - 2%, 3 rollups.

glwb.set_fee(0.0050)
glwb.penalty = 0.02

fees = np.arange(0.00, 0.0205, 0.001)
rollups = [0.04, 0.06, 0.08]

df = pd.DataFrame(np.zeros((len(fees), len(rollups))), index=fees, columns=rollups)

for k, fee in enumerate(fees):
    for j, rup in enumerate(rollups):
        glwb.set_fee(fee.item())
        glwb.set_rollup(rup)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("fees_rollups.csv")

glwb.set_rollup(0.06)
glwb.set_fee(0.0050)


# Grafico del valore del contratto dinamico con una fee da 0 al 3% e in cui g=4,5,6%

fees = np.arange(0.00, 0.0305, 0.001)
g_rates = [0.04, 0.05, 0.06]

df = pd.DataFrame(np.zeros((len(fees), len(g_rates))), index=fees, columns=g_rates)

for k, fee in enumerate(fees):
    for j, g in enumerate(g_rates):
        glwb.set_fee(fee.item())
        glwb.set_g_rate(g)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("fees_g_rates.csv")

# tabella a doppia entrata (k, fee) con k=0,1,2,3,4,5 e fee= 0.2%, 0.4%, 0.6%, 0.8%, 1%.

glwb.set_fee(0.0050)
glwb.set_g_rate(0.05)

penalties = [0, 0.01, 0.02, 0.03, 0.04, 0.05]
fees = [0.002, 0.004, 0.006, 0.008, 0.01]

df = pd.DataFrame(np.zeros((len(penalties), len(fees))), index=penalties, columns=fees)

for k, pen in enumerate(penalties):
    for j, fee in enumerate(fees):
        glwb.penalty = pen
        glwb.set_fee(fee)
        df.iloc[k, j] = glwb.price("dynamic")

df.to_csv("penalties_fees.csv")