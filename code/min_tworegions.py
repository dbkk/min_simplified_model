#%%
import numpy as np
import matplotlib.pyplot as plt

# Components
# A: minD in cytoplasm
# B: active-minD in cytoplasm
# C: minD in membrane
# E: minE in cytoplasm
# F: minE in membrane

# Define the rate constants
k1 = 0.1  # Rate constant for A -> B
km1 = 0.00  # Rate constant for B -> A
k2 = 2.0  # Rate constant for C -> A
km2 = 0.00  # Rate constant for A -> C
k3 = 10.0  # Rate constant for B -> C
km3 = 0.00  # Rate constant for C -> B
k4 = 15.0  # Rate constant for E -> F 
km4 = 0.3 # Rate constant for F -> E

# Rate constants for A1 <-> A2, B1 <-> B2, C1 <-> C2, E1 <-> E2, F1 <-> F2 (diffusion)

d = 1.0 # just a scaling factor
DA = 0.1 * d
DB = 0.1 * d
DE = 0.1 * d

DC = 0.0
DF = 0.0

# Initial concentrations
A1, A2 = 0.0, 0.0
B1, B2 = 0.0, 0.0
C1, C2 = 0.6, 0.4
E1, E2 = 1.0, 1.0
F1, F2 = 0.5, 0.5

# Time parameters
dt = 0.01  # Time step
t_end = 400  # End time
time_points = np.arange(0, t_end, dt)

# Arrays to store concentrations over time
A1_vals, A2_vals = [], []
B1_vals, B2_vals = [], []
C1_vals, C2_vals = [], []
E1_vals, E2_vals = [], []
F1_vals, F2_vals = [], []

# Simulation loop
for t in time_points:
    # Calculate reaction rates for the original reactions
    rA1B1 = k1 * A1
    rA2B2 = k1 * A2
    rB1A1 = km1 * B1
    rB2A2 = km1 * B2
    rC1A1 = k2 * C1 * E1
    rC2A2 = k2 * C2 * E2
    rA1C1 = km2 * A1
    rA2C2 = km2 * A2
    rB1C1 = k3 * B1 * C1
    rB2C2 = k3 * B2 * C2
    rC1B1 = km3 * C1
    rC2B2 = km3 * C2
    
    # Calculate reaction rates for the linear transformations
    rA1A2 = DA * A1
    rA2A1 = DA * A2
    rB1B2 = DB * B1
    rB2B1 = DB * B2
    rC1C2 = DC * C1
    rC2C1 = DC * C2
    
    # Calculate reaction rates for the new reactions
    rE1F1 = k4 * E1 * C1
    rE2F2 = k4 * E2 * C2
    rE1E2 = DE * E1
    rE2E1 = DE * E2
    
    # New reactions F1 -> E1 and F2 -> E2
    rF1E1 = km4 * F1
    rF2E2 = km4 * F2
    
    # Calculate reaction rates for the conversions between F1 and F2
    rF1F2 = DF * F1
    rF2F1 = DF * F2
    
    # Update concentrations using Euler's method
    dA1 = -rA1B1 + rB1A1 - rA1C1 + rC1A1 - rA1A2 + rA2A1 # rAC, rBA terms are zero
    dA2 = -rA2B2 + rB2A2 - rA2C2 + rC2A2 + rA1A2 - rA2A1
    dB1 = rA1B1 - rB1A1 - rB1C1 + rC1B1 - rB1B2 + rB2B1 # rCB, rBA terms are zero
    dB2 = rA2B2 - rB2A2 - rB2C2 + rC2B2 + rB1B2 - rB2B1
    dC1 = rA1C1 - rC1A1 + rB1C1 - rC1B1 - rC1C2 + rC2C1 # rAC, rCB, rCC terms are zero
    dC2 = rA2C2 - rC2A2 + rB2C2 - rC2B2 + rC1C2 - rC2C1
    dE1 = -rE1F1 - rE1E2 + rE2E1 + rF1E1
    dE2 = -rE2F2 + rE1E2 - rE2E1 + rF2E2
    dF1 = rE1F1 - rF1E1 - rF1F2 + rF2F1 # rFF terms are zero
    dF2 = rE2F2 - rF2E2 + rF1F2 - rF2F1
    
    A1 += dA1 * dt
    A2 += dA2 * dt
    B1 += dB1 * dt
    B2 += dB2 * dt
    C1 += dC1 * dt
    C2 += dC2 * dt
    E1 += dE1 * dt
    E2 += dE2 * dt
    F1 += dF1 * dt
    F2 += dF2 * dt
    
    # Store the concentrations
    A1_vals.append(A1)
    A2_vals.append(A2)
    B1_vals.append(B1)
    B2_vals.append(B2)
    C1_vals.append(C1)
    C2_vals.append(C2)
    E1_vals.append(E1)
    E2_vals.append(E2)
    F1_vals.append(F1)
    F2_vals.append(F2)

# Plotting the results, 1 and 2 in two plots vertically
plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(time_points, A1_vals, label='A1')
plt.plot(time_points, B1_vals, label='B1')
plt.plot(time_points, C1_vals, label='C1')
plt.plot(time_points, E1_vals, label='E1')
plt.plot(time_points, F1_vals, label='F1')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.subplot(2, 1, 2)
plt.plot(time_points, A2_vals, label='A2')
plt.plot(time_points, B2_vals, label='B2')
plt.plot(time_points, C2_vals, label='C2')
plt.plot(time_points, E2_vals, label='E2')
plt.plot(time_points, F2_vals, label='F2')
plt.legend()


#%%
# plot C1 and C2, with the labels being [D_m]_L and [D_m]_R
# plot F1 and F2, with the labels being [E_m]_L and [E_m]_R
# 1: solid line, 2: dashed line
# D: thick, E: thin

plt.figure(figsize=(5, 5), facecolor='white')
plt.plot(time_points, C1_vals, label='[D_m]_R', linestyle='dashed', linewidth=2, color='black')
plt.plot(time_points, C2_vals, label='[D_m]_L', linestyle='solid', linewidth=2, color='black')
plt.plot(time_points, F1_vals, label='[E_m]_R', linestyle='dashed', linewidth=1, color='black')
plt.plot(time_points, F2_vals, label='[E_m]_L', linestyle='solid', linewidth=1, color='black')
plt.xlabel('Time')
plt.ylabel('Concentration')
# legend outside
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(0, 200)
# plt.savefig('../plots/minD_minE_sim.pdf', bbox_inches='tight')
plt.savefig('../plots/minD_minE_sim.png', bbox_inches='tight')
# plt.savefig('../plots/minD_minE_sim.eps', bbox_inches='tight')

#%%