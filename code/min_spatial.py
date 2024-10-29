#%%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit, prange  
import pickle,os

# delete the previous results if any
import shutil
shutil.rmtree('../results', ignore_errors=True)

# create the results directory
os.makedirs('../results', exist_ok=True)

# parameters
L1 = 1.0
L2 = 5.0  # system size
T = 5*10**5  # simulation steps
save_interval = 2000  # 


dx = 0.2  # spatial resolution
dt = dx**2 / 100 # time steps
nx, ny = round(L1 / dx), round(L2 / dx)  # grid points
nxb,nyb = nx,ny

d=1
D_D=14/dx**2 * d 
D_E=14/dx**2 * d 
D_d=0.06/dx**2 * d 
D_de=0.3/dx**2 * d 
D_e=0.06/dx**2 * d 

c=1
omega_D=0.1 *c 
omega_dD = 8.8*10**(-3)/dx *c 
omega_E = 6.96*10**(-5)/dx *c 
omega_ed = 0.139/dx *c
omega_de_c=0.08 *c
omega_de_m=1.5 *c 
omega_e=0.5 *c

u2_val=6000*(dx**2) # this value (initial value of u2) seems to be the most important parameter (under fixing the above parameters)

c_max=5.4*10**3*dx 

print(f"dx={dx}, dt={dt}, nx={nx}, ny={ny}")
# initial conditions
u1 = np.zeros((ny, nx))
u2 = np.zeros((ny, nx))
u3 = 0 * np.ones(2 * (nx + ny)) # mol per box (not density)
u4 = 0 * np.ones(2 * (nx + ny))
u5 = 0 * np.ones(2 * (nx + ny))

corner_cells = np.array([nx, nx+ny-1, 2*nx + ny, 2*(nx + ny) -1], dtype=np.int32)

pre_corner_cells = np.array([nx-1, nx+ny-2, 2*nx + ny-1, 2*(nx + ny) -2], dtype=np.int32)

post_corner_cells = np.array([nx+1, nx+ny, 2*nx + ny+1, 0], dtype=np.int32)

all_other_cells = np.array([i for i in range(2 * (nx + ny)) if i not in corner_cells and i not in pre_corner_cells and i not in post_corner_cells], dtype=np.int32)

# save parameters as dictionary and pickle it in /results
params = {'L1': L1, 'L2': L2, 'dx': dx, 'dt': dt, 'nx': nx, 'ny': ny, 'T': T, 'save_interval': save_interval,'c':c,'corner_cells':corner_cells,'pre_corner_cells':pre_corner_cells,'post_corner_cells':post_corner_cells}
with open('../results/params.pkl', 'wb') as f:
    pickle.dump(params, f)

# for i in range(nx):
#     for j in range(ny):
#         u2[j,i] = u2_val*(dx**2/0.01)
#         # u1[j,i] = 5*(dx**2/0.01) * np.exp(-0.2*dx*(j))

u2 = u2_val* np.ones((ny, nx))

for i in range(2 * (nx + ny)):
    u3[i]=2000*(L1*L2/(3.8*0.8))*np.exp(-0.1*dx/0.1*(i-(nx//2+nx+ny))**2)*dx

u3[corner_cells] = 0

@jit(nopython=True, cache=True)
def apply_periodic_boundary_1d(u):
    u_new = np.zeros(2 * (nx + ny) + 2)
    u_new[1:-1] = u
    u_new[0] = u[-1]
    u_new[-1] = u[0]
    return u_new

@jit(nopython=True, cache=True)
def apply_fixed_boundary_2d(u):
    u_new = np.zeros((ny + 2, nx + 2))  # ny rows, nx columns
    u_new[1:ny + 1, 1:nx + 1] = u
    return u_new

@jit(nopython=True, cache=True)
def apply_boundary_conditions(u1, u2, u3, u4, u5, u1_new, u2_new):
    # Equation (17) for u1 (c_D) and u2 (c_E)
    for i in range(nx):
        u1_new[-1, i] -= dt*(u1[-1, i] * (omega_D + omega_dD * u3[i]) * (c_max - u3[i] - u4[i])/c_max - (omega_de_m+omega_de_c)*u4[i])
        u2_new[-1, i] -= dt*(omega_E * u2[-1, i] * u3[i] - omega_e*u5[i] -omega_de_c*u4[i])

    for j in range(1,ny-1): # exclude the corner
        u1_new[-j - 1, -1] -= dt*(u1[-j - 1, -1] * (omega_D + omega_dD * u3[nx + j]) * (c_max - u3[nx + j] - u4[nx + j])/c_max - (omega_de_m+omega_de_c)*u4[nx + j])
        u2_new[-j - 1, -1] -= dt*(omega_E * u2[-j - 1, -1] * u3[nx + j] - omega_e*u5[nx + j] -omega_de_c*u4[nx + j])

    for i in range(nx):
        u1_new[0, -i - 1] -= dt*(u1[0, -i - 1] * (omega_D + omega_dD * u3[nx + ny + i]) * (c_max - u3[nx + ny + i] - u4[nx + ny + i])/c_max - (omega_de_m+omega_de_c)*u4[nx + ny + i])
        u2_new[0, -i - 1] -= dt*(omega_E * u2[0, -i - 1] * u3[nx + ny + i] - omega_e*u5[nx + ny + i] -omega_de_c*u4[nx + ny + i])
    for j in range(1,ny-1): # exclude the corner
        u1_new[j, 0] -= dt*(u1[j, 0] * (omega_D + omega_dD * u3[2 * nx + ny + j]) * (c_max - u3[2 * nx + ny + j] - u4[2 * nx + ny + j])/c_max - (omega_de_m+omega_de_c)*u4[2 * nx + ny + j])
        u2_new[j, 0] -= dt*(omega_E * u2[j, 0] * u3[2 * nx + ny + j] - omega_e*u5[2 * nx + ny + j] -omega_de_c*u4[2 * nx + ny + j]) 


@jit(nopython=True, parallel=True, cache=True)
def diffuse_and_react(u1, u2, u3, u4, u5):
    u1_new = u1.copy()
    u2_new = u2.copy()
    u3_new = u3.copy()
    u4_new = u4.copy()
    u5_new = u5.copy()

    u1_bound = apply_fixed_boundary_2d(u1)
    u2_bound = apply_fixed_boundary_2d(u2)
    u3_bound = apply_periodic_boundary_1d(u3)
    u4_bound = apply_periodic_boundary_1d(u4)
    u5_bound = apply_periodic_boundary_1d(u5)

    # 2D grid diffusion
    for i in prange(1, ny + 1):  # 'prange' for parallel processing
        for j in prange(1, nx + 1):

            diff_u1 = D_D * dt  * (u1_bound[i, j - 1] + u1_bound[i, j + 1] + u1_bound[i - 1, j] + u1_bound[i + 1, j] - 4 * u1_bound[i, j])
            diff_u2 = D_E * dt  * (u2_bound[i, j - 1] + u2_bound[i, j + 1] + u2_bound[i - 1, j] + u2_bound[i + 1, j] - 4 * u2_bound[i, j])

            if i == ny or i == 1:
                diff_u1 += D_D * dt  * u1_bound[i, j]
                diff_u2 += D_E * dt  * u2_bound[i, j]

            if j == nx or j == 1:
                diff_u1 += D_D * dt * u1_bound[i, j]
                diff_u2 += D_E * dt * u2_bound[i, j]

            u1_new[i - 1, j - 1] = u1[i - 1, j - 1] + diff_u1
            u2_new[i - 1, j - 1] = u2[i - 1, j - 1] + diff_u2

    cell_type = np.zeros(2 * (nx + ny), dtype=np.int8)  # 0: other, 1: pre_corner, 2: post_corner (have to do it this way for numba)

    for idx in pre_corner_cells:
        cell_type[idx] = 1
    for idx in post_corner_cells:
        cell_type[idx] = 2
    for idx in corner_cells:
        cell_type[idx] = 3

    # 1D grid diffusion
    for i in prange(1, 2 * (nx + ny) + 1):
        diff_u3, diff_u4, diff_u5 = 0.0, 0.0, 0.0
        
        if cell_type[i-1] == 0:
            diff_u3 = D_d * dt * (u3_bound[i + 1] + u3_bound[i - 1] - 2 * u3_bound[i])
            diff_u4 = D_de * dt  * (u4_bound[i + 1] + u4_bound[i - 1] - 2 * u4_bound[i])
            diff_u5 = D_e * dt  * (u5_bound[i + 1] + u5_bound[i - 1] - 2 * u5_bound[i])
        
        elif cell_type[i-1] == 1:
            diff_u3 = D_d * dt  * (u3_bound[i + 2] + u3_bound[i - 1] - 2 * u3_bound[i])
            diff_u4 = D_de * dt  * (u4_bound[i + 2] + u4_bound[i - 1] - 2 * u4_bound[i])
            diff_u5 = D_e * dt * (u5_bound[i + 2] + u5_bound[i - 1] - 2 * u5_bound[i])

        elif cell_type[i-1] == 2:
            diff_u3 = D_d * dt  * (u3_bound[i + 1] + u3_bound[i - 2] - 2 * u3_bound[i])
            diff_u4 = D_de * dt  * (u4_bound[i + 1] + u4_bound[i - 2] - 2 * u4_bound[i])
            diff_u5 = D_e * dt  * (u5_bound[i + 1] + u5_bound[i - 2] - 2 * u5_bound[i])
        if i==1:
            diff_u3 = D_d * dt  * (u3_bound[i + 1] + u3_bound[-3] - 2 * u3_bound[i])
            diff_u4 = D_de * dt  * (u4_bound[i + 1] + u4_bound[-3] - 2 * u4_bound[i])
            diff_u5 = D_e * dt  * (u5_bound[i + 1] + u5_bound[-3] - 2 * u5_bound[i])

        u3_new[i - 1] += diff_u3
        u4_new[i - 1] += diff_u4
        u5_new[i - 1] += diff_u5

    # Reactions at boundaries
    for i in prange(nx):
        # do for all including corners since it will still be diconnected and irrelevant
        react_u3 = dt * (u1[-1, i] * (omega_D + omega_dD * u3[i]) * (c_max - u3[i] - u4[i])/c_max - omega_E * u2[-1, i] * u3[i] - omega_ed * u3[i] * u5[i])
        react_u4 = dt * (omega_E * u2[-1, i] * u3[i] + omega_ed * u5[i] * u3[i] - (omega_de_m + omega_de_c) * u4[i])
        react_u5 = dt * (omega_de_m * u4[i] - omega_ed * u5[i] * u3[i] - omega_e * u5[i] )

        u3_new[i] = u3_new[i] + react_u3
        u4_new[i] = u4_new[i] + react_u4
        u5_new[i] = u5_new[i] + react_u5

    for j in prange(1,ny-1):
        react_u3 = dt * (u1[-j - 1, -1] * (omega_D + omega_dD * u3[nx + j]) * (c_max - u3[nx + j] - u4[nx + j])/c_max - omega_E * u2[-j - 1, -1] * u3[nx + j] - omega_ed * u3[nx + j] * u5[nx + j])
        react_u4 = dt * (omega_E * u2[-j - 1, -1] * u3[nx + j] + omega_ed * u5[nx + j] * u3[nx + j] - (omega_de_m + omega_de_c) * u4[nx + j])
        react_u5 = dt * (omega_de_m * u4[nx + j] - omega_ed * u5[nx + j] * u3[nx + j] - omega_e * u5[nx + j] )

        u3_new[nx + j] = u3_new[nx + j] + react_u3
        u4_new[nx + j] = u4_new[nx + j] + react_u4
        u5_new[nx + j] = u5_new[nx + j] + react_u5

    for i in prange(nx):
        react_u3 = dt * (u1[0, -i - 1] * (omega_D + omega_dD * u3[nx + ny + i]) * (c_max - u3[nx + ny + i] - u4[nx + ny + i])/c_max - omega_E * u2[0, -i - 1] * u3[nx + ny + i] - omega_ed * u3[nx + ny + i] * u5[nx + ny + i])
        react_u4 = dt * (omega_E * u2[0, -i - 1] * u3[nx + ny + i] + omega_ed * u5[nx + ny + i] * u3[nx + ny + i] - (omega_de_m + omega_de_c)*u4[nx + ny + i])
        react_u5 = dt * (omega_de_m*u4[nx + ny + i] - omega_ed * u5[nx + ny + i] * u3[nx + ny + i] - omega_e*u5[nx + ny + i] )

        u3_new[nx + ny + i] = u3_new[nx + ny + i] + react_u3
        u4_new[nx + ny + i] = u4_new[nx + ny + i] + react_u4
        u5_new[nx + ny + i] = u5_new[nx + ny + i] + react_u5

    for j in prange(1,ny-1):
        react_u3 = dt * (u1[j, 0] * (omega_D + omega_dD * u3[2 * nx + ny + j]) * (c_max - u3[2 * nx + ny + j] - u4[2 * nx + ny + j])/c_max - omega_E * u2[j, 0] * u3[2 * nx + ny + j] - omega_ed * u3[2 * nx + ny + j] * u5[2 * nx + ny + j])
        react_u4 = dt * (omega_E * u2[j, 0] * u3[2 * nx + ny + j] + omega_ed * u5[2 * nx + ny + j] * u3[2 * nx + ny + j] - (omega_de_m + omega_de_c)*u4[2 * nx + ny + j])
        react_u5 = dt * (omega_de_m*u4[2 * nx + ny + j] - omega_ed * u5[2 * nx + ny + j] * u3[2 * nx + ny + j] - omega_e*u5[2 * nx + ny + j] )

        u3_new[2 * nx + ny + j] = u3_new[2 * nx + ny + j] + react_u3
        u4_new[2 * nx + ny + j] = u4_new[2 * nx + ny + j] + react_u4
        u5_new[2 * nx + ny + j] = u5_new[2 * nx + ny + j] + react_u5

    apply_boundary_conditions(u1, u2, u3, u4, u5, u1_new, u2_new)

    return u1_new, u2_new, u3_new, u4_new, u5_new


def save_simulation_state(timestep, u1, u2, u3, u4, u5):
    data = {'timestep': timestep, 'u1': u1, 'u2': u2, 'u3': u3, 'u4': u4, 'u5': u5}
    with open(f'../results/step_{timestep}.pkl', 'wb') as f:
        pickle.dump(data, f)


for t in range(0, T, save_interval):
    print(f"Simulation step {t}/{T}")
    for _ in range(save_interval):
        u1, u2, u3, u4, u5 = diffuse_and_react(u1, u2, u3, u4, u5)
    # if any of them are nan, break and report
    if np.isnan(u1).any() or np.isnan(u2).any() or np.isnan(u3).any() or np.isnan(u4).any() or np.isnan(u5).any():
        print(f"Simulation step {t}/{T} resulted in NaN values. Exiting...")
        break
    save_simulation_state(t, u1, u2, u3, u4, u5)


# %%
