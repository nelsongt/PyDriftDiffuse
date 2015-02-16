from __future__ import division
import os
import math
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg


## Material Constants ##
T = 300 # K
e_r = 12.9 # GaAs relative permittivity
N_d = 1.0E+22 # 1/m^3 # doping in n-type
N_a = 1.0E+22 # 1/m^3 # doping in p-type
n_i = 2.498E+12 # 1/m^3

## Constants ##
q = 1 # e
e_0 = 10.91 # e^2/chbar # vacuum permittivity
k_B = 8.6173324E-05 # eV/K
meter_eV_factor = 1.9733E-07 # m*eV/chbar

## Mode Settings ## 
grid_pnts = 100 # must be even number for now


 
#### MAIN ####
V_bi = (k_B*T/q)*math.log(N_d*N_a/n_i**2) # Find built-in Voltage

N_d_eV = N_d * meter_eV_factor**3 # using units in relation to eV to keep values close to 1
N_a_eV = N_a * meter_eV_factor**3 # could use any arbitrary units but this is good

dep_width_n = (1/N_d_eV) * math.sqrt(2*e_r*e_0*V_bi/(q*((1/N_a_eV) + (1/N_d_eV))))
dep_width_p = (1/N_a_eV) * math.sqrt(2*e_r*e_0*V_bi/(q*((1/N_a_eV) + (1/N_d_eV))))

dep_width = dep_width_n + dep_width_p # this program is solving for inside the SCR

del_x = dep_width/(grid_pnts - 1) # want 1 fewer 'gaps' than points
del_x_2 = del_x ** 2


rho_n = -q * N_d_eV # charge density in n-type, depletion approx.
rho_p = q * N_a_eV # charge density in p-type, depletion approx.

rhs_n = del_x_2 * rho_n / (e_r*e_0) # setup rhs for n-type
rhs_p = del_x_2 * rho_p / (e_r*e_0) # setup rhs for p-type

potentials = np.zeros(grid_pnts) # Create an array of potentials and RHS, one for each grid point
rhs = np.zeros(grid_pnts-2) # only interior grid points needed here

# Take note of junction polarity and assign charge densities for each grid point
halfway = int(rhs.size/2)
for i in xrange(halfway):
  rhs[i] = rhs_p
for i in xrange(halfway):
  rhs[i+halfway] = rhs_n

# Incorporate boundary conditions
rhs[0] = rhs[0] - 0 # add potential at p-type depletion-neutral region iface 
rhs[rhs.size-1] = rhs[rhs.size-1] - V_bi # add potential at n-type depletion-neutral region iface 


# Construct solver matrix #
diag = np.zeros(grid_pnts-2) - 2
udiag = np.zeros(grid_pnts-3) + 1
ldiag = np.zeros(grid_pnts-3) + 1
fdm_mat = sparse.diags([diag, udiag, ldiag], [0, 1, -1], shape=(grid_pnts-2, grid_pnts-2), format="csc")


# Solve set of eqns #
potentials[1:potentials.size-1:1] = sparse.linalg.spsolve(fdm_mat,rhs) # Trick because numpy can't append arrays without copying them... but does this copy the array?

potentials[0] = 0 # insert known boundary potentials
potentials[potentials.size-1] = V_bi


# Setup plot #
distances = np.linspace(0,dep_width*1E6*meter_eV_factor,num=grid_pnts)

plt.plot(distances, potentials, '-', c = 'b')
plt.xlabel(r'Distance ($\mu$m)')
plt.ylabel('Potential (V)')
plt.show()
