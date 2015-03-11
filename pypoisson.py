from __future__ import division
import os
import math
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import optimize
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
grid_pnts = 200 # must be even number for now
cell_width = 4E-06 / meter_eV_factor


## Material Calculations ##
V_bi_p = (k_B*T/q)*math.asinh(-N_a/(2*n_i))
V_bi_n = (k_B*T/q)*math.asinh(N_d/(2*n_i))

N_d_eV = N_d * meter_eV_factor**3 # using units in relation to eV to keep values close to 1
N_a_eV = N_a * meter_eV_factor**3 # could use any arbitrary units but this is good
n_i_eV = n_i * meter_eV_factor**3

del_x = cell_width/(grid_pnts - 1) # want 1 fewer 'gaps' than points
del_x_2 = del_x ** 2


## Construct constant solver matrix ##
diag = np.zeros(grid_pnts-2) - 2
udiag = np.zeros(grid_pnts-3) + 1
ldiag = np.zeros(grid_pnts-3) + 1
fdm_mat = sparse.diags([diag, udiag, ldiag], [0, 1, -1], shape=(grid_pnts-2, grid_pnts-2), format="csc")


##Functions##
def big_func(Phi):
  
  rhs = np.zeros(grid_pnts-2) # only interior grid points needed here

  # Take note of junction polarity and assign charge densities for each grid point
  halfway = int(rhs.size/2)
  for i in xrange(halfway):
    n_eV = n_i_eV * math.exp(-q*Phi[i]/(k_B*T))
    p_eV = n_i_eV * math.exp(q*Phi[i]/(k_B*T))
    rho_n = q * (p_eV - n_eV + N_d_eV) # charge density in n-type
    rhs[i] = del_x_2 * rho_n / (e_r*e_0) # setup rhs for n-type
  for i in xrange(halfway):
    n_eV = n_i_eV * math.exp(-q*Phi[i+halfway]/(k_B*T))
    p_eV = n_i_eV * math.exp(q*Phi[i+halfway]/(k_B*T))
    rho_p = q * (p_eV - n_eV - N_a_eV) # charge density in p-type
    rhs[i+halfway] = del_x_2 * rho_p / (e_r*e_0) # setup rhs for p-type

  # Incorporate boundary conditions
  rhs[0] = rhs[0] - V_bi_p # add potential at p-type depletion-neutral region iface 
  rhs[rhs.size-1] = rhs[rhs.size-1] - V_bi_n # add potential at n-type depletion-neutral region iface 

  
  return fdm_mat * Phi - rhs


 
#### MAIN ####

# Build guess vector
phi_guess = np.zeros(grid_pnts - 2)
halfway = int(phi_guess.size/2)
for i in xrange(halfway):
  phi_guess[i] = V_bi_p
for i in xrange(halfway):
  phi_guess[i+halfway] = V_bi_n

potentials = np.zeros(grid_pnts) # Create an array of potentials, one for each grid point

potentials[1:potentials.size-1:1] = optimize.newton_krylov(big_func,phi_guess,verbose=1) # Trick because numpy can't append arrays without copying them

potentials[0] = V_bi_p # insert known boundary potentials
potentials[potentials.size-1] = V_bi_n


# Setup plot #
distances = np.linspace(0,cell_width*1E6*meter_eV_factor,num=grid_pnts)

plt.plot(distances, potentials, '-', c = 'b')
plt.xlabel(r'Distance ($\mu$m)')
plt.ylabel('Potential (V)')
plt.show()
