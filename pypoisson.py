from __future__ import division
import os
import math
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import optimize
from scipy.sparse import linalg


## Constants ##
q = 1 # e
e_0 = 10.91 # e^2/chbar # vacuum permittivity
k_B = 8.6173324E-05 # eV/K
meter_eV_factor = 1.9733E-07 # m*eV/chbar
time_eV_factor = 6.5821E-16 # s*eV/hbar


## Material Constants ##
T = 300 # K
e_r = 12.9 # GaAs relative permittivity
N_a = 1.0E+22 # 1/m^3 # doping in p-type
N_d = 1.0E+22 # 1/m^3 # doping in n-type
n_i = 2.498E+12 # 1/m^3
mu_n = 0.85 # m^2/V-s
mu_p = 0.04 # m^2/V-s

p_type_width = 2E-06 # m
n_type_width = 2E-06 # m


## Mode Settings ## 
grid_pnts = 200 # must be an even number for now


## Construct constant FDM matrix ##
diag = np.zeros(grid_pnts-2) - 2
udiag = np.zeros(grid_pnts-3) + 1
ldiag = np.zeros(grid_pnts-3) + 1
fdm_mat = sparse.diags([diag, udiag, ldiag], [0, 1, -1], shape=(grid_pnts-2, grid_pnts-2), format="csc")


## Build material functions ##
def one_sided_Vbi(doping): # pass negative doping for p-type material
  return (k_B*T/q)*math.asinh(doping/(2*n_i))

def meter_conc_to_eV_conc(concentration):
  return concentration * meter_eV_factor**3

def meter_to_eV(distance):
  return distance / meter_eV_factor

def mobility_to_eV(mobility):
  return mobility * time_eV_factor / (meter_eV_factor**2)

def diffusion_coeff(mobility_eV):
  return k_B * T * mobility_eV / q

def pnts_in_type(width,grid_distance):
  return int(width / grid_distance) + 1


## Material Calculations ##
V_bi_p = one_sided_Vbi(-N_a)
V_bi_n = one_sided_Vbi(N_d)

N_a_eV = meter_conc_to_eV_conc(N_a) # using units in relation to eV to keep values close to 1
N_d_eV = meter_conc_to_eV_conc(N_d) # could use any arbitrary units but this is good
n_i_eV = meter_conc_to_eV_conc(n_i)

mu_p_eV = mobility_to_eV(mu_p)
mu_n_eV = mobility_to_eV(mu_n)

D_p = diffusion_coeff(mu_p_eV)
D_n = diffusion_coeff(mu_n_eV)

cell_width_eV = meter_to_eV(n_type_width + p_type_width)

del_x = cell_width_eV/(grid_pnts - 1) # want 1 fewer 'gaps' than points
del_x_2 = del_x ** 2

pnts_in_p = pnts_in_type(meter_to_eV(p_type_width),del_x)
pnts_in_n = pnts_in_type(meter_to_eV(n_type_width),del_x)


## Functions that use material calculations ##
def carrier_conc_from_phi(Phi): # pass negative phi for p-type material
  return n_i_eV * math.exp(q*Phi/(k_B*T))
  

def Bernoulli(x):
  idxs = (np.abs(x) < 1e-12)
  idxs2 = ( np.abs(x) >= 1e-12)
  B = np.zeros(x.shape)
  B[idxs] = 1.0
  B[idxs2] = x[idxs2] / np.expm1( x[idxs2] )
  return B


def carrier_conc_from_continuity(Phi)
  for i in xrange(Phi.size):
    diag2[i] = 2
  for i in xrange(Phi.size-1):
    udiag2[i] = D_n * Bernoulli((Phi[i+1] - Phi[i])/(k_B*T)) / del_x_2
    ldiag2[i] = D_n * Bernoulli((Phi[i+1] - Phi[i])/(k_B*T)) / del_x_2
  fdm_mat = sparse.diags([diag2, udiag2, ldiag2], [0, 1, -1], shape=(Phi.size, Phi.size), format="csc")


def big_func(Phi,n,p): # The big linear algebra setup function 

  rhs = np.zeros(phi.size) # only interior grid points needed here

  # Take note of junction polarity and assign charge densities for each grid point
  for i in xrange(rhs.size):
    if i < (pnts_in_p):
      rho = -q * (p - n - N_a_eV) # charge density in p-type
    else:
      rho = -q * (p - n + N_d_eV) # charge density in n-type
    rhs[i] = del_x_2 * rho / (e_r*e_0) # setup rhs for p-type

  # Incorporate boundary conditions
  rhs[0] = rhs[0] - V_bi_p # add potential at p-type depletion-neutral region iface 
  rhs[rhs.size-1] = rhs[rhs.size-1] - V_bi_n # add potential at n-type depletion-neutral region iface 

  
  return fdm_mat * Phi - rhs

## End Functions ##
 
 
#### MAIN ####

# Build guess vectors
phi_guess = np.zeros(grid_pnts - 2)
for i in xrange(phi_guess.size):
  if i < (pnts_in_p-1):
    phi_guess[i] = V_bi_p # charge density in p-type
  else:
    phi_guess[i] = V_bi_n # charge density in n-type

potentials = np.zeros(grid_pnts) # Create an array of potentials, one for each grid point


potentials[0] = V_bi_p # insert known boundary potentials
potentials[potentials.size-1] = V_bi_n

for i in xrange(2):
  potentials[1:potentials.size-1:1] = optimize.newton_krylov(big_func,phi_guess,verbose=1,iter=10) # Trick because numpy can't append arrays without copying them




# Setup plot #
distances = np.linspace(0,cell_width_eV*1E6*meter_eV_factor,num=grid_pnts)

plt.plot(distances, potentials, '-', c = 'b')
plt.xlabel(r'Distance ($\mu$m)')
plt.ylabel('Potential (V)')
plt.show()
