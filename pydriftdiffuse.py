from __future__ import division
import os
import math
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import optimize
from scipy import integrate
from scipy.sparse import linalg


## Constants ##
T = 300 # K # Temp of cell
T_s = 5760 # K # Temp of black-body sun
q = 1 # e
e_0 = 10.91 # e^2/chbar # vacuum permittivity
k_B = 8.6173324E-05 # eV/K
h_eV = 2*math.pi # hbar
c_eV = 1 # c
subtend_angle_degrees = 0.26 # degrees
meter_eV_factor = 1.9733E-07 # m*eV/chbar
time_eV_factor = 6.5821E-16 # s*eV/hbar


## Material Constants ##
e_r = 12.9 # GaAs relative permittivity
E_g = 1.424 # eV # Bandgap
chi = 4.07 # eV # GaAs electron affinity

N_c = 4.7E+23 # 1/m^3 # GaAs effective conduction band density of states
N_v = 9.0E+24 # 1/m^3 # GaAs effective valence band density of states
N_a = 1.0E+22 # 1/m^3 # doping in p-type
N_d = 1.0E+22 # 1/m^3 # doping in n-type
n_i = 2.498E+12 # 1/m^3

mu_n = 0.85 # m^2/V-s
mu_p = 0.85#0.04 # m^2/V-s

p_type_width = 2E-06 # m
n_type_width = 2E-06 # m

G = 0
R = 0

V_applied = 0.938 # V # Applied bias


## Mode Settings ## 
grid_pnts = 400 # must be an even number for now
suns_factor = 1 # suns


## Construct constant FDM matrix ##
diag = np.zeros(grid_pnts-2) - 2
udiag = np.zeros(grid_pnts-3) + 1
ldiag = np.zeros(grid_pnts-3) + 1
fdm_mat = sparse.diags([diag, udiag, ldiag], [0, 1, -1], shape=(grid_pnts-2, grid_pnts-2), format="csc")


## Build material functions ##
def type_modifier(type): # 1 for p-type, 0 for n-type... I should make a typedef'ed enumerated list here
  if type: # is true
    return -1
  else:
    return 1

def one_sided_Vbi(doping,type):
  mod = type_modifier(type)
  return (k_B*T/q)*math.asinh(mod*doping/(2*n_i))

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

def bose_einstein(E,T,qV):
  return 1 / (math.exp((E-qV)/(k_B*T)) - 1)

def planck_rad_func(E,T,qV):
  return E**3 * bose_einstein(E,T,qV)


## Material Calculations ##
V_bi_p = one_sided_Vbi(N_a,1)
V_bi_n = one_sided_Vbi(N_d,0)

N_a_eV = meter_conc_to_eV_conc(N_a) # using units in relation to eV to keep values close to 1
N_d_eV = meter_conc_to_eV_conc(N_d) # could use some arbitrary units but this is good
n_i_eV = meter_conc_to_eV_conc(n_i)


mu_p_eV = mobility_to_eV(mu_p) # 
mu_n_eV = mobility_to_eV(mu_n)

D_p = diffusion_coeff(mu_p_eV)
D_n = diffusion_coeff(mu_n_eV)

cell_width_eV = meter_to_eV(n_type_width + p_type_width)

del_x = cell_width_eV/(grid_pnts - 1) # want 1 fewer 'gaps' than points
del_x_2 = del_x ** 2

pnts_in_p = pnts_in_type(meter_to_eV(p_type_width),del_x)
pnts_in_n = pnts_in_type(meter_to_eV(n_type_width),del_x)

n_conc = np.zeros(grid_pnts)
p_conc = np.zeros(grid_pnts)

E_i = 0.5 * (E_g + k_B*T*math.log(N_c/N_v))

flux_const = 2 / (h_eV**3 * c_eV**2)

sun_geo_factor = math.pi * math.sin(math.radians(subtend_angle_degrees)) ** 2

## Integrate to find the power from the sun ##
I_0, I_err = integrate.quad(planck_rad_func,E_g,10,args=(T_s,0)) # Integrate the planck radiation function from 0 eV to 10 eV (~inf)
I_0 = I_0 * flux_const * sun_geo_factor * suns_factor

G = I_0 # Units should be W/m^2 equivalent


## Functions that use material calculations ##
def carrier_conc_from_phi(Phi): # pass negative phi for p-type material
  return n_i_eV * math.exp(q*Phi/(k_B*T))

def carrier_conc_from_phi2(Phi,n_or_p,type): # pass negative phi for p-type material
  mod = type_modifier(type)
  #phi_p_n = E_i + chi + Phi - mod*(k_B*T/q)*math.log(n_or_p/n_i_eV)
  phi_p_n = Phi - mod*(k_B*T/q)*math.log(n_or_p/n_i_eV)
  #print phi_p_n
  #print E_i
  #print Phi
  #print mod*(k_B*T/q)*math.log(n_or_p/n_i_eV)
  return n_i_eV * math.exp(q*mod*(Phi - phi_p_n)/(k_B*T))
  

def ohmic_carrier_conc(type):  
  mod = type_modifier(type)
  if mod == -1:
    doping = N_a_eV
  elif mod == 1:
    doping = N_d_eV
  majority = np.sqrt(doping**2 + 4 * n_i_eV**2)/2 + (doping)/2 
  minority = n_i_eV**2 / majority
  return (majority, minority)


def Bernoulli(x):
  if np.abs(x) < 1e-12:
    return 1.0
  else:
    return x / np.expm1(x)


def conc_factor(Diff,Phi1,Phi2,type):
  mod = type_modifier(type)
  return Diff*Bernoulli(mod*(Phi1 - Phi2)/(k_B*T))


def carrier_conc_from_continuity(Phi,Diff,type,polarity):  # Phi has total grid points, polarity is 1 for p-type at 0, 0 for n-type at 0 ... typedef?
  n_or_p = np.zeros(Phi.size)
  diag_construct = np.zeros(Phi.size-2)
  udiag_construct = np.zeros(Phi.size-3)
  ldiag_construct = np.zeros(Phi.size-3)

  # For now set Generation = 0 and Recombination = 0
  rhs = np.zeros(Phi.size-2)
  
  for i in xrange(1,Phi.size-1):
    diag_construct[i-1] = -(conc_factor(Diff[i+1],Phi[i],Phi[i+1],type) + conc_factor(Diff[i],Phi[i],Phi[i-1],type))
    #print i
    #print diag_construct[i-1]
    #print Phi[i]
    rhs[i-1] = (R - G)*del_x_2
  for i in xrange(1,Phi.size-2):
    udiag_construct[i-1] = conc_factor(Diff[i+1],Phi[i+1],Phi[i],type)
    ldiag_construct[i-1] = conc_factor(Diff[i+1],Phi[i],Phi[i+1],type)  # This diagonal actually starts on line 2, so values are plussed 1
    

  
  conc_mat = sparse.diags([diag_construct, udiag_construct, ldiag_construct], [0, 1, -1], shape=(Phi.size-2, Phi.size-2), format="csc")

  #print diag_construct
  #print udiag_construct
  #print ldiag_construct
  #print conc_mat
  
  # Incorporate boundary conditions
  if polarity: # is true (p-type at 0)
    (p0,n0) = ohmic_carrier_conc(1)
    (nF,pF) = ohmic_carrier_conc(0)
  else: # n-type at 0
    (n0,p0) = ohmic_carrier_conc(0)
    (pF,nF) = ohmic_carrier_conc(1)
  
  if type:
    n_or_p[0] = p0
    n_or_p[n_or_p.size-1] = pF
  else:
    n_or_p[0] = n0
    n_or_p[n_or_p.size-1] = nF
    
  rhs[0] = rhs[0] - conc_factor(Diff[1],Phi[1],Phi[0],type) * n_or_p[0] 
  rhs[rhs.size-1] = rhs[rhs.size-1] - conc_factor(Diff[n_or_p.size-1],Phi[n_or_p.size-2],Phi[n_or_p.size-1],type) * n_or_p[n_or_p.size-1]
  
  
  # Solve for carrier concentrations using linear solver
  n_or_p[1:n_or_p.size-1:1] = sparse.linalg.spsolve(conc_mat,rhs)
  return n_or_p


def big_func(Phi): # The big linear algebra setup function, phi has (grid points - 2)

  rhs = np.zeros(Phi.size) # only interior grid points needed here

  # Take note of junction polarity and assign charge densities for each grid point
  for i in xrange(rhs.size):
    p_eV = carrier_conc_from_phi(-Phi[i])
    n_eV = carrier_conc_from_phi(Phi[i])
    if i < (pnts_in_p-1):
      rho = -q * (p_eV - n_eV - N_a_eV) # charge density in p-type
    else:
      rho = -q * (p_eV - n_eV + N_d_eV) # charge density in n-type

    rhs[i] = del_x_2 * rho / (e_r*e_0) # setup rhs for p-type

  # Incorporate boundary conditions
  rhs[0] = rhs[0] - V_bi_p # add potential at p-type depletion-neutral region iface 
  rhs[rhs.size-1] = rhs[rhs.size-1] - V_bi_n # add potential at n-type depletion-neutral region iface 

  
  return fdm_mat * Phi - rhs

def big_func2(Phi): # The big linear algebra setup function, phi has (grid points - 2)

  rhs = np.zeros(Phi.size) # only interior grid points needed here
  
  newPot = np.zeros(Phi.size+2)
  newPot[1:newPot.size-1:1] = Phi
  newPot[0] = V_bi_p + V_applied
  newPot[newPot.size-1] = V_bi_n
  
  p_conc = carrier_conc_from_continuity(newPot,diffusivity,1,1)
  n_conc = carrier_conc_from_continuity(newPot,diffusivity,0,1)

  # Take note of junction polarity and assign charge densities for each grid point
  for i in xrange(rhs.size):
    #p_eV = carrier_conc_from_phi(-Phi[i])
    #n_eV = carrier_conc_from_phi(Phi[i])
    #print p_eV
    #print n_eV
    #p_eV = carrier_conc_from_phi2(Phi[i],p_conc[i+1],1)
    #n_eV = carrier_conc_from_phi2(Phi[i],n_conc[i+1],0)
    #print p_eV
    #print n_eV
    if i < (pnts_in_p-1):
      rho = -q * (p_conc[i+1] - n_conc[i+1] - N_a_eV) # charge density in p-type
    else:
      rho = -q * (p_conc[i+1] - n_conc[i+1] + N_d_eV) # charge density in n-type

    rhs[i] = del_x_2 * rho / (e_r*e_0) # setup rhs for p-type

  # Incorporate boundary conditions
  rhs[0] = rhs[0] - V_bi_p - V_applied# add potential at p-type depletion-neutral region iface
  rhs[rhs.size-1] = rhs[rhs.size-1] - V_bi_n # add potential at n-type depletion-neutral region iface 

  
  return fdm_mat * Phi - rhs

def calc_current(polarity):  # returns in units of mA/cm^2
  J = diffusivity[diffusivity.size-1] * (Bernoulli((potentials[potentials.size-2] - potentials[potentials.size-1])/(k_B*T)) * n_conc[p_conc.size-2] - Bernoulli((potentials[potentials.size-1] - potentials[potentials.size-2])/(k_B*T)) * n_conc[p_conc.size-1]) / del_x
  return J * sc.e / (time_eV_factor * meter_eV_factor * meter_eV_factor * 10)


## End Functions ##
 
 
 
#### MAIN ####

# Build guess vectors
phi_guess = np.zeros(grid_pnts - 2)
diffusivity = np.zeros(grid_pnts)
potentials = np.zeros(grid_pnts) # Create an array of potentials, one for each grid point

voltages =  np.linspace(0.0,1.0,21)
currents = list()
for V_applied in voltages:

  for i in xrange(phi_guess.size):
    if i < (pnts_in_p-1):
      phi_guess[i] = V_bi_p + V_applied # charge density in p-type
      diffusivity[i+1] = D_p
    else:
      phi_guess[i] = V_bi_n # charge density in n-type
      diffusivity[i+1] = D_n
  diffusivity[0] = D_p
  diffusivity[diffusivity.size-1] = D_n



  potentials[0] = V_bi_p + V_applied # insert known boundary potentials
  potentials[potentials.size-1] = V_bi_n

  potentials[1:potentials.size-1:1] = phi_guess

  potentials[1:potentials.size-1:1] = optimize.newton_krylov(big_func2,potentials[1:potentials.size-1:1],verbose=1,iter=10) # Trick because numpy can't append arrays without copying them

  p_conc = carrier_conc_from_continuity(potentials,diffusivity,1,1)
  n_conc = carrier_conc_from_continuity(potentials,diffusivity,0,1)

  currents.append(calc_current(1))
  print V_applied
  #print current

npcurrents = np.asarray(currents)
print currents
print npcurrents
print voltages
# Setup plot #
distances = np.linspace(0,cell_width_eV*1E6*meter_eV_factor,num=grid_pnts)

plt.plot(distances, potentials, '-', c = 'b')
plt.xlabel(r'Distance ($\mu$m)')
plt.ylabel('Potential (V)')
plt.show()

plt.plot(distances, n_conc, '-', c = 'r')
plt.plot(distances, p_conc, '-', c = 'b')
plt.xlabel(r'Distance ($\mu$m)')
plt.ylabel('Concentration (density)')
plt.yscale('log')
plt.show()

plt.plot(voltages, npcurrents, '-', c = 'r')
plt.xlabel('Voltage (V)')
plt.ylabel(r'Current Density (mA/cm$^2$)')
plt.show()