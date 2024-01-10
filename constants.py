import numpy as np

#Fundamental:
G = 6.67e-11

#Units:
Msun = 1.989e30
kpc = 3.086e19
Mpc = kpc*1000.0
pc = kpc/1000.0
kms = 1000.

#Conversions:
deg = 360./2./np.pi
arcsec = 360./2./np.pi * 60. * 60.
arcmin = arcsec / 60.
Msunkpc3toGeVcm3 = 3.7973132915271756e-08
kpccm = 1e3*3.0856775*1e18
year = 365.0*24.0*60.0*60.0
Gyr = year*1.0e9

#Cosmology [Msun/kpc^-3]:
OmegaM_h2 = 0.12
rhocrit = 135.05
oden = 200
h = 0.7
OmegaM = OmegaM_h2 / h**2.0
OmegaL = 1.0-OmegaM

#Specific to gravsphere:
Guse = G * Msun / kpc

#Output directory:
output_base = '/vol/ph/astro_temp/jread/gravsphere/Output/'
