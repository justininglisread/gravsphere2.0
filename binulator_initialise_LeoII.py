import numpy as np
from binulator_apis import *
from constants import * 

#Data files and output base filename:
whichgal = 'LeoII'
infile_kin = './Data/Walker_dwarfs/leo2_justin1_spec.dat'
infile_phot = './Data/Walker_dwarfs/leo2_justin1_phot.dat'
outfile = output_base+whichgal+'/'+whichgal

#Plot ranges:
xpltmin = 1e-2
xpltmax = 10.0
surfpltmin = 1e-6
surfpltmax = 100
vztwopltmin = 0
vztwopltmax = 15
vzfourpltmin = 1e2
vzfourpltmax = 1e6

#Number of stars per bin [-1 indicates that
#binning was already done elsewhere]:
Nbin = 30
Nbinkin = 30

#Priors for surface density fit. Array values are:
#[M1,M2,M3,a1,a2,a3] where M,a are the Plummer mass
#and scale length. [-1 means use full radial range].
p0in_min = np.array([-1e1,-1e1,-1e1,0.01,0.01,0.01])
p0in_max = np.array([1e2,1e2,1e2,1.0,1.0,2.0])
Rfitmin = -1
Rfitmax = -1

#Priors for binulator velocity dispersion calculation.
#Array values are: [vzmean,alp,bet,backamp1,backmean1,backsig1,
#backamp2,backmean2,backsig2], where alp is ~the dispersion,
#bet=[1,5] is a shape parameter, and "back" is a Gaussian of
#amplitude "backamp", describing some background. The first
#Gaussian is assumed to have mean >0; the second mean <0, hence
#the prior range on the mean is +ve definite for both.
p0vin_min = np.array([-20.0,5.0,1.0,\
                      1e-5,20.0,30.0,\
                      1e-5,20.0,30.0])
p0vin_max = np.array([20.0,25.0,5.0,\
                      1e-4,95.0,125.0,\
                      1e-4,95.0,125.0])
vfitmin = 0
vfitmax = 0
Rfitvmin = -1
Rfitvmax = -1

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
dgal_kpc = 233.0
R, surfden, surfdenerr, Rhalf, \
    Rkin, vz, vzerr, mskin, vsys = \
    walker_api(infile_phot,infile_kin,dgal_kpc,Nbin)
use_dataRhalf = 'no'

#Propermotions:
propermotion = 'no'
