import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
whichgal = 'Ocenmock3'

#Set projection axis (0=x; 1=y; 2=z) and distance offset
#from "true" value for the mock:
axis = 2
dgal_kpc_true = 5.0
dgal_kpc_offset = 5.9
if (whichgal == 'Ocenmock'):
    data_file = \
        '../Data/Ocen_mock/h383_fid_input.dat'
elif (whichgal == 'Ocenmock2'):
    data_file = \
        '../Data/Ocen_mock/h383_288_input.dat'
elif (whichgal == 'Ocenmock3'):
    data_file = \
        '../Data/Ocen_mock/h383_late_input.dat'
outfile = output_base+whichgal+'/dgal_%.1f_%d/' % (dgal_kpc_offset,axis)+whichgal
print('Setting mock Omega cen distance to: %.1f kpc' % (dgal_kpc_offset))
print('True Omega cen distance: %.1f kpc' % (dgal_kpc_true))
print('Projecting along axis:', axis)
Nbin = 50.0
Nbinkin = 100.0
Nbinkin_prop = Nbinkin
verr = 2.0

#Plot ranges:
xpltmin = 1e-6
xpltmax = 5.0
surfpltmin = 1e-3
surfpltmax = 1e5
vztwopltmin = 0
vztwopltmax = 30
vzfourpltmin = 1e5
vzfourpltmax = 1e7

#Priors for surface density fit. Array values are:
#[M1,M2,M3,a1,a2,a3] where M,a are the Plummer mass
#and scale length. [-1 means use full radial range].
p0in_min = np.array([-1e3,-1e3,-1e3,1.0e-5,1.0e-5,1.0e-5])
if (dgal_kpc_offset == 5.5):
    p0in_max = np.array([1e3,1e3,1e3,1.0,1.0,2.0])
    Rfitmin = -1
    Rfitmax = -1
else:
    p0in_max = np.array([1e3,1e3,1e3,0.2,0.2,0.2])
    Rfitmin = -1
    Rfitmax = 40.0/1000.0

#Priors for binulator velocity dispersion calculation.
#Array values are: [vzmean,alp,bet,backamp1,backmean1,backsig1,
#backamp2,backmean2,backsig2], where alp is ~the dispersion,
#bet=[1,5] is a shape parameter, and "back" is a Gaussian of
#amplitude "backamp", describing some background. The first
#Gaussian is assumed to have mean >0; the second mean <0, hence
#the prior range on the mean is +ve definite for both.
p0vin_min = np.array([-20.0,1.0,1.0,\
                      1e-5,20.0,30.0,\
                      1e-5,20.0,30.0])
p0vin_max = np.array([20.0,30.0,5.0,\
                      1e-4,95.0,125.0,\
                      1e-4,95.0,125.0])
vfitmin = 0
vfitmax = 0
if (dgal_kpc_offset == 5.5):
    Rfitvmin = -1
    Rfitvmax = -1
else:
    Rfitvmin = -1
    Rfitvmax = 40.0/1000.0

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
R, surfden, surfdenerr, Rhalf, Rkin, vz, vzerr, mskin, \
    x, y, vx, vxerr, vy, vyerr, msprop = \
    ocenmock_api(data_file,verr,Nbin,axis)
use_dataRhalf = 'no'

#Shift the distance:
R = R * dgal_kpc_offset / dgal_kpc_true
Rkin = Rkin * dgal_kpc_offset / dgal_kpc_true
x = x * dgal_kpc_offset / dgal_kpc_true
y = y * dgal_kpc_offset / dgal_kpc_true
vx = vx * dgal_kpc_offset / dgal_kpc_true
vy = vy * dgal_kpc_offset / dgal_kpc_true
vxerr = vxerr * dgal_kpc_offset / dgal_kpc_true
vyerr = vyerr * dgal_kpc_offset / dgal_kpc_true

#Set propermotions to be on:
propermotion = 'yes'
