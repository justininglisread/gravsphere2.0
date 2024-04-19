import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
whichgal = 'Ocen_10e3_1e-3'

#Set projection axis (0=x; 1=y; 2=z) and distance offset
#from "true" value for the mock:
axis = 2
dgal_kpc_true = 5.0
dgal_kpc_offset = 5.0
if (whichgal == 'Ocen_10e3_1e-3'):
    data_file = '/vol/ph/astro_data/jread/Ocen_mock/mock_omegacen_MBH_10e3_rBH_1e-3pc.dat'
outfile = output_base+whichgal+'/dgal_%.1f_%d/' % (dgal_kpc_offset,axis)+whichgal
print('Setting mock Omega cen distance to: %.1f kpc' % (dgal_kpc_offset))
print('True Omega cen distance: %.1f kpc' % (dgal_kpc_true))
print('Projecting along axis:', axis)
Nbin = 100.0
Nbinkin = 250.0
Nbinkin_prop = Nbinkin
verr = 4.0

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
#Double alp-bet-gam SB model:
p0in_min = np.array([np.log10(1.0e-4),np.log10(1e2),\
                     np.log10(1e-2),np.log10(1.0e-5),\
                     0.25,0.25,\
                     3.1,3.1,\
                     0.0,0.0])
p0in_max = np.array([np.log10(1.0e5),np.log10(1.0e7),\
                     np.log10(3.0),np.log10(1e-1),\
                     9,9,\
                     12,12,\
                     0.5,0.5])
Rfitmin = -1
Rfitmax = 1.0

#Priors for binulator velocity dispersion calculation.
#Array values are: [vzmean,alp,bet,backamp1,backmean1,backsig1,
#backamp2,backmean2,backsig2], where alp is ~the dispersion,
#bet=[1,5] is a shape parameter, and "back" is a Gaussian of
#amplitude "backamp", describing some background. The first
#Gaussian is assumed to have mean >0; the second mean <0, hence
#the prior range on the mean is +ve definite for both.
p0vin_min = np.array([-1.0e-3,1.0,1.0,\
                      1e-5,20.0,30.0,\
                      1e-5,20.0,30.0])
p0vin_max = np.array([1.0e-3,30.0,5.0,\
                      1e-4,95.0,125.0,\
                      1e-4,95.0,125.0])
vfitmin = 0
vfitmax = 0
Rfitvmin = -1
Rfitvmax = 100.0/1000.0

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

#Downsample number of stars to match real Ocen data.
nsamples = 15000
sample_choose = np.random.randint(len(Rkin),size=nsamples)
Rkin = Rkin[sample_choose]
vz = vz[sample_choose]
vzerr = vzerr[sample_choose]
mskin = mskin[sample_choose]
if (propermotion == 'yes'):
    x = x[sample_choose]
    y = y[sample_choose]
    vx = vx[sample_choose]
    vxerr = vxerr[sample_choose]
    vy = vy[sample_choose]
    vyerr = vyerr[sample_choose]
    msprop = msprop[sample_choose]

print('Downsampling ...')
print('New total effective no. of tracers (kinematic):', np.sum(mskin))
print('New total effective no. of tracers (proper motions):', np.sum(msprop))
