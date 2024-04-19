import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
#whichgal = 'retic'
whichgal = 'retevol'

#Set projection axis (0=x; 1=y; 2=z) and distance offset
#from "true" value for the mock:
axis = 2
if (whichgal == 'retic'):
    data_file = '../Data/Ret_mock/ret2_ics_jr_v1.h5'
if (whichgal == 'retevol'):
    data_file = '../Data/Ret_mock/ret2_jr_v1.h5'
outfile = output_base+whichgal+'/%d/' % (axis)+whichgal
print('Projecting along axis:', axis)
Nbin = 100.0
Nbinkin = 100.0
Nbinkin_prop = Nbinkin
verr = 2.0

#Plot ranges:
xpltmin = 1e-2
xpltmax = 2.0
surfpltmin = 1e-3
surfpltmax = 1e5
vztwopltmin = 0
vztwopltmax = 30
vzfourpltmin = 1e5
vzfourpltmax = 1e7

#Priors for surface density fit. Array values are:
#Double alp-bet-gam SB model:
p0in_min = np.array([np.log10(1.0e-4),np.log10(1e-6),\
                     np.log10(1e-2),np.log10(1.0e-5),\
                     0.25,0.25,\
                     3.1,3.1,\
                     0.0,0.0])
p0in_max = np.array([np.log10(1.0e7),np.log10(1.0e-5),\
                     np.log10(3.0),np.log10(1e-1),\
                     3,3,\
                     7,7,\
                     1.99,1.99])
Rfitmin = -1
Rfitmax = -1

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
Rfitvmax = -1

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
R, surfden, surfdenerr, Rhalf, Rkin, vz, vzerr, mskin, \
    x, y, vx, vxerr, vy, vyerr, msprop = \
    retmock_api(data_file,verr,Nbin,axis)
use_dataRhalf = 'no'

#Set propermotions:
propermotion = 'yes'

#Downsample number of stars to match real data.
nsamples = 1000
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
