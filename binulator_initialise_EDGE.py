import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
nstars = 100
whichgal = 'halo383early'
#whichgal = 'halo600'
if (whichgal == 'halo383early'):
    data_file = \
        './Data/EDGE/halo383early_stars.dat'
    Nbin = 20
    Nbinkin = 20
elif (whichgal == 'halo600'):
    data_file = \
        './Data/EDGE/halo600_stars.dat'
    Nbin = 20
    Nbinkin = 20
outfile = output_base+'EDGE/'+whichgal+'/'+whichgal
verr = 7.0

#Plot ranges:
xpltmin = 1e-2
xpltmax = 10.0
surfpltmin = 1e-6
surfpltmax = 100
vztwopltmin = 0
vztwopltmax = 25
vzfourpltmin = 1e3
vzfourpltmax = 1e7

#Priors for surface density fit. Array values are:
#[M1,M2,M3,a1,a2,a3] where M,a are the Plummer mass
#and scale length. [-1 means use full radial range].
p0in_min = np.array([1e-4,1e-4,1e-4,0.01,0.01,0.01])
p0in_max = np.array([1e2,1e2,1e2,1.0,1.0,1.0])
Rfitmin = -1
Rfitmax = 1.0

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
Rfitvmin = -1
Rfitvmax = 1.0

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
axis = 2
R, surfden, surfdenerr, Rhalf, Rkin, vz, vzerr, mskin, \
    x, y, vx, vxerr, vy, vyerr, msprop = \
    ocenmock_api(data_file,verr,Nbin,axis)
use_dataRhalf = 'no'

#Propermotions:
propermotion = 'no'

#Subsample the velocity data:
sample_choose = np.random.randint(len(Rkin),size=nstars)
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
