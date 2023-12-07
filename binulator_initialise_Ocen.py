import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
whichgal = 'Ocen'
whichcen = 'RenAnd'
#whichcen = 'RenDisp'
#whichcen = 'RenRot'
#whichcen = 'RenNoy08'
#whichcen = 'RenNoy10'
infile_phot = '../Data/Ocen/Ocen_SB_%s.txt' % (whichcen)
infile_kin = '../Data/Ocen/Ocen_RV_rawprops_%s.txt' % (whichcen)
infile_prop = '../Data/Ocen/Ocen_HST_Gaia_PMs_rawprops_%s.txt' % (whichcen)
dgal_kpc = 5.0
outfile = output_base+whichgal+'/dgal_%.1f/%s/' % (dgal_kpc,whichcen)+whichgal
print('Using Omega cen distance: %.1f kpc' % (dgal_kpc))

#Plot ranges:
xpltmin = 1e-6
xpltmax = 5.0
surfpltmin = 1e-3
surfpltmax = 1e5
vztwopltmin = 0
vztwopltmax = 30
vzfourpltmin = 1e5
vzfourpltmax = 1e7

#Number of stars per bin [-1 indicates that
#binning was already done elsewhere]:
Nbin = -1

#Use variable number of stars per bin:
Nbinkin = 250.0
Nbinkin_prop = Nbinkin
Nbinkinarr = np.array([25.0,25.0,25.,\
                       50.0,50.0,50.0,\
                       100.0,100.0,100.0,\
                       250.0,250.0,250.0,\
                       500.0])
Nbinkinarr_prop = np.array([100.0,100.0,100.0,\
                            250.0,250.0,250.0,\
                            500.0])

#Priors for surface density fit. Array values are:
#[M1,M2,M3,a1,a2,a3] where M,a are the Plummer mass
#and scale length. [-1 means use full radial range].
p0in_min = np.array([-10,-10,-10,0.001,0.001,0.001])
p0in_max = np.array([10,10,10,0.05,0.05,0.05])
Rfitmin = -1
Rfitmax = 40.0 / 1000.0

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
Rfitvmax = -1

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
R, surfden, surfdenerr, Rhalf, \
    Rkin, vz, vzerr, mskin, \
    x, y, vx, vxerr, vy, vyerr, msprop = \
        ocen_api(dgal_kpc,infile_phot,infile_kin,infile_prop)
use_dataRhalf = 'no'
propermotion = 'yes'
