import numpy as np
from binulator_apis import *
from constants import * 

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
whichgal = 'SMCmock'
if (whichgal == 'SMCmock_3kpc'):
    infile_kin = '../Data/SMC_mock/Cut_3kpc/Sims_Radial_Vel_raw_3kpc_EDR3_err.dat'
    infile_phot = '../Data/SMC_mock/Cut_3kpc/Sims_Star_counts_3kpc.dat'
    infile_prop = '../Data/SMC_mock/Cut_3kpc/Sims_Prop_Mot_raw_3kpc_EDR3_err.dat'
else:
    infile_kin = '../Data/SMC_mock/Sims_Radial_Vel_raw_EDR3_err.dat'
    infile_phot = '../Data/SMC_mock/Sims_Star_counts_full.dat'
    infile_prop = '../Data/SMC_mock/Sims_Prop_Mot_raw_EDR3_err.dat'
outfile = output_base+whichgal+'/'+whichgal

#Plot ranges:
xpltmin = 1e-2
xpltmax = 10.0
surfpltmin = 1e-6
surfpltmax = 100
vztwopltmin = 0
vztwopltmax = 25
vzfourpltmin = 1e2
vzfourpltmax = 1e6

#Number of stars per bin [-1 indicates that
#binning was already done elsewhere]:
Nbin = -1
Nbinkin = 50
Nbinkinarr = np.array([50,50,50,\
                       100,100,100,\
                       250])
Nbinkinarr_prop = Nbinkinarr

#Priors for surface density fit. Array values are:
#[M1,M2,M3,a1,a2,a3] where M,a are the Plummer mass
#and scale length. [-1 means use full radial range].
p0in_min = np.array([-1e2,-1e2,-1e2,0.01,0.01,0.01])
p0in_max = np.array([1e2,1e2,1e2,2.0,2.0,5.0])
Rfitmin = -1
Rfitmax = 2.0

#Priors for binulator velocity dispersion calculation. 
#Array values are: [vzmean,alp,bet,backamp1,backmean1,backsig1,
#backamp2,backmean2,backsig2], where alp is ~the dispersion,
#bet=[1,5] is a shape parameter, and "back" is a Gaussian of
#amplitude "backamp", describing some background. The first
#Gaussian is assumed to have mean >0; the second mean <0, hence
#the prior range on the mean is +ve definite for both.
p0vin_min = np.array([-20.0,2.0,1.0,\
                      1e-4,20.0,2.0,\
                      1e-4,20.0,2.0])
p0vin_max = np.array([20.0,25.0,5.0,\
                      0.35,75.0,95.0,\
                      0.35,75.0,95.0])
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
R, surfden, surfdenerr, Rhalf, Rkin, vz, vzerr, mskin = \
    smc_api(infile_phot,infile_kin)
use_dataRhalf = 'yes'

#Propermotions:
propermotion = 'yes'
dgal_kpc = 60.0
if (propermotion == 'yes'):
    Nbinkin_prop = Nbinkin
    x, y, vx, vxerr, vy, vyerr, msprop = \
        smc_prop_api(infile_prop,dgal_kpc)

#No need to downsample number of stars to match real SMC data.
#Less than real number of data points once we slice on R<Rfitvmax.
#nsamples = 6000
#sample_choose = np.random.randint(len(Rkin),size=nsamples)
#Rkin = Rkin[sample_choose]
#vz = vz[sample_choose]
#vzerr = vzerr[sample_choose]
#mskin = mskin[sample_choose]
#if (propermotion == 'yes'):
#    x = x[sample_choose]
#    y = y[sample_choose]
#    vx = vx[sample_choose]
#    vxerr = vxerr[sample_choose]
#    vy = vy[sample_choose]
#    vyerr = vyerr[sample_choose]
#    msprop = msprop[sample_choose]
