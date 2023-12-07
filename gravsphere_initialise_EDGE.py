import numpy as np
from constants import * 
from functions import * 

#This file contains all the code options and choices for 
#running a given model. Throughout, -1 means auto-calculate.

#Run on multiple processors:
nprocs = 10

#Data files and output base filename:
whichgal = 'halo383early'
#whichgal = 'halo600'
infile = output_base+'EDGE/'+whichgal+'/'+whichgal
outdirbase = output_base+'EDGE2/'+whichgal+'/'

#Plot ranges and sample points [-1 means auto-calculate]:
rplot_inner = 1e-3
rplot_outer = 10.0
rplot_pnts = 2500
y_sigLOSmax = 35
ymin_Sigstar = 1e-3
ymax_Sigstar = 1e5
yMlow = 1e2
yMhigh = 1e10
yrholow = 1e3
yrhohigh = 1e12
alp3sig = 0.0
sigmlow = 1e-3
sigmhigh = 5.0

#Code options:
propermotion = 'no'
virialshape = 'yes'

#Overplot true solution (for mock data). If 
#yes, then the true solutions should be passed
#in: ranal,betatrue(ranal),betatruestar(ranal),
#truemass(ranal),trueden(ranal),truedlnrhodlnr(ranal).
overtrue = 'yes'
if (whichgal == 'halo383early'):
    truedata = np.genfromtxt('./Data/EDGE/halo383early_profiles.dat',dtype='f8')
else:
    truedata = np.genfromtxt('./Data/EDGE/halo600_profiles.dat',dtype='f8')    
ranal = truedata[:,0]

#Cumulative dark matter mass:
truemass = truedata[:,3]

#To be properly calculated + implemented:
betatruestar = np.zeros(len(ranal))
betatrue = np.zeros(len(ranal))

#Dark matter density:
trueden = truedata[:,6]

#To be properly implemented:
truedlnrhodlnr = np.zeros(len(ranal))

#Radial grid range for Jeans calculation:
rmin = 1.0e-3
rmax = 10.0
intpnts = np.int(250)

#Galaxy properties. Assume here that the baryonic mass
#has the same radial profile as the tracer stars. If this
#is not the case, you should set Mstar_rad and Mstar_prof 
#here. The variables barrad_min, barrad_max and bar_pnts 
#set the radial range and sampling of the baryonic mass model.
if (whichgal == 'halo383early'):
    Mstar = 4.4e6
else:
    Mstar = 5.1e5
#Mstar_err = Mstar*0.95    
Mstar_err = Mstar*0.25
baryonmass_follows_tracer = 'yes'
barrad_min = 1.0e-3
barrad_max = 10.0
bar_pnts = 5000

###########################################################
#Priors

#For surface density fit tracertol = [0,1] sets the spread 
#around the best-fit value from the binulator.
tracertol = 0.1

#Cosmology priors on the coreNFWtides model. mWDM(keV) is
#the mass of a thermal relic; <0 means CDM; sig_c200 is 
#the scatter of c200 in log10 space. If the cosmo_cprior
#is set, then we include a Gaussian spread in M200-c200 in
#the likelihood. Without this, M200-c200 enters only if 
#used to set the priors, below.
cosmo_cprior = 'yes'
sig_c200 = 0.1
mWDM = -1
if (mWDM > 0):
    cosmo_cfunc = lambda M200,h : \
        cosmo_cfunc_WDM(M200,h,OmegaM,rhocrit,mWDM)

#Velocity anisotropy priors. If logbetr0min=logbetr0max,
#it will be set automatically based on the half light
#radius:
logbetr0min = 0.0
logbetr0max = 0.0
betnmin = 1.0
betnmax = 3.0
bet0min = -0.2
bet0max = 0.2
betinfmin = -0.2
betinfmax = 0.2

#CoreNFWtides priors:
#logM200low = 8.5
#logM200high = 10.5
logM200low = 9.05
logM200high = 9.95

clow = 5.0
chigh = 50.0
rclow = 1e-3
rchigh = 2.0
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)
nlow = 0.0
nhigh = 1.0
rtlow = 20.0
rthigh = 100.0
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)
dellow = 4.0
delhigh = 6.0

if (cosmo_cprior == 'yes'):
    clow = 1.0
    chigh = 100.0

#Priors on central dark mass [set logMcenlow/high very negative
#to switch this off. Mcen is the mass in Msun; acen is the
#scale length in kpc, usually assumed smaller than Rhalf
#to avoid degeneracies with the stellar mass]:
logMcenlow = -4
logMcenhigh = -3
logacenlow = -2
logacenhigh = -1

#Marginalise over rotation:
#Arotlow = 0.0
#Arothigh = 1.0
Arotlow = 0.0
Arothigh = 1.0e-12

#Priors on distance [True distance follows as:
#dgal_kpc + drange. Not used if drangelow and
#drangehigh are very small]:
dgal_kpc = 100.0
drangelow = -1e-5
drangehigh = 1e-5

###########################################################
#Post processing options:

#For calculating D+J-factors:
calc_Jfac = 'no'
alpha_Jfac_deg = 0.5
calc_Dfac = 'no'
alpha_Dfac_deg = 0.5
