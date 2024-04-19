import numpy as np
from constants import * 
from functions import * 

#This file contains all the code options and choices for 
#running a given model. Throughout, -1 means auto-calculate.

#Set number of processors to run on:
nprocs = 10

#Data files and output base filename:
whichgal = 'Fornax'
infile = output_base+whichgal+'/'+whichgal
outdirbase = output_base+whichgal+'/'

#Plot ranges and sample points [-1 means auto-calculate]:
rplot_inner = 1e-2
rplot_outer = 5.0
rplot_pnts = 50
y_sigLOSmax = 25
ymin_Sigstar = 1e-4
ymax_Sigstar = 100
yMlow = 1e4
yMhigh = 1e10
yrholow = 1e5
yrhohigh = 1e10
alp3sig = 0.0

#Code options:
propermotion = 'no'
virialshape = 'yes'

#Overplot true solution (for mock data). If 
#yes, then the true solutions should be passed
#in: ranal,betatrue(ranal),betatruestar(ranal),
#truemass(ranal),trueden(ranal),truedlnrhodlnr(ranal).
overtrue = 'no'

#Radial grid range for Jeans calculation:
rmin = -1.0
rmax = -1.0
intpnts = -1.0

#Galaxy properties. Assume here that the baryonic mass
#has the same radial profile as the tracer stars. If this
#is not the case, you should set Mstar_rad and Mstar_prof 
#here. The variables barrad_min, barrad_max and bar_pnts 
#set the radial range and sampling of the baryonic mass model.
Mstar = 4.3e7
Mstar_err = Mstar * 0.25
baryonmass_follows_tracer = 'yes'
nurad_min = 1.0e-3
nurad_max = 50.0
nu_pnts = np.int(1e4)

###########################################################
#Priors

#Velocity anisotropy priors. If logbetr0min=logbetr0max,
#it will be set automatically based on the half light
#radius:
logbetr0min = 0.0
logbetr0max = 0.0
betnmin = 1.0
betnmax = 3.0
bet0min = -0.1
bet0max = 0.1
betinfmin = -0.1
betinfmax = 1.0

#Dark matter model priors:
#M200option = 'abund'
M200option = 'free'
WDMoption = 'CDM'
#WDMoption = 'free'
SIDMoption = 'CDM'
#SIDMoption = 'free'
tideoption = 'orbit'
#tideoption = 'free'
#DMheatoption = 'tight'
DMheatoption = 'free'
#DMheatoption = 'CDM'
zinoption = 'zero'
outdirbase = outdirbase+M200option+WDMoption+SIDMoption+\
             tideoption+DMheatoption+zinoption+'/'

#tSF constrained by SF in Fornax
#(see Read et al. 2019):
tSFlow = 13.8-1.75-1.0
tSFhigh = 13.8-1.75+1.0
print('Min/max star formation length (Gyrs)', \
      tSFlow, tSFhigh)

#Set M200 prior based on abundance matching
#(see Read & Erkal 2019):
if (M200option == 'abund'):
    M200abundlow, M200abundhigh = \
            estimate_M200abund(Mstar-Mstar_err,\
                               Mstar+Mstar_err,tSFlow,tSFhigh)
    M200abundhigh = M200abundhigh*2
elif (M200option == 'free'):
    M200abundlow = 1.0e9
    M200abundhigh = 5.0e10
print('Using prior on M200 (1e9 Msun):', \
      M200abundlow/1.0e9,M200abundhigh/1.0e9)
logM200low = np.log10(M200abundlow)
logM200high = np.log10(M200abundhigh)
nsig_c200low = -10.0
nsig_c200high = 10.0

#Cosmology priors:
if (WDMoption == 'free'):
    logmWDMlow = np.log10(0.01)
    logmWDMhigh = np.log10(1000)
elif (WDMoption == 'CDM'):
    logmWDMlow = np.log10(1.0e5)
    logmWDMhigh = np.log10(1.0e6)
if (SIDMoption == 'free'):
    logsigmaSIlow = np.log10(1.0e-3)
    logsigmaSIhigh = np.log10(1.0e3)
elif (SIDMoption == 'CDM'):
    logsigmaSIlow = np.log10(1.0e-6)
    logsigmaSIhigh = np.log10(1.0e-5)        

#DMheating priors. Here, mfac is a dimensionless prior
#that determines the amount of DM heating via the ratio
#mfac*Mstar/M200 such that mfac=1 is the default model,
#while higher/lower mfac means more/less heating.
if (DMheatoption == 'free'):
    mfaclow = 0.0
    mfachigh = 2.0
elif (DMheatoption == 'tight'):
    mfaclow = 0.99
    mfachigh = 1.01
elif (DMheatoption == 'CDM'):
    mfaclow = 0.0
    mfachigh = 1.0e-3

#Constrain mass loss using known Fornax
#apo and peri (from Pace et al. 22).
#Note that mleft parameter is in range
#[0,1] and describes the fraction of initial
#M200 left.
if (tideoption == 'orbit'):
    rp, ra = 76.7-27.9, 152.7+9.7
    mleftmin = mleftmaxcalc(M200abundlow,rp,ra,\
                            galmodel='MW')
    mleftmin = mleftmin / M200abundhigh
elif (tideoption == 'free'):
    mleftmin = 1.0e-6
mleftmax = 1.0
logmleftlow = np.log10(mleftmin)
logmlefthigh = np.log10(mleftmax)
print('Min/max mass left based on orbit (fraction of M200):',\
      mleftmin,mleftmax)

#Infall redshift fixed by tSF. For
#isolated galaxies that are star forming today,
#this should be a tight prior around zin=0.
#[cosmological parameters set in constants.py]
if (zinoption == 'zero'):
    zinlow = 0.0
    zinhigh = 0.01
    print('M200-c200 relation fixed at redshift z=0:', \
          zinlow, zinhigh)
elif (zinoption == 'infall'):
    zinlow = redshift_from_time(tSFhigh,OmegaM,OmegaL,h)
    zinhigh = redshift_from_time(tSFlow,OmegaM,OmegaL,h)
    print('M200-c200 relation fixed by redshift of infall based on tSF:', \
          zinlow, zinhigh)

#Priors on central dark mass [set logMcenlow/high very negative
#to switch this off. Mcen is the mass in Msun; acen is the
#scale length in kpc, usually assumed smaller than Rhalf
#to avoid degeneracies with the stellar mass]:
logMcenlow = -4
logMcenhigh = -3
logacenlow = -5
logacenhigh = -2

#Priors on rotation [Arot defined as:
#vphimean^2 / (2 sigr^2) = Arot(r/Rhalf) which yields linear
#rotation with radius. (Arot = 0.5 means an equal balance of
#rotation and pressure support at Rhalf.)]:
Arotlow = 0.0
Arothigh = 1.0e-12

#Priors on distance [True distance follows as:
#dgal_kpc + drange. Not used if drangelow and
#drangehigh are very small]:
dgal_kpc = 138.0
drangelow = -1.0e-5
drangehigh = 1.0e-5

###########################################################
#Post processing options:

#For calculating D+J-factors:
calc_Jfac = 'no'
alpha_Jfac_deg = 0.5 
calc_Dfac = 'no'
alpha_Dfac_deg = 0.5
