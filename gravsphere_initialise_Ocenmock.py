import numpy as np
from constants import * 
from functions import * 

#This file contains all the code options and choices for 
#running a given model. Throughout, -1 means auto-calculate.

#Run on multiple processors:
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
infile = output_base+whichgal+'/dgal_%.1f_%d/' % (dgal_kpc_offset,axis)+whichgal
outdirbase = output_base+whichgal+'/dgal_%.1f_%d/' % (dgal_kpc_offset,axis)
print('Using mock Omega cen distance of: %.1f kpc' % (dgal_kpc_offset))
print('True Omega cen distance: %.1f kpc' % (dgal_kpc_true))
print('Projecting along axis:', axis)
dgal_kpc = dgal_kpc_offset

#Plot ranges and sample points [-1 means auto-calculate]:
rplot_inner = 1e-6
rplot_outer = 5.0
rplot_pnts = 2500
y_sigLOSmax = 25
ymin_Sigstar = 1e-3
ymax_Sigstar = 1e5
yMlow = 1e2
yMhigh = 1e9
yrholow = 1e3
yrhohigh = 1e12
alp3sig = 0.0
sigmlow = 1e-3
sigmhigh = 5.0

#Code options:
propermotion = 'yes'
virialshape = 'yes'

#Overplot true solution (for mock data). If 
#yes, then the true solutions should be passed
#in: ranal,betatrue(ranal),betatruestar(ranal),
#truemass(ranal),trueden(ranal),truedlnrhodlnr(ranal).
overtrue = 'yes'
massdata = np.genfromtxt('../Data/Ocen_mock/h383_mass_late.txt',dtype='f8')
ranal = massdata[:,0]
truemass = massdata[:,1]
betdata = np.genfromtxt('../Data/Ocen_mock/h383_beta_late.txt',dtype='f8')
betatruestar = np.interp(ranal,betdata[:,0],betdata[:,1])
betatrue = 2.0*betatruestar/(1.0+betatruestar)
dendata = np.genfromtxt('../Data/Ocen_mock/h383_density_late.txt',dtype='f8')
trueden = np.interp(ranal,dendata[:,0],dendata[:,1])
truedlnrhodlnr = np.zeros(len(ranal))

#Radial grid range for Jeans calculation:
rmin = 5.0e-6
rmax = 5.0
intpnts = np.int(250)

#Galaxy properties. Assume here that the baryonic mass
#has the same radial profile as the tracer stars. If this
#is not the case, you should set Mstar_rad and Mstar_prof 
#here. The variables barrad_min, barrad_max and bar_pnts 
#set the radial range and sampling of the baryonic mass model.
Mstar = 3.0e6
Mstar_err = Mstar*0.95
baryonmass_follows_tracer = 'yes'
nurad_min = 1.0e-6
nurad_max = 10.0
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
tSFlow = 13.8-12.0
tSFhigh = 13.8-10.0
M200abundlow = 1.0e9
M200abundhigh = 5.0e10
logM200low = np.log10(M200abundlow)
logM200high = np.log10(M200abundhigh)
nsig_c200low = -10.0
nsig_c200high = 10.0
logmWDMlow = np.log10(1.0e5)
logmWDMhigh = np.log10(1.0e6)
logsigmaSIlow = np.log10(1.0e-6)
logsigmaSIhigh = np.log10(1.0e-5)
mleftmax = 1.0
mleftmin = 1.0e-6
logmleftlow = np.log10(mleftmin)
logmlefthigh = np.log10(mleftmax)
mfaclow = 0.0
mfachigh = 10.0

#Infall redshift fixed by tSF. For
#isolated galaxies that are star forming today
#this should be a tight prior around zin=0.
#[cosmological parameters set in constants.py]
zinlow = redshift_from_time(tSFhigh,OmegaM,OmegaL,h)
zinhigh = redshift_from_time(tSFlow,OmegaM,OmegaL,h)
print('Redshift of infall based on tSF:', \
      zinlow, zinhigh)

#Priors on central dark mass [set logMcenlow/high very negative
#to switch this off. Mcen is the mass in Msun; acen is the
#scale length in kpc, usually assumed smaller than Rhalf
#to avoid degeneracies with the stellar mass]:
logMcenlow = 2.0
logMcenhigh = 6.0
logacenlow = -6.0
logacenhigh = -3.5

#Priors on rotation [set Arotlow = Arothigh = 0 to switch off.
#Arot defined s.t. vphimean^2 / (2 sigr^2) = Arot(r/Rhalf)
#which yields linear rotation with radius. Arot = 0.5 means
#an equal balance of rotation and pressure support at Rhalf].
Arotlow = 0.0
Arothigh = 1.0e-12

#Priors on distance [True distance follows as:
#dgal_kpc + drange. Not used if drangelow and
#drangehigh are very small]:
drangelow = -1.0
drangehigh = 1.0

###########################################################
#Post processing options:

#For calculating D+J-factors:
calc_Jfac = 'no'
alpha_Jfac_deg = 0.5
calc_Dfac = 'no'
alpha_Dfac_deg = 0.5
