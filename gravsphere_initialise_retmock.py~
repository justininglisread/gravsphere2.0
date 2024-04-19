import numpy as np
from constants import * 
from functions import * 
import h5py

#Functions:
def vdisp(x,y,z,vx,vy,vz,ms,rgmin,rgmax,rgbins):
    rs = np.sqrt(x**2. + y**2. + z**2.)
    vr = (vx*x+vy*y+vz*z)/rs
    vphi = (vy*x - vx*y)/np.sqrt(x**2.+y**2.)
    vtheta = (vr*z-vz*rs)/np.sqrt(x**2.+y**2.)
    
    if (rgbins > 0):
        #Linear binning:
        rgbin = np.linspace(rgmin,rgmax,rgbins)
        rgbins_plus_one = np.linspace(rgmin,rgmax,rgbins+1)
    else:
        #Log binning with -rgbins in log-space:
        rgbin = np.linspace(np.log10(rgmin),\
                            np.log10(rgmax),-rgbins)
        rgbins_plus_one = np.linspace(np.log10(rgmin),\
                                      np.log10(rgmax),-rgbins+1)
        rs = np.log10(rs)
        
    vsnorm = np.histogram(rs,rgbins_plus_one,\
                          weights=ms)[0]
    vrmean = np.histogram(rs,rgbins_plus_one,\
                          weights=vr*ms)[0] / vsnorm
    vrrms = np.histogram(rs,rgbins_plus_one,\
                         weights=vr**2.*ms)[0] / vsnorm
    vrsig = np.sqrt(vrrms - vrmean**2.)
    vphimean = np.histogram(rs,rgbins_plus_one,\
                            weights=vphi*ms)[0] / vsnorm
    vphirms = np.histogram(rs,rgbins_plus_one,\
                           weights=vphi**2.*ms)[0] / vsnorm
    vphisig = np.sqrt(vphirms - vphimean**2.)
    vthetamean = np.histogram(rs,rgbins_plus_one,\
                              weights=vtheta*ms)[0] / vsnorm
    vthetarms = np.histogram(rs,rgbins_plus_one,\
                             weights=vtheta**2.*ms)[0] / vsnorm
    vthetasig = np.sqrt(vthetarms - vthetamean**2.)

    if (rgbins < 0):
        rgbin = 10.0**rgbin
    
    return rgbin, vrmean, vrrms, vrsig, vphimean, \
        vphirms, vphisig, vthetamean, vthetarms, vthetasig, vsnorm

def calc_mass_density(r,ms,masstrue,Nbin):
    #Nbin is the number of particles / bin:
    index = np.argsort(r)
    right_bin_edge = np.zeros(len(r))
    norm = np.zeros(len(r))
    mass = np.zeros(len(r))
    cnt = 0
    jsum = 0

    for i in range(len(r)):
        if (jsum < Nbin):
            norm[cnt] = norm[cnt] + masstrue[index[i]]
            right_bin_edge[cnt] = r[index[i]]
            jsum = jsum + ms[index[i]]
            mass[cnt] = mass[cnt] + masstrue[index[i]]
        if (jsum >= Nbin):
            jsum = 0.0
            cnt = cnt + 1
            if (cnt < len(mass)):
                mass[cnt] = mass[cnt-1]
            
    right_bin_edge = right_bin_edge[:cnt]
    norm = norm[:cnt]
    mass = mass[:cnt]
    den = np.zeros(cnt)
    rbin = np.zeros(cnt)

    for i in range(len(rbin)):
        if (i == 0):
            den[i] = norm[i] / \
                     (4.0/3.0*np.pi*right_bin_edge[i]**3.0)
            rbin[i] = right_bin_edge[i]/2.0
        else:
            den[i] = norm[i] / \
                     (4.0/3.0*np.pi*right_bin_edge[i]**3.0-\
                      4.0/3.0*np.pi*right_bin_edge[i-1]**3.0)
            rbin[i] = (right_bin_edge[i]+right_bin_edge[i-1])/2.0
            denerr = den / np.sqrt(Nbin)
            
    return rbin, den, mass, denerr

#This file contains all the code options and choices for 
#running a given model. Throughout, -1 means auto-calculate.

#Run on multiprocessor:
nprocs = 10

#Data files and output base filename:
data_file = \
    '/vol/ph/astro_data/jread/FIRE_mock/m12_halos.hdf5'

#sim_name, sat_index = 'm12m_res7100', 38
#sim_name, sat_index = 'm12m_res7100', 62
#sim_name, sat_index = 'm12f_res7100', 10
#sim_name, sat_index = 'm12i_res7100', 4
#sim_name, sat_index = 'm12m_res7100', 103
#sim_name, sat_index = 'm12f_res7100', 23
#sim_name, sat_index = 'm12f_res7100', 54
#sim_name, sat_index = 'm12f_res7100', 15
#sim_name, sat_index = 'm12m_res7100', 13
#sim_name, sat_index = 'm12m_res7100', 73
#sim_name, sat_index = 'm12m_res7100', 28
#sim_name, sat_index = 'm12f_res7100', 2
#sim_name, sat_index = 'm12f_res7100', 36
#sim_name, sat_index = 'm12f_res7100', 39
#sim_name, sat_index = 'm12i_res7100', 28
#sim_name, sat_index = 'm12m_res7100', 24
#sim_name, sat_index = 'm12m_res7100', 60
#sim_name, sat_index = 'm12f_res7100', 23
sim_name, sat_index = 'm12m_res7100', 43
#sim_name, sat_index = 'm12i_res7100', 15
#sim_name, sat_index = 'm12m_res7100', 33

whichgal = '%s_%d' % (sim_name,sat_index)
infile = output_base+'FIRE_mock/%s/%s' % (whichgal,whichgal)
outdirbase = output_base+'FIRE_mock/%s/' % (whichgal)

#Plot ranges and sample points [-1 means auto-calculate]:
rplot_inner = 1e-2
rplot_outer = 5.0
rplot_pnts = 50
y_sigLOSmax = 25
ymin_Sigstar = 1e-6
ymax_Sigstar = 100
yMlow = 1e4
yMhigh = 1e10
yrholow = 1e5
yrhohigh = 1e9
alp3sig = 0.2
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
with h5py.File(data_file, 'r') as f:
    gr = f[sim_name]
    position_center = gr['position'][sat_index]
    velocity_center = gr['velocity'][sat_index]
    star_position = np.stack(gr['star.part.position'][sat_index], axis=1)
    star_velocity = np.stack(gr['star.part.velocity'][sat_index], axis=1)
    dark_position = np.stack(gr['dark.part.position'][sat_index], axis=1)
    dark_velocity = np.stack(gr['dark.part.velocity'][sat_index], axis=1)
    
    mass_star = gr['star.part.mass'][sat_index]
    mass_dark = gr['dark.part.mass'][sat_index]
    dark_position = dark_position - position_center
    dark_velocity = dark_velocity - velocity_center
    star_position = star_position - position_center
    star_velocity = star_velocity - velocity_center
    
#Here assuming stars contribute negligibly:
rdark = np.sqrt(dark_position[:,0]**2.0+\
                dark_position[:,1]**2.0+\
                dark_position[:,2]**2.0)
Nbintrue = 250
masstruenorm = mass_dark/np.sum(mass_dark)*len(mass_dark)
ranal, trueden, truemass, denerr = \
    calc_mass_density(rdark,masstruenorm,mass_dark,Nbintrue)

masstruenormstar = mass_star/np.sum(mass_star)*len(mass_star)
rgmin = np.min(ranal)
rgmax = np.max(ranal)
rgbins = -25
rgbin, vrmean, vrrms, vrsig, vphimean, \
    vphirms, vphisig, vthetamean, \
    vthetarms, vthetasig, vnorm = \
    vdisp(star_position[:,0],star_position[:,1],\
          star_position[:,2],star_velocity[:,0],\
          star_velocity[:,1],star_velocity[:,2],masstruenormstar,\
          rgmin,rgmax,rgbins)
vt = np.sqrt((vphisig**2.0 + vthetasig**2.0)/2.0)
vterr = vt / np.sqrt(vnorm)
vrerr = vrsig / np.sqrt(vnorm)
beterrfractemp = (vterr/vt + vrerr/vrsig)
betatruetemp = 1.0 - vt/vrsig
betatruestartemp = (vrsig - vt)/(vrsig + vt)
betatrue = np.interp(ranal,rgbin,betatruetemp)
betatruestar = np.interp(ranal,rgbin,betatruestartemp)
beterrfrac = np.interp(ranal,rgbin,beterrfractemp)
truedlnrhodlnr = np.zeros(len(trueden))

#Calculate true M200 and true vmax:
vctrue = np.sqrt(Guse * truemass / ranal)
vmaxtrue = np.max(vctrue)/kms
rhomean = truemass / (4.0/3.0*np.pi*ranal**3.0)
jj = 0
while (rhomean[jj] > 200.0*rhocrit):
    jj = jj + 1
M200true = truemass[jj]
print('Mock data truth properties .... ')
print('M200 (1.0e9 Msun), Log10(M200/Msun):', M200true/1.0e9, np.log10(M200true))
print('vmax (km/s):', vmaxtrue)

#Radial grid range for Jeans calculation:
rmin = -1
rmax = -1
intpnts = -1.0

#Galaxy properties. Assume here that the baryonic mass
#has the same radial profile as the tracer stars. If this
#is not the case, you should set Mstar_rad and Mstar_prof 
#here. The variables barrad_min, barrad_max and bar_pnts 
#set the radial range and sampling of the baryonic mass model.
Mstar = -1.0
Mstar_err = 1.0
baryonmass_follows_tracer = 'yes'
barrad_min = 0.0
barrad_max = 10.0
bar_pnts = 250


###########################################################
#Priors

#For surface density fit tracertol = [0,1] sets the spread 
#around the best-fit value from the binulator.
tracertol = 0.75

#Cosmology priors on the coreNFWtides model. mWDM(keV) is
#the mass of a thermal relic; <0 means CDM; sig_c200 is 
#the scatter of c200 in log10 space. If the cosmo_cprior
#is set, then we include a Gaussian spread in M200-c200 in
#the likelihood. Without this, M200-c200 enters only if 
#used to set the priors, below.
cosmo_cprior = 'no'
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
bet0min = -0.5
bet0max = 1.0
betinfmin = -0.5
betinfmax = 1.0

#CoreNFWtides priors:
logM200low = 7.5
logM200high = 11.5
clow = 1.0
chigh = 100.0
rclow = 1e-2
rchigh = 10.0
logrclow = np.log10(rclow)
logrchigh = np.log10(rchigh)
nlow = 0.0
nhigh = 1.0
rtlow = 1.0
rthigh = 20.0
logrtlow = np.log10(rtlow)
logrthigh = np.log10(rthigh)
dellow = 3.01
delhigh = 5.0

if (cosmo_cprior == 'yes'):
    clow = 1.0
    chigh = 100.0

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
dgal_kpc = 100.0
drangelow = -1.0e-5
drangehigh = 1.0e-5
    
###########################################################
#Post processing options:

#For calculating D+J-factors:
calc_Jfac = 'no'
alpha_Jfac_deg = 0.5 
calc_Dfac = 'no'
alpha_Dfac_deg = 0.5
