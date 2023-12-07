import numpy as np
from binulator_apis import *
from constants import * 
import h5py

#Run on multiprocessor:
nprocs = 10

#Parameters:
Nbinkin = 25
Nbin = 25
vzerr_in = 1.0

#Data files and output base filename:
data_file = \
    '/vol/ph/astro_data/jread/FIRE_mock/m12_halos.hdf5'

#Full suite to run:
#sim_name, sat_index = 'm12m_res7100', 38
#sim_name, sat_index = 'm12m_res7100', 62
#sim_name, sat_index = 'm12f_res7100', 10
#sim_name, sat_index = 'm12i_res7100', 4
#sim_name, sat_index = 'm12m_res7100', 103
#sim_name, sat_index = 'm12f_res7100', 23
#sim_name, sat_index = 'm12f_res7100', 54
#sim_name, sat_index = 'm12f_res7100', 15
#sim_name, sat_index = 'm12m_res7100', 13
sim_name, sat_index = 'm12m_res7100', 73
#sim_name, sat_index = 'm12m_res7100', 28
#sim_name, sat_index = 'm12f_res7100', 2
#sim_name, sat_index = 'm12f_res7100', 36
#sim_name, sat_index = 'm12f_res7100', 39
#sim_name, sat_index = 'm12i_res7100', 28
#sim_name, sat_index = 'm12m_res7100', 24
#sim_name, sat_index = 'm12m_res7100', 60
#sim_name, sat_index = 'm12f_res7100', 23
#sim_name, sat_index = 'm12m_res7100', 43
#sim_name, sat_index = 'm12i_res7100', 15
#sim_name, sat_index = 'm12m_res7100', 33
    
with h5py.File(data_file, 'r') as f:
    gr = f[sim_name]
    position_center = gr['position'][sat_index]
    velocity_center = gr['velocity'][sat_index]
    star_position = np.stack(gr['star.part.position'][sat_index], axis=1)
    star_velocity = np.stack(gr['star.part.velocity'][sat_index], axis=1)
    dark_position = np.stack(gr['dark.part.position'][sat_index], axis=1)
    dark_velocity = np.stack(gr['dark.part.velocity'][sat_index], axis=1)
    
    mass = gr['star.part.mass'][sat_index]
    dark_position = dark_position - position_center
    dark_velocity = dark_velocity - velocity_center
    star_position = star_position - position_center
    star_velocity = star_velocity - velocity_center

whichgal = '%s_%d' % (sim_name,sat_index)
outfile = output_base+'FIRE_mock/%s/%s' % (whichgal,whichgal)

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
p0in_min = np.array([1e-4,1e-4,1e-4,0.1,0.1,0.1])
p0in_max = np.array([1e3,1e3,1e3,1.5,1.5,1.5])
Rfitmin = -1
Rfitmax = -1

#Priors for binulator velocity dispersion calculation.
#Array values are: [vzmean,alp,bet,backamp1,backmean1,backsig1,
#backamp2,backmean2,backsig2], where alp is ~the dispersion,
#bet=[1,5] is a shape parameter, and "back" is a Gaussian of
#amplitude "backamp", describing some background. The first
#Gaussian is assumed to have mean >0; the second mean <0, hence
#the prior range on the mean is +ve definite for both.
p0vin_min = np.array([-20.0,5.0,1.0,\
                      1e-5,20.0,30.0,\
                      1e-5,20.0,30.0])
p0vin_max = np.array([20.0,60.0,5.0,\
                      1e-4,95.0,125.0,\
                      1e-4,95.0,125.0])
vfitmin = 0
vfitmax = 0
Rfitvmin = -1
Rfitvmax = -1

#Output total number of stars:
print('Total stars in this mock:', len(star_position))

#Convert input data to binulator format (see APIs, above).
#Note that we also calculate Rhalf directly from the data here.
#If use_dataRhalf = 'yes', then we will use this data Rhalf
#instead of the fitted Rhalf. This can be useful if the 
#SB falls off very steeply, which cannot be captured easily
#by the sum over Plummer spheres that binulator/gravsphere
#assumes.
Rkin = np.sqrt(star_position[:,0]**2.0 + star_position[:,1]**2.0)
ms = mass/np.sum(mass)*len(mass)
print('Total effective no. of tracers (photometric):', np.sum(ms))
R, surfden, surfdenerr, Rhalf = \
    binthedata(Rkin,ms,Nbin)
print('Data Rhalf:', Rhalf)
mskin = ms
vz = star_velocity[:,2] + \
     np.random.normal(0.0, vzerr_in, len(ms))
vzerr = np.zeros(len(vz)) + vzerr_in
use_dataRhalf = 'no'

#Propermotions:
propermotion = 'yes'
if (propermotion == 'yes'):
    Nbinkin_prop = Nbinkin
    x = star_position[:,0]
    y = star_position[:,1]
    vx = star_velocity[:,0] + \
         np.random.normal(0.0, vzerr_in, len(ms))
    vy = star_velocity[:,1] + \
         np.random.normal(0.0, vzerr_in, len(ms))
    vxerr = np.zeros(len(vz)) + vzerr_in
    vyerr = np.zeros(len(vz)) + vzerr_in
    msprop = ms
