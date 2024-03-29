###########################################################
#Binulator
###########################################################

#Python programme to bin discrete dwarf galaxy
#data for Jeans modelling with GravSphere. To
#work with your own data, you should write/add an
#"API" to put it in "binulator" format (see below).
#Binulator will then output all the files that 
#GravSphere needs to run its models.

###########################################################
#Main code:

#Suppress warning output:
import warnings
warnings.simplefilter("ignore")

#Forbid plots to screen so Binulator can run
#remotely:
import matplotlib as mpl
mpl.use('Agg')

#Imports & dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import emcee
from scipy.integrate import simps as integrator
from functions import *
from constants import *
from binulator_surffuncs import * 
from binulator_velfuncs import * 
from figures import * 
import sys

#Welcome blurb: 
print('###### BINULATOR VERSION 2.0 ######\n')

#Default to run on single processor:
nprocs = 1

#Default to contsant stars per bin:
Nbinkinarr = np.zeros(2)
Nbinkinarr_prop = Nbinkinarr

###########################################################
#Input data selection here:

#Initialise the data. This bit sets up everything for
#binulator, returning "outfile", rbin,surfden,surfdenerr,
#Nbin, Nbinkin, Rkin, vz, vzerr, mskin etc. (see e.g.
#binulator_initialise_Draco.py for details).

#MW satellites:
#from binulator_initialise_Draco import *
#from binulator_initialise_UMi import *
#from binulator_initialise_Carina import *
#from binulator_initialise_CVnI import *
#from binulator_initialise_LeoI import *
#from binulator_initialise_LeoII import *
#from binulator_initialise_Sculptor import *
#from binulator_initialise_Sextans import *
from binulator_initialise_Fornax import *
#from binulator_initialise_SegI import *
#from binulator_initialise_SMC import *
#from binulator_initialise_Ocen import *
#from binulator_initialise_OcenPB import *

#Mocks:
#from binulator_initialise_SMCmock import *
#from binulator_initialise_PlumCoreOm import *
#from binulator_initialise_PlumCuspOm import *
#from binulator_initialise_Ocenmock import *
#from binulator_initialise_FIRE import *
#from binulator_initialise_EDGE import *

#M31 satellites:
#from binulator_initialise_And21 import *

#Code options:
#quicktestSB => set to 'yes' if you want to fit just the 
#SB alone. This is worthwhile to make sure you have a 
#good fit before running the fit to the velocity data
#that takes longer to complete.
quicktestSB = 'no'

#velfiteasy => set to 'yes' if you want rapid non-para
#velocity binning with incorrect errors. This is worthwhile
#for quick testing of your velocity Nbinkin values. Not to
#be used for the real analysis.
velfiteasy = 'no'
if (velfiteasy == 'yes'):
    velfit = velfit_easy
else:
    velfit = velfit_full

#Some output about parameter choices:
print('Running on %d processors' % (nprocs))
print('Doing galaxy:',whichgal)
if (Nbin > 0):
    print('Number of stars per photometric bin:', Nbin)
else:   
    print('Photometric data pre-binned')
if (np.sum(Nbinkinarr) == 0):
    if (Nbinkin > 0):
        print('Number of stars per kinematic bin:', Nbinkin)
    else:
        print('Kinematic data pre-binned')
else:
    print('Manual radial binning for velocity data!')
    print('Number of stars in each bin:', Nbinkinarr)
    print('Will assume fixed stars per bin for any remaining bins:', Nbinkinarr[::-1][0])
print('Priors allow for kurtosis in the range: %.2f < k < %.2f' % \
      (kurt_calc(p0vin_max),kurt_calc(p0vin_min)))
if (Rfitmin > 0):
    print('Fitting surfden R >',Rfitmin)
if (Rfitmax > 0):
    print('Fitting surfden R <',Rfitmax)
if (Rfitvmin > 0):
    print('Fitting vel R >',Rfitvmin)
if (Rfitvmax > 0):
    print('Fitting vel R <',Rfitvmax)            
if (vfitmin != 0):
    print('Fitting v >',vfitmin)
if (vfitmax != 0):
    print('Fitting v <',vfitmax)    
print('Will write output to: %s_*' % (outfile))

#Fit surface density:
if (quicktestSB == 'yes'):
    print('### Only fitting the surface brightness profile ###')
print('Fitting the surface density profile:\n')

#Cut back to fit range:
if (Rfitmin > 0):
    Rfit_t = R[R > Rfitmin]
    surfdenfit_t = surfden[R > Rfitmin]
    surfdenerrfit_t = surfdenerr[R > Rfitmin]
else:
    Rfit_t = R
    surfdenfit_t = surfden
    surfdenerrfit_t = surfdenerr
if (Rfitmax > 0):
    Rfit = Rfit_t[Rfit_t < Rfitmax]
    surfdenfit = surfdenfit_t[Rfit_t < Rfitmax]
    surfdenerrfit = surfdenerrfit_t[Rfit_t < Rfitmax]
else:
    Rfit = Rfit_t
    surfdenfit = surfdenfit_t
    surfdenerrfit = surfdenerrfit_t
    
Rplot,surf_int,Rhalf_int,p0best = \
    tracerfit(Rfit,surfdenfit,surfdenerrfit,\
              p0in_min,p0in_max,nprocs)
print('Fitted Rhalf: %f -%f +%f' % \
    (Rhalf_int[0],Rhalf_int[0]-Rhalf_int[1],\
     Rhalf_int[2]-Rhalf_int[0]))
if (use_dataRhalf == 'yes'):
    Rhalf_int[0] = Rhalf

#Bin the velocity data and calculate the VSPs with uncertainties:
if (quicktestSB == 'no' and Nbinkin > 0):
    print('Fitting the velocity data:\n')
    nsamples = 2500
    
    #Cut back to fit range:
    Rf, vzfit, vzerrfit, msfit = \
        Rcutback(Rkin,vz,vzerr,mskin,Rfitvmin,Rfitvmax)
    
    #Do the fit:
    rbin,vzmeanbin,vzmeanbinlo,vzmeanbinhi,\
        vztwobin,vztwobinlo,vztwobinhi,\
        vzfourbin,vzfourbinlo,vzfourbinhi,\
        backampbin1,backampbinlo1,backampbinhi1,\
        backmeanbin1,backmeanbinlo1,backmeanbinhi1,\
        backsigbin1,backsigbinlo1,backsigbinhi1,\
        backampbin2,backampbinlo2,backampbinhi2,\
        backmeanbin2,backmeanbinlo2,backmeanbinhi2,\
        backsigbin2,backsigbinlo2,backsigbinhi2,\
        vsp1,vsp1lo,vsp1hi,vsp2,vsp2lo,vsp2hi,\
        ranal,vzfourstore,vsp1store,vsp2store = \
            velfit(Rf,vzfit,vzerrfit,msfit,Nbinkin,Nbinkinarr,\
                vfitmin,vfitmax,\
                p0vin_min,p0vin_max,p0best,\
                nsamples,outfile+'_vz',nprocs)
    print('Fitted VSP1: %f+%f-%f' % (vsp1,vsp1hi-vsp1,vsp1-vsp1lo))
    print('Fitted VSP2: %f+%f-%f' % (vsp2,vsp2hi-vsp2,vsp2-vsp2lo))
    
    #Bin the propermotion data:
    if (propermotion == 'yes'):
        print('Doing propermotions:')
        #Calculate the dipsersions in the radial
        #and tangential directions on the sky.
        #This assumes we have data:
        #x(kpc), y(kpc), vx(km/s), vy(km/s), and
        #errors vxerr(km/s), vyerr(km/s) with
        #weights msprop. Errors are assumed
        #to be Gaussian. Points per bin are given
        #by Nbinkin_prop.

        #Calculate R, vR and vphi and errors:
        Rp = np.sqrt(x**2. + y**2.)
        vR = (vx*x+vy*y)/Rp
        vphi = (vx*y-vy*x)/Rp
        vRerr = np.sqrt(vxerr**2.0 + vyerr**2.0)
        vphierr = np.sqrt(vxerr**2.0 + vyerr**2.0)
        
        #First tangential direction:
        #Cut back to fit range:
        Rpf, vphifit, vphierrfit, mspropfit = \
                Rcutback(Rp,vphi,vphierr,msprop,Rfitvmin,Rfitvmax)
                 
        rbinpt,vphimeanbin,vphimeanbinlo,vphimeanbinhi,\
            vphitwobin,vphitwobinlo,vphitwobinhi,\
            vphifourbin,vphifourbinlo,vphifourbinhi,\
            backptampbin1,backptampbinlo1,backptampbinhi1,\
            backptmeanbin1,backptmeanbinlo1,backptmeanbinhi1,\
            backptsigbin1,backptsigbinlo1,backptsigbinhi1,\
            backptampbin2,backptampbinlo2,backptampbinhi2,\
            backptmeanbin2,backptmeanbinlo2,backptmeanbinhi2,\
            backptsigbin2,backptsigbinlo2,backptsigbinhi2,\
            vsppt1,vsppt1lo,vsppt1hi,vsppt2,vsppt2lo,vsppt2hi,\
            ranalvphi,vphifourstore,vsppt1store,vsppt2store = \
                velfit(Rpf,vphifit,vphierrfit,mspropfit,Nbinkin_prop,Nbinkinarr_prop,\
                       vfitmin,vfitmax,\
                       p0vin_min,p0vin_max,p0best,\
                       nsamples,outfile+'_vphi',nprocs)
        print('Fitted VSP1prop tan: %f+%f-%f' % \
              (vsppt1,vsppt1hi-vsppt1,vsppt1-vsppt1lo))
        print('Fitted VSP2prop tan: %f+%f-%f' % \
              (vsppt2,vsppt2hi-vsppt2,vsppt2-vsppt2lo))
        
        #Now radial direction:
        #Cut back to fit range:
        Rpf, vRfit, vRerrfit, mspropfit = \
                Rcutback(Rp,vR,vRerr,msprop,Rfitvmin,Rfitvmax)
        
        rbinpR,vRmeanbin,vRmeanbinlo,vRmeanbinhi,\
            vRtwobin,vRtwobinlo,vRtwobinhi,\
            vRfourbin,vRfourbinlo,vRfourbinhi,\
            backpRampbin1,backpRampbinlo1,backpRampbinhi1,\
            backpRmeanbin1,backpRmeanbinlo1,backpRmeanbinhi1,\
            backpRsigbin1,backpRsigbinlo1,backpRsigbinhi1,\
            backpRampbin2,backpRampbinlo2,backpRampbinhi2,\
            backpRmeanbin2,backpRmeanbinlo2,backpRmeanbinhi2,\
            backpRsigbin2,backpRsigbinlo2,backpRsigbinhi2,\
            vsppR1,vsppR1lo,vsppR1hi,vsppR2,vsppR2lo,vsppR2hi,\
            ranalvR,vRfourstore,vsppR1store,vsppR2store = \
                velfit(Rpf,vRfit,vRerrfit,mspropfit,Nbinkin_prop,Nbinkinarr_prop,\
                       vfitmin,vfitmax,\
                       p0vin_min,p0vin_max,p0best,\
                       nsamples,outfile+'_vR',nprocs)
        print('Fitted VSP1prop rad: %f+%f-%f' % \
              (vsppR1,vsppR1hi-vsppR1,vsppR1-vsppR1lo))
        print('Fitted VSP2prop rad: %f+%f-%f' % \
              (vsppR2,vsppR2hi-vsppR2,vsppR2-vsppR2lo))

###########################################################
#Store the surface density, velocity dispersions, VSPs,
#best fit surfden parameters and Rhalf for GravSphere:
f = open(outfile+'_surfden.txt','w')
for i in range(len(Rfit)):
    f.write('%f %f %f\n' % \
        (Rfit[i], surfdenfit[i], surfdenerrfit[i]))
f.close()
f = open(outfile+'_p0best.txt','w')
for i in range(len(p0best)):
    f.write('%f\n' % \
        (p0best[i]))
f.close()
f = open(outfile+'_Rhalf.txt','w')
for i in range(len(Rhalf_int)):
    f.write('%f\n' % (Rhalf_int[i]))
f.close()
if (quicktestSB == 'no'):
    f = open(outfile+'_vel.txt','w')
    for i in range(len(rbin)):
        f.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                (rbin[i],vzmeanbin[i],vzmeanbinlo[i],vzmeanbinhi[i],\
                 vztwobin[i],vztwobinlo[i],vztwobinhi[i],\
                 vzfourbin[i],vzfourbinlo[i],vzfourbinhi[i],\
                 backampbin1[i],backampbinlo1[i],backampbinhi1[i],\
                 backmeanbin1[i],backmeanbinlo1[i],backmeanbinhi1[i],\
                 backsigbin1[i],backsigbinlo1[i],backsigbinhi1[i],\
                 backampbin2[i],backampbinlo2[i],backampbinhi2[i],\
                 backmeanbin2[i],backmeanbinlo2[i],backmeanbinhi2[i],\
                 backsigbin2[i],backsigbinlo2[i],backsigbinhi2[i]))
    f.close()
    f = open(outfile+'_vsps.txt','w')
    f.write('%f %f %f\n' % (vsp1,vsp1lo,vsp1hi))
    f.write('%f %f %f\n' % (vsp2,vsp2lo,vsp2hi))
    f.close()
    f = open(outfile+'_vsp1full.txt','w')
    for i in range(len(vsp1store)):
        f.write('%f\n' % (vsp1store[i]))
    f.close()
    f = open(outfile+'_vsp2full.txt','w')
    for i in range(len(vsp2store)):
        f.write('%f\n' % (vsp2store[i]))
    f.close()

    if (propermotion == 'yes'):
        f = open(outfile+'_velproptan.txt','w')
        for i in range(len(rbinpt)):
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                    (rbinpt[i],vphimeanbin[i],vphimeanbinlo[i],vphimeanbinhi[i],\
                     vphitwobin[i],vphitwobinlo[i],vphitwobinhi[i],\
                     vphifourbin[i],vphifourbinlo[i],vphifourbinhi[i],\
                     backptampbin1[i],backptampbinlo1[i],backptampbinhi1[i],\
                     backptmeanbin1[i],backptmeanbinlo1[i],backptmeanbinhi1[i],\
                     backptsigbin1[i],backptsigbinlo1[i],backptsigbinhi1[i],\
                     backptampbin2[i],backptampbinlo2[i],backptampbinhi2[i],\
                     backptmeanbin2[i],backptmeanbinlo2[i],backptmeanbinhi2[i],\
                     backptsigbin2[i],backptsigbinlo2[i],backptsigbinhi2[i]))
        f.close()
        f = open(outfile+'_velpropR.txt','w')
        for i in range(len(rbinpR)):
            f.write('%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % \
                    (rbinpR[i],vRmeanbin[i],vRmeanbinlo[i],vRmeanbinhi[i],\
                     vRtwobin[i],vRtwobinlo[i],vRtwobinhi[i],\
                     vRfourbin[i],vRfourbinlo[i],vRfourbinhi[i],\
                     backpRampbin1[i],backpRampbinlo1[i],backpRampbinhi1[i],\
                     backpRmeanbin1[i],backpRmeanbinlo1[i],backpRmeanbinhi1[i],\
                     backpRsigbin1[i],backpRsigbinlo1[i],backpRsigbinhi1[i],\
                     backpRampbin2[i],backpRampbinlo2[i],backpRampbinhi2[i],\
                     backpRmeanbin2[i],backpRmeanbinlo2[i],backpRmeanbinhi2[i],\
                     backpRsigbin2[i],backpRsigbinlo2[i],backpRsigbinhi2[i]))
        f.close()

###########################################################
#Plots:

##### Surface density fit #####
fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
ax.minorticks_on()
ax.tick_params('both', length=10, width=2, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.xlabel(r'Radius [kpc]',fontsize=myfontsize)
plt.ylabel(r'Surface density [${\rm kpc}^{-2}$]',fontsize=myfontsize)

plt.loglog()

plt.errorbar(R,surfden,surfdenerr,color='black',ecolor='black',\
             fmt='o',\
             linewidth=mylinewidth,marker='o',markersize=5,\
             alpha=0.5,label='Data')
plt.fill_between(Rplot,surf_int[3,:],surf_int[4,:],\
                 facecolor='blue',alpha=0.33,\
                 edgecolor='none',label='Fit (95\%)')
plt.fill_between(Rplot,surf_int[1,:],surf_int[2,:],\
                 facecolor='blue',alpha=0.66,\
                 edgecolor='none',label='Fit (68\%)')
            
#Best fit model:
surfbest = threeplumsurf(Rplot,p0best[0],p0best[1],p0best[2],\
                         p0best[3],p0best[4],p0best[5])
plt.plot(Rplot,surfbest,color='red',linewidth=2,label='Best fit')

#Check for negative density:
index = np.argsort(surfbest)
if (surfbest[index[0]] < 0):
    print('Warning! negative surface density at:',\
          Rplot[index[0]],surfbest[index[0]])

#Numerical projection (test):
intpnts = 250
theta = np.linspace(0,np.pi/2.0-1e-6,num=intpnts)
cth = np.cos(theta)
cth2 = cth**2.0
surf = np.zeros(len(Rplot))
for i in range(len(Rplot)):
    q = Rplot[i]/cth
    y = threeplumden(q,p0best[0],p0best[1],p0best[2],\
                     p0best[3],p0best[4],p0best[5])
    surf[i] = 2.0*Rplot[i]*integrator(y/cth2,theta)
plt.plot(Rplot,surf,color='orange',linewidth=2,linestyle='dashed',\
         label='Best fit (numerical)')

plt.axvline(x=Rhalf_int[0],color='blue',alpha=0.5,\
            linewidth=mylinewidth)

plt.xlim([xpltmin,xpltmax])
plt.ylim([surfpltmin,surfpltmax])

plt.legend(loc='upper right',fontsize=mylegendfontsize)
plt.savefig(outfile+'_surfden.pdf',bbox_inches='tight')

##### Tracer 3D density profile (best fit) #####
fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
ax.minorticks_on()
ax.tick_params('both', length=10, width=2, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.xlabel(r'Radius [kpc]',fontsize=myfontsize)
plt.ylabel(r'Density [${\rm kpc}^{-3}$]',fontsize=myfontsize)

plt.loglog()

denbest = threeplumden(Rplot,p0best[0],p0best[1],p0best[2],\
                         p0best[3],p0best[4],p0best[5])
plt.plot(Rplot,denbest,color='red',linewidth=2,label='Best fit')

plt.axvline(x=Rhalf_int[0],color='blue',alpha=0.5,\
            linewidth=mylinewidth)

plt.xlim([xpltmin,xpltmax])
plt.ylim([surfpltmin,surfpltmax])

plt.legend(loc='upper right',fontsize=mylegendfontsize)
plt.savefig(outfile+'_tracerden.pdf',bbox_inches='tight')

##### Tracer 3D mass profile (best fit) #####
fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
ax.minorticks_on()
ax.tick_params('both', length=10, width=2, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.xlabel(r'Radius [kpc]',fontsize=myfontsize)
plt.ylabel(r'Normalised Mass',fontsize=myfontsize)

plt.loglog()

massbest = threeplummass(Rplot,p0best[0],p0best[1],p0best[2],\
                         p0best[3],p0best[4],p0best[5])
plt.plot(Rplot,massbest,color='red',linewidth=2,label='Best fit')

plt.axvline(x=Rhalf_int[0],color='blue',alpha=0.5,\
            linewidth=mylinewidth)

plt.xlim([xpltmin,xpltmax])
plt.ylim([1e-4,10.0])

plt.legend(loc='upper left',fontsize=mylegendfontsize)
plt.savefig(outfile+'_tracermass.pdf',bbox_inches='tight')

##### Plot velocity bin fits ##### 
if (quicktestSB == 'no'):
    ##### Velocity dispersion profile ##### 
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    ax.minorticks_on()
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.xlabel(r'Radius [kpc]',fontsize=myfontsize)
    plt.ylabel(r'Velocity dispersion [${\rm km\,s}^{-1}$]',\
           fontsize=myfontsize)

    plt.semilogx()

    plt.errorbar(rbin,vztwobin,\
             yerr=[vztwobin-vztwobinlo,\
                   vztwobinhi-vztwobin],\
             color='black',ecolor='black',\
             fmt='o',label=r'$\sigma_z$',\
             linewidth=mylinewidth,marker='o',\
             markersize=5,alpha=0.5)

    if (propermotion == 'yes'):
        plt.errorbar(rbinpt,vphitwobin,\
                     yerr=[vphitwobin-vphitwobinlo,\
                           vphitwobinhi-vphitwobin],\
                     color='blue',ecolor='blue',\
                     fmt='o',label=r'$\sigma_\phi$',\
                     linewidth=mylinewidth,marker='o',\
                     markersize=5,alpha=0.5)
        plt.errorbar(rbinpR,vRtwobin,\
                     yerr=[vRtwobin-vRtwobinlo,\
                           vRtwobinhi-vRtwobin],\
                     color='red',ecolor='red',\
                     fmt='o',label=r'$\sigma_R$',\
                     linewidth=mylinewidth,marker='o',\
                     markersize=5,alpha=0.5)

    plt.axvline(x=Rhalf_int[0],color='blue',alpha=0.5,\
            linewidth=mylinewidth)

    plt.xlim([xpltmin,xpltmax])
    plt.ylim([vztwopltmin,vztwopltmax])

    plt.legend(fontsize=18,loc='upper right')
    plt.savefig(outfile+'_vztwo.pdf',bbox_inches='tight')

    ##### Fourth velocity moment ##### 
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    ax.minorticks_on()
    ax.tick_params('both', length=10, width=2, which='major')
    ax.tick_params('both', length=5, width=1, which='minor')
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    plt.xlabel(r'Radius [kpc]',fontsize=myfontsize)
    plt.ylabel(r'Fourth velocity moment [${\rm km}^4\,{\rm s}^{-4}$]',\
            fontsize=myfontsize)

    plt.loglog()

    plt.errorbar(rbin,vzfourbin,\
             yerr=[vzfourbin-vzfourbinlo,\
                   vzfourbinhi-vzfourbin],\
             color='black',ecolor='black',\
             fmt='o',\
             linewidth=mylinewidth,marker='o',\
             markersize=5,alpha=0.5)

    #Plot individual fits + falloff:
    if (Nbinkin > 0):
        for i in range(nsamples):
            plt.plot(ranal,vzfourstore[i,:],linewidth=1,alpha=0.25)

    plt.axvline(x=Rhalf_int[0],color='blue',alpha=0.5,\
            linewidth=mylinewidth)

    plt.xlim([xpltmin,xpltmax])
    plt.ylim([vzfourpltmin,vzfourpltmax])

    plt.savefig(outfile+'_vzfour.pdf',bbox_inches='tight')

    ###### VSP1 and VSP2 histograms #####
    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    nbin = 25
    n, bins, patches = plt.hist(vsp1store,bins=nbin,\
        range=(np.min(vsp1store),\
               np.max(vsp1store)),\
        facecolor='b', \
        histtype='bar',alpha=0.5, \
        label='vs_1')
 
    plt.xlabel(r'$v_{s1}\,[{\rm km}^4\,{\rm s}^{-4}]$',\
           fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    plt.ylim([0,np.max(n)])
    plt.savefig(outfile+'output_vsp1.pdf',bbox_inches='tight')

    fig = plt.figure(figsize=(figx,figy))
    ax = fig.add_subplot(111)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(mylinewidth)
    plt.xticks(fontsize=myfontsize)
    plt.yticks(fontsize=myfontsize)

    nbin = 25
    n, bins, patches = plt.hist(vsp2store,bins=nbin,\
        range=(np.min(vsp2store),\
               np.max(vsp2store)),\
        facecolor='b', \
        histtype='bar',alpha=0.5, \
        label='vs_2')
 
    plt.xlabel(r'$v_{s2}\,[{\rm km}^4\,{\rm s}^{-4}]$',\
           fontsize=myfontsize)
    plt.ylabel(r'$N$',fontsize=myfontsize)
    plt.ylim([0,np.max(n)])
    plt.savefig(outfile+'output_vsp2.pdf',bbox_inches='tight')


###########################################################
#Exit:
print('\nThank you for using the Binulator! Have a nice day.\n')
