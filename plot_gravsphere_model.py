###########################################################
#plot_gravsphere_model
###########################################################

#Python programme to plot the line of sight velocity
#dispersion for a specific chosen GravSphere model.

###########################################################
#Main code:

#Imports & dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from functions import *
from constants import *
from figures import *
import sys

#Plot parameters:
y_sigLOSmax = 15.0

#Set up example model:

#Velocity anisotropy profile of stars:
bet0star = 0.0
betinfstar = 0.5
log10r0 = np.log10(0.5)
n = 2.0
betpars = np.array([bet0star,betinfstar,log10r0,n])

#Baryonic mass profile (three Plummer model):
nu = multiplumden
Sigfunc = multiplumsurf
M1 = 1.0
M2 = 0.5
M3 = 0.0
a1 = 0.25
a2 = 0.75
a3 = 1.0
nupars = np.array([M1,M2,M3,a1,a2,a3])
Mstar = 1.0e6

#Mass profile (coreNFWtides = dark matter; stars are added to this):
M = lambda r, Mpars: \
    corenfw_tides_mass(r,Mpars[0],Mpars[1],Mpars[2],\
                         Mpars[3],Mpars[4],Mpars[5])
M200 = 5.0e9
c200 = 10.0
rc = 0.75
n = 1.0
rt = 5.0
delta = 5.0
Mpars = np.array([M200,c200,rc,n,rt,delta])

#Set up radial bins:
rmin = 0.01
rmax = 5.0
pnts = 500
rbin = np.logspace(np.log10(rmin),np.log10(rmax),pnts)

#Set up baryonic mass profile (assuming stellar mass profile
#follows tracer distribution profile):
Mstar_rad = np.linspace(rmin,\
    rmax,pnts)
norm = nupars[0] + nupars[1] + nupars[2]
Mstar_prof = \
    threeplummass(Mstar_rad,nupars[0]/norm,\
                  nupars[1]/norm,nupars[2]/norm,\
                  nupars[3],nupars[4],nupars[5])

#Calculate radial + line of sight dispersion + surface brightness
#profile for this model:
sigp_fit = lambda r1, r2, nupars, Mpars, betpars, Mstar: \
            sigp(r1,r2,nu,Sigfunc,M,beta,betaf,nupars,Mpars,betpars,\
                 Mstar_rad,Mstar_prof,Mstar,Guse,rmin,rmax)
sigr2, Sig, sigLOS2 = \
        sigp_fit(rbin,rbin,nupars,\
                 Mpars,betpars,Mstar)

###########################################################
#Plots:

##### Projected velocity dispersion #####  
fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.plot(np.log10(rbin),np.sqrt(sigLOS2)/1000.0,'k',linewidth=mylinewidth,\
         label=r'GravSphere model')

plt.xlabel(r'${\rm Log}_{10}[R/{\rm kpc}]$',\
               fontsize=myfontsize)
plt.ylabel(r'$\sigma_{\rm LOS}[{\rm km\,s}^{-1}]$',\
               fontsize=myfontsize)

plt.ylim([0,y_sigLOSmax])
plt.xlim([np.log10(np.min(rbin)),np.log10(np.max(rbin))])

plt.savefig('gs_model_sigLOS.pdf',bbox_inches='tight')
