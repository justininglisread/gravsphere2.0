###########################################################
#plot_coreNFWtides
###########################################################

#Python programme to compare the coreNFWtides model to the
#GC mock density profiles.

###########################################################
#Main code:

#Imports & dependencies:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.optimize import curve_fit
from functions import *
from constants import *
from figures import *
import sys

#Set up mock density profile (PlumCoreOm):
rho0,r0,alp,bet,gam,rstar,ra = \
    np.array([400./1000. * 1000.**3.,1.0,1.0,\
              3.0,0.0,25./100.,100./100.*25./100.])
ranal = np.logspace(-3,1,np.int(250))
betatrue = ranal**2./(ranal**2. + ra**2.)
betatruestar = betatrue/(2.0-betatrue)
truemass = alpbetgammass(ranal,rho0,r0,alp,bet,gam)
trueden = alpbetgamden(ranal,rho0,r0,alp,bet,gam)
truedlnrhodlnr = alpbetgamdlnrhodlnr(ranal,rho0,r0,alp,bet,gam)

#Perform a simple fit of coreNFWtides to the above:
p0in = np.array([10653324241.236410,51.678920,14.714924,0.479031,\
                 1.0,2.0])
pfits, pcovar = curve_fit(corenfw_tides_den,ranal,trueden,\
                          p0=p0in,sigma=None,maxfev=10000)
corenfwtides = corenfw_tides_den(ranal,pfits[0],pfits[1],\
                                 pfits[2],pfits[3],\
                                 pfits[4],pfits[5])

#Output best fit parameters:
print('Best fit: M200: %f | c %f | rc %f | n %f | rt %f | delta %f\n' \
      % (pfits[0],pfits[1],pfits[2],pfits[3],pfits[4],pfits[5]))


###########################################################
#Plots:

fig = plt.figure(figsize=(figx,figy))
ax = fig.add_subplot(111)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(mylinewidth)
ax.minorticks_on()
ax.tick_params('both', length=10, width=2, which='major')
ax.tick_params('both', length=5, width=1, which='minor')
plt.xticks(fontsize=myfontsize)
plt.yticks(fontsize=myfontsize)

plt.loglog()

plt.xlabel(r'Radius\,[kpc]',fontsize=myfontsize)
plt.ylabel(r'Density\,[${\rm M}_\odot$\,kpc$^{-3}$]',fontsize=myfontsize)

plt.plot(ranal,trueden,linewidth=mylinewidth,color='blue',\
    label='True density',alpha=0.5)
plt.plot(ranal,corenfwtides,linewidth=mylinewidth,color='green',\
    label='coreNFWtides',alpha=0.5)

#Show what happens as we change "n":
npnts = 9
nplot = np.linspace(-1.0,2.0,npnts)
jet = plt.get_cmap('jet')
cNorm  = colors.Normalize(vmin=np.min(nplot), vmax=np.max(nplot))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
for i in range(len(nplot)):
    colorVal = scalarMap.to_rgba(nplot[i])
    den = corenfw_tides_den(ranal,pfits[0],pfits[1],\
                pfits[2],nplot[i],\
                pfits[4],pfits[5])
    plt.plot(ranal,den,color=colorVal,alpha=0.5,\
             label=r'$n=%.1f$' % nplot[i])

plt.legend(fontsize=mylegendfontsize)
plt.xlim([1e-2,5])
plt.legend(loc='upper right',fontsize=18)
plt.savefig(output_base+'coreNFWtides_example.pdf',bbox_inches='tight')
