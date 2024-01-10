#Little program to test the mass model:
import disSat as dis
from functions import *
import numpy as np

# =============================================================================================
# CONSTANTS

PC          = 3.086e18        # in cm
KPC         = 1e3*PC          # in cm
MPC         = 1e6*PC          # in cm
KM          = 1e5             # in cm
KMS         = 1e5             # cm/s

TIME        = KPC/1e5         # seconds in a unit of Gadget's time
GYR         = 3600*24*365.25*1e9 # seconds in a Gyr
TIME_IN_GYR = TIME/GYR        # conversion from Gadget time units to Gyr
MSUN        = 1.9891e33       # in g
G           = 6.67e-8         # in cgs
pi = np.pi

M200 = 10.0**9.4
nsig_c200 = 0.0
tSF = 1.91756
zin = 3.0
mWDM = 50.0
sigmaSI = 5.8e-12
mleft = M200/2.0

cNFW_method='d15'
csig = dis.dark_matter.concentrations.Diemer19.scatter()
c200mean = dis.genutils.cNFW(M200,z=zin,method=cNFW_method,\
                wdm=True,mWDM=mWDM)
c200 = c200mean*10.0**(nsig_c200*csig)
rt = 1.0e5
delta = 3.0

ETA,KAPPA = 3.,0.04
fCORENFW = lambda x: (np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x))  # x = r/rc
rs,r200 = dis.genutils.nfw_r(M200,c200,z=zin,cNFW_method=cNFW_method)  

if  tSF==None: tSF = tin
tSF *= GYR
tDYN = 2*pi*np.sqrt((rs*KPC)**3/G/(dis.vutils.menc(rs,M200,'nfw',mleft=1,zin=zin,\
                                                   cNFW_method=cNFW_method,c200=c200,wdm=True,mWDM=mWDM)*\
                    MSUN))
q = KAPPA * tSF / tDYN
n = fCORENFW(q)
mstar = np.exp(dis.vutils.mstarD17(np.log(M200)))
Re0 = 10**(0.268*np.log10(mstar)-2.11)
rc = ETA * Re0

r = np.logspace(-2,5,250)
test = cosmo_profile_mass(r,M200,nsig_c200,tSF,mWDM,sigmaSI,\
                          mleft,zin)
test2 = corenfw_tides_mass(r,M200,c200,rc,n,rt,delta)

print(test)
print(test2)
print('TEST', c200)
