import numpy as np
from scipy.integrate import simps as integrator
from scipy.special import gamma
from constants import *
from functions import * 
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count

#Functions:
def lnprob_surf(theta, x, y, yerr, p0in_min, p0in_max):
    lp = lnprior_set_surf(theta,p0in_min,p0in_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_surf(theta, x, y, yerr)

def lnlike_surf(theta, x, y, yerr):
    thetau = theta
    thetau[0] = 10.0**theta[0]
    thetau[1] = 10.0**theta[1]
    thetau[2] = 10.0**theta[2]
    thetau[3] = 10.0**theta[3]
    model = doublealpbetgamsurf_fit(x,thetau[0],thetau[1],thetau[2],thetau[3],\
                                    thetau[4],thetau[5],thetau[6],thetau[7],\
                                    thetau[8],thetau[9])
    inv_sigma2 = 1.0/(yerr)**2.0
    lnlike_out = -0.5*(np.sum((y-model)**2*inv_sigma2))
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf
    return lnlike_out

def lnprior_set_surf(theta,p0in_min,p0in_max):
    ndims = len(theta)
    minarr = np.zeros(ndims)
    maxarr = np.zeros(ndims)
    for i in range(ndims):
        minarr[i] = p0in_min[i]
        maxarr[i] = p0in_max[i]
    if all(minarr < thetau < maxarr for \
           minarr,thetau,maxarr in \
           zip(minarr,theta,maxarr)):
        return 0.0
    return -np.inf

#Functions for emcee (surface density):
def tracerfit(R,surfden,surfdenerr,p0in_min,p0in_max,nprocs):
    #Code to fit the surface density profile with
    #a double alp-bet-gam model:

    #Emcee parameters:
    nwalkers = 250
    nmodels = 2500
    
    #Starting guess:
    ndims = len(p0in_min)
    pos = np.zeros((nwalkers, ndims), dtype='float')

    for i in range(ndims):
        poslow, poshigh = blobcalc(p0in_min[i],p0in_max[i])
        pos[:,i] = np.random.uniform(poslow,poshigh,nwalkers)

#    #TEST start model:
#    import pylab as plt
#    plt.figure()
#    plt.loglog()
#    plt.errorbar(R,surfden,surfdenerr)
#    theta = pos[0,:]
#    thetau = theta
#    thetau[0] = 10.0**theta[0]
#    thetau[1] = 10.0**theta[1]
#    thetau[2] = 10.0**theta[2]
#    thetau[3] = 10.0**theta[3]
#    rho01,rho02,r01,r02,alp1,alp2,bet1,bet2,gam1,gam2 = \
#        thetau
#    r = R
#    intpnts = np.int(1e2)
#    theta = np.linspace(0,np.pi/2.0-1.0e-30,num=intpnts)
#    cth = np.cos(theta)
#    cth2 = cth**2
#    surf = np.zeros(len(r), 'double')
#    for i in range(len(r)):
#        q = r[i]/cth
#        y = doublealpbetgamden(q,rho01,rho02,r01,r02,alp1,alp2,\
#                               bet1,bet2,gam1,gam2)
#        surf[i] = 2.0*r[i]*integrator(y/cth2,theta)
#    
#    testmodel = doublealpbetgamsurf(R,thetau[0],thetau[1],thetau[2],thetau[3],\
#                                    thetau[4],thetau[5],thetau[6],thetau[7],\
#                                    thetau[8],thetau[9],np.int(1e2))
#    plt.plot(R,testmodel)
#    plt.plot(R,surf)
#    plt.xlim([1e-6,5])
#    plt.ylim([1e-3,1e5])
#    plt.savefig('test.png')
#    print(thetau)
#    print(testmodel)
#    print(R)
#    print(surf) 
#    sys.exit(0)
    
    #Run chains:
    with Pool(processes = nprocs) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndims, \
                                        lnprob_surf, \
                                        args=(R, surfden, surfdenerr, \
                                              p0in_min, p0in_max),pool=pool)
        sampler.run_mcmc(pos, nmodels, progress = True)
        
    #Extract results + errors:
    burn = np.int(0.75*nmodels)
    chisq = -2.0 * \
            sampler.get_log_prob(discard=burn, flat=True)
    par_test = sampler.get_chain(discard=burn, flat=True)
    
    #Store best fit model:
    index = np.argsort(chisq)
    p0best = par_test[index[0],:]
    p0best[0] = 10.0**p0best[0]
    p0best[1] = 10.0**p0best[1]
    p0best[2] = 10.0**p0best[2]
    p0best[3] = 10.0**p0best[3]
    
    #Choose number of models to draw from the chains:
    nsamples = 1000
    sample_choose = np.random.randint(len(chisq), \
                                      size=nsamples)
    
    #Set up arrays to store med,68%,95%,99% confidence intervals:
    Rpmin = np.min(R)/50.0
    Rpmax = np.max(R)*50.0
    Rplot = np.logspace(np.log10(Rpmin),np.log10(Rpmax),1000)
    surf_int = np.zeros((7,len(Rplot)))
    surf_store = np.zeros((len(Rplot),nsamples))
    Rhalf_int = np.zeros(7)
    Rhalf_store = np.zeros(nsamples)
    
    for i in range(nsamples):
        theta = par_test[sample_choose[i],:]
        thetau = theta
        thetau[0] = 10.0**theta[0]
        thetau[1] = 10.0**theta[1]
        thetau[2] = 10.0**theta[2]
        thetau[3] = 10.0**theta[3]                     
        surf = doublealpbetgamsurf_fit(Rplot,thetau[0],thetau[1],thetau[2],\
                                       thetau[3],thetau[4],thetau[5],thetau[6],\
                                       thetau[7],thetau[8],thetau[9])
        Rhalf = Rhalf_func_doublealpbetgam(thetau[0],thetau[1],thetau[2],\
                                           thetau[3],thetau[4],thetau[5],thetau[6],\
                                           thetau[7],thetau[8],thetau[9])
        surf_store[:,i] = surf
        Rhalf_store[i] = Rhalf
    
    #Solve for confidence intervals:
    for j in range(len(Rplot)):
        surf_int[0,j], surf_int[1,j], surf_int[2,j], \
            surf_int[3,j], \
            surf_int[4,j], surf_int[5,j], surf_int[6,j] = \
            calcmedquartnine(surf_store[j,:])
    Rhalf_int[0], Rhalf_int[1], Rhalf_int[2], Rhalf_int[3], \
        Rhalf_int[4], Rhalf_int[5], Rhalf_int[6] = \
        calcmedquartnine(Rhalf_store)

    return Rplot,surf_int,Rhalf_int,p0best
