#Forbid plots to screen so Binulator can run
#remotely:
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.integrate import simps as integrator
from scipy.special import gamma
from constants import *
from functions import * 
from figures import *
import emcee
from multiprocessing import Pool
from multiprocessing import cpu_count
import sys

#Select velocity distribution function:
velpdfuse = velpdfdoubleback

#Functions for slicing the data:
def Rcutback(Rkin,vz,vzerr,mskin,Rfitvmin,Rfitvmax):
    if (Rfitvmin > 0):
        Rf_t = Rkin[Rkin > Rfitvmin]
        vzfit_t = vz[Rkin > Rfitvmin]
        vzerrfit_t = vzerr[Rkin > Rfitvmin]
        msfit_t = mskin[Rkin > Rfitvmin]
    else:
        Rf_t = Rkin
        vzfit_t = vz
        vzerrfit_t = vzerr
        msfit_t = mskin
    if (Rfitvmax > 0):
        Rf = Rf_t[Rf_t < Rfitvmax]
        vzfit = vzfit_t[Rf_t < Rfitvmax]
        vzerrfit = vzerrfit_t[Rf_t < Rfitvmax]
        msfit = msfit_t[Rf_t < Rfitvmax]
    else:
        Rf = Rf_t
        vzfit = vzfit_t
        vzerrfit = vzerrfit_t
        msfit = msfit_t
    return Rf, vzfit, vzerrfit, msfit

#Functions for emcee (velocity data):
def velfit_easy(R,vz,vzerr,ms,Nbin,\
                vfitmin,vfitmax,\
                p0vin_min,p0vin_max,p0best,\
                nsamples,outfile,nprocs):
    #Uses easy non-para estimates rather than VDF fit
    #**only use for rapid testing; not recommended
    #for real analysis**. Assumes only Poisson error
    #on each bin. Not really correct.
    
    rbin = np.zeros(len(R))
    right_bin_edge = np.zeros(len(R))
    vzmeanbin = np.zeros(len(R))
    vzmeanbinlo = np.zeros(len(R))
    vzmeanbinhi = np.zeros(len(R))
    vztwobin = np.zeros(len(R))
    vztwobinlo = np.zeros(len(R))
    vztwobinhi = np.zeros(len(R))
    vzfourbin = np.zeros(len(R))
    vzfourbinlo = np.zeros(len(R))
    vzfourbinhi = np.zeros(len(R))
    backampbin1 = np.zeros(len(R))
    backampbinlo1 = np.zeros(len(R))
    backampbinhi1 = np.zeros(len(R))
    backmeanbin1 = np.zeros(len(R))
    backmeanbinlo1 = np.zeros(len(R))
    backmeanbinhi1 = np.zeros(len(R))
    backsigbin1 = np.zeros(len(R))
    backsigbinlo1 = np.zeros(len(R))
    backsigbinhi1 = np.zeros(len(R))
    backampbin2 = np.zeros(len(R))
    backampbinlo2 = np.zeros(len(R))
    backampbinhi2 = np.zeros(len(R))
    backmeanbin2 = np.zeros(len(R))
    backmeanbinlo2 = np.zeros(len(R))
    backmeanbinhi2 = np.zeros(len(R))
    backsigbin2 = np.zeros(len(R))
    backsigbinlo2 = np.zeros(len(R))
    backsigbinhi2 = np.zeros(len(R))
    
    #This for storing the vzfour pdf
    #for calculating the VSPs:
    vzfour_pdf = np.zeros((nsamples,len(R)))
    
    #Loop through the bins, assuming Nbin stars
    #(weighted by ms) per bin:
    index = np.argsort(R)
    vzstore = np.zeros(len(R))
    vzerrstore = np.zeros(len(R))
    msstore = np.zeros(len(R))
    cnt = 0
    jsum = 0
    js = 0
    for i in range(len(R)):
        #Find stars in bin:
        if (jsum < Nbin):
            vzstore[js] = vz[index[i]]
            vzerrstore[js] = vzerr[index[i]]
            msstore[js] = ms[index[i]]
            right_bin_edge[cnt] = R[index[i]]
            jsum = jsum + ms[index[i]]
            js = js + 1
        if (jsum >= Nbin):
            #Non-para estimate of vel moments for these stars:
            vzuse = vzstore[:js]
            vzerruse = vzerrstore[:js]
            msuse = msstore[:js]
            
            #Cut back to fit range. If doing this,
            #perform an initial centre for this
            #bin first to ensure a symmetric
            #"haircut" on the data.
            if (vfitmin != 0 or vfitmax != 0):
                vzuse = vzuse - \
                        np.sum(vzuse*msuse)/np.sum(msuse)
            if (vfitmin != 0):
                vzuse_t = vzuse[vzuse > vfitmin]
                vzerruse_t = vzerruse[vzuse > vfitmin]
                msuse_t = msuse[vzuse > vfitmin]
            else:
                vzuse_t = vzuse
                vzerruse_t = vzerruse
                msuse_t = msuse
            if (vfitmax != 0):
                vzuse = vzuse_t[vzuse_t < vfitmax]
                vzerruse = vzerruse_t[vzuse_t < vfitmax]
                msuse = msuse_t[vzuse_t < vfitmax]                    
            else:
                vzuse = vzuse_t
                vzerruse = vzerruse_t
                msuse = msuse_t
                
            #Non-param estimates assuming Poisson errors:
            #**only use for testing; not recommended for real analysis**
            vztwobin[cnt] = \
                    (np.sum(vzuse**2.0*msuse)-np.sum(vzerruse**2.0*msuse))/np.sum(msuse)
            vztwobin[cnt] = np.sqrt(vztwobin[cnt])
            vztwobinlo[cnt] = vztwobin[cnt]-vztwobin[cnt]/np.sqrt(Nbin)
            vztwobinhi[cnt] = vztwobin[cnt]+vztwobin[cnt]/np.sqrt(Nbin)
            vzfourbin[cnt] = \
                    (np.sum(vzuse**4.0*msuse)-np.sum(3.0*vzerruse**4.0*msuse))/np.sum(msuse)
            vzfourbinlo[cnt] = vzfourbin[cnt]-vzfourbin[cnt]/np.sqrt(Nbin)
            vzfourbinhi[cnt] = vzfourbin[cnt]+vzfourbin[cnt]/np.sqrt(Nbin)

            #Assume vzfour distributed as Gaussian (again, for rapid testing only):
            vzfour_pdf[:,cnt] = np.random.normal(loc = vzfourbin[cnt],\
                                                 scale = vzfourbin[cnt]/np.sqrt(Nbin),\
                                                 size = nsamples)

            #Make a plot of the bin:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(mylinewidth)    
            ax.minorticks_on()
            ax.tick_params('both', length=10, width=2, which='major')
            ax.tick_params('both', length=5, width=1, which='minor')
            plt.xticks(fontsize=myfontsize)
            plt.yticks(fontsize=myfontsize)
            
            plt.xlabel(r'v\,[km/s]',fontsize=myfontsize)
            plt.ylabel(r'frequency',fontsize=myfontsize)

            n, bins, patches = plt.hist(vzuse,np.int(Nbin/10.0),weights=msuse,\
                                        facecolor='g',\
                                        alpha=0.75)
            plt.axvline(x=vztwobin[cnt],\
                        color='blue',linewidth=mylinewidth)
            plt.axvline(x=vztwobinlo[cnt],linestyle='dashed',\
                        color='blue',linewidth=mylinewidth)
            plt.axvline(x=vztwobinhi[cnt],linestyle='dashed',\
                        color='blue',linewidth=mylinewidth)
            
            vhistmax = np.max(np.abs(vzuse))
            plt.xlim([-vhistmax,vhistmax])
            plt.savefig(outfile+'hist_%d.pdf' % (cnt),\
                        bbox_inches='tight')
        
            #Calculate bin radius:
            if (cnt == 0):
                rbin[cnt] = right_bin_edge[cnt]/2.0
            else:
                rbin[cnt] = \
                    (right_bin_edge[cnt] + right_bin_edge[cnt-1])/2.0
            
            #Output the fit values (with non-para comparison):
            print('Bin: %d | radius: %f | vztwo %.2f+%.2f-%.2f | vzfour %.2f+%.2f-%.2f' \
                  % (cnt,rbin[cnt],\
                     vztwobin[cnt],\
                     vztwobinhi[cnt]-vztwobin[cnt],\
                     vztwobin[cnt]-vztwobinlo[cnt],\
                     vzfourbin[cnt],\
                     vzfourbinhi[cnt]-vzfourbin[cnt],\
                     vzfourbin[cnt]-vzfourbinlo[cnt]))

            #Move on to the next bin:
            jsum = 0.0
            js = 0
            cnt = cnt + 1

    #Cut back the output arrays:
    rbin = rbin[:cnt]
    vzmeanbin = vzmeanbin[:cnt]
    vzmeanbinlo = vzmeanbinlo[:cnt]
    vzmeanbinhi = vzmeanbinhi[:cnt]
    vztwobin = vztwobin[:cnt]
    vztwobinlo = vztwobinlo[:cnt]
    vztwobinhi = vztwobinhi[:cnt]
    vzfourbin = vzfourbin[:cnt]
    vzfourbinlo = vzfourbinlo[:cnt]
    vzfourbinhi = vzfourbinhi[:cnt]
    backampbin1 = backampbin1[:cnt]
    backampbinlo1 = backampbinlo1[:cnt]
    backampbinhi1 = backampbinhi1[:cnt]
    backmeanbin1 = backmeanbin1[:cnt]
    backmeanbinlo1 = backmeanbinlo1[:cnt]
    backmeanbinhi1 = backmeanbinhi1[:cnt]
    backsigbin1 = backsigbin1[:cnt]
    backsigbinlo1 = backsigbinlo1[:cnt]
    backsigbinhi1 = backsigbinhi1[:cnt]
    backampbin2 = backampbin2[:cnt]
    backampbinlo2 = backampbinlo2[:cnt]
    backampbinhi2 = backampbinhi2[:cnt]
    backmeanbin2 = backmeanbin2[:cnt]
    backmeanbinlo2 = backmeanbinlo2[:cnt]
    backmeanbinhi2 = backmeanbinhi2[:cnt]
    backsigbin2 = backsigbin2[:cnt]
    backsigbinlo2 = backsigbinlo2[:cnt]
    backsigbinhi2 = backsigbinhi2[:cnt]

    #Calculate the VSPs with uncertainties. This
    #assumes negligible error in the surface density
    #profile as compared to the velocity uncertainties.
    #This is usually fine, but something to bear in mind.
    ranal = np.logspace(-5,3,np.int(5e4))
    surfden = threeplumsurf(ranal,p0best[0],p0best[1],p0best[2],\
                            p0best[3],p0best[4],p0best[5])
        
    #This assumes a flat or linearly falling relation
    #beyond the last data point:
    vsp1 = np.zeros(nsamples)
    vsp2 = np.zeros(nsamples)
    vsp1_int = np.zeros(7)
    vsp2_int = np.zeros(7)
    vzfourstore = np.zeros((nsamples,len(ranal)))
    for i in range(nsamples):
        vzfour_thissample = vzfour_pdf[i,:cnt]
        vzfour = vzfourfunc(ranal,rbin,vzfour_thissample)
        vzfourstore[i,:] = vzfour
        vsp1[i] = integrator(surfden*vzfour*ranal,ranal)
        vsp2[i] = integrator(surfden*vzfour*ranal**3.0,ranal)
    vsp1_int[0], vsp1_int[1], vsp1_int[2], vsp1_int[3], \
        vsp1_int[4], vsp1_int[5], vsp1_int[6] = \
                                    calcmedquartnine(vsp1)
    vsp2_int[0], vsp2_int[1], vsp2_int[2], vsp2_int[3], \
        vsp2_int[4], vsp2_int[5], vsp2_int[6] = \
                                    calcmedquartnine(vsp2)
        
    return rbin,vzmeanbin,vzmeanbinlo,vzmeanbinhi,\
        vztwobin,vztwobinlo,vztwobinhi,\
        vzfourbin,vzfourbinlo,vzfourbinhi,\
        backampbin1,backampbinlo1,backampbinhi1,\
        backmeanbin1,backmeanbinlo1,backmeanbinhi1,\
        backsigbin1,backsigbinlo1,backsigbinhi1,\
        backampbin2,backampbinlo2,backampbinhi2,\
        backmeanbin2,backmeanbinlo2,backmeanbinhi2,\
        backsigbin2,backsigbinlo2,backsigbinhi2,\
        vsp1_int[0],vsp1_int[1],vsp1_int[2],\
        vsp2_int[0],vsp2_int[1],vsp2_int[2],\
        ranal,vzfourstore,vsp1,vsp2

def velfit_full(R,vz,vzerr,ms,Nbin,Nbinarr,\
                vfitmin,vfitmax,\
                p0vin_min,p0vin_max,p0best,\
                nsamples,outfile,nprocs):
    #Code to fit the velocity data with
    #the velpdf model.
    
    #Arrays to store binned:
    #radius (rbin),
    #<vlos> (vzmeanbin), 
    #<vlos^2>^1/2 (vztwobin),
    #<vlos^4> (vzfourbin),
    #and their confidence intervals; and
    #the mean and dispersion of the background
    #model.
    rbin = np.zeros(len(R))
    right_bin_edge = np.zeros(len(R))
    vzmeanbin = np.zeros(len(R))
    vzmeanbinlo = np.zeros(len(R))    
    vzmeanbinhi = np.zeros(len(R))    
    vztwobin = np.zeros(len(R))
    vztwobinlo = np.zeros(len(R))
    vztwobinhi = np.zeros(len(R))
    vzfourbin = np.zeros(len(R))
    vzfourbinlo = np.zeros(len(R))
    vzfourbinhi = np.zeros(len(R))
    backampbin1 = np.zeros(len(R))
    backampbinlo1 = np.zeros(len(R))
    backampbinhi1 = np.zeros(len(R))
    backmeanbin1 = np.zeros(len(R))
    backmeanbinlo1 = np.zeros(len(R))
    backmeanbinhi1 = np.zeros(len(R))
    backsigbin1 = np.zeros(len(R))
    backsigbinlo1 = np.zeros(len(R))
    backsigbinhi1 = np.zeros(len(R))
    backampbin2 = np.zeros(len(R))
    backampbinlo2 = np.zeros(len(R))
    backampbinhi2 = np.zeros(len(R))
    backmeanbin2 = np.zeros(len(R))
    backmeanbinlo2 = np.zeros(len(R))
    backmeanbinhi2 = np.zeros(len(R))
    backsigbin2 = np.zeros(len(R))
    backsigbinlo2 = np.zeros(len(R))
    backsigbinhi2 = np.zeros(len(R))
    
    #This for storing the vzfour pdf
    #for calculating the VSPs:
    vzfour_pdf = np.zeros((nsamples,len(R)))

    #Loop through the bins, assuming Nbin stars
    #(weighted by ms) per bin:
    index = np.argsort(R)
    vzstore = np.zeros(len(R))
    vzerrstore = np.zeros(len(R))
    msstore = np.zeros(len(R))
    cnt = 0
    jsum = 0
    js = 0

    for i in range(len(R)):
        #For manual bins:
        if (np.sum(Nbinarr) > 0):
            if (cnt < len(Nbinarr)):
                Nbin = Nbinarr[cnt]
            else:
                Nbin = Nbinarr[::-1][0]

        #Find stars in bin:
        if (jsum < Nbin):
            vzstore[js] = vz[index[i]]
            vzerrstore[js] = vzerr[index[i]]
            msstore[js] = ms[index[i]]
            right_bin_edge[cnt] = R[index[i]]
            jsum = jsum + ms[index[i]]
            js = js + 1
        if (jsum >= Nbin):
            #Fit the velpdf model to these stars:
            vzuse = vzstore[:js]
            vzerruse = vzerrstore[:js]
            msuse = msstore[:js]
            
            #Cut back to fit range. If doing this,
            #perform an initial centre for this
            #bin first to ensure a symmetric
            #"haircut" on the data.
            if (vfitmin != 0 or vfitmax != 0):
                vzuse = vzuse - \
                    np.sum(vzuse*msuse)/np.sum(msuse)
            if (vfitmin != 0):
                vzuse_t = vzuse[vzuse > vfitmin]
                vzerruse_t = vzerruse[vzuse > vfitmin]
                msuse_t = msuse[vzuse > vfitmin]
            else:
                vzuse_t = vzuse
                vzerruse_t = vzerruse
                msuse_t = msuse
            if (vfitmax != 0):
                vzuse = vzuse_t[vzuse_t < vfitmax]
                vzerruse = vzerruse_t[vzuse_t < vfitmax]
                msuse = msuse_t[vzuse_t < vfitmax]
            else:
                vzuse = vzuse_t
                vzerruse = vzerruse_t
                msuse = msuse_t

            vzmeanbin[cnt],vzmeanbinlo[cnt],vzmeanbinhi[cnt],\
            vztwobin[cnt],vztwobinlo[cnt],vztwobinhi[cnt],\
            vzfourbin[cnt],vzfourbinlo[cnt],vzfourbinhi[cnt],\
            backampbin1[cnt],backampbinlo1[cnt],backampbinhi1[cnt],\
            backmeanbin1[cnt],backmeanbinlo1[cnt],backmeanbinhi1[cnt],\
            backsigbin1[cnt],backsigbinlo1[cnt],backsigbinhi1[cnt],\
            backampbin2[cnt],backampbinlo2[cnt],backampbinhi2[cnt],\
            backmeanbin2[cnt],backmeanbinlo2[cnt],backmeanbinhi2[cnt],\
            backsigbin2[cnt],backsigbinlo2[cnt],backsigbinhi2[cnt],\
            vzfour_store,p0vbest = \
                velfitbin(vzuse,vzerruse,msuse,\
                          p0vin_min,p0vin_max,nsamples,nprocs)
            vzfour_pdf[:,cnt] = vzfour_store

            #Calculate non-para versions for comparison:
            vztwo_nonpara = \
                (np.sum(vzuse**2.0*msuse)-np.sum(vzerruse**2.0*msuse))/np.sum(msuse)
            vztwo_nonpara = np.sqrt(vztwo_nonpara)
            vzfour_nonpara = \
                (np.sum(vzuse**4.0*msuse)-np.sum(3.0*vzerruse**4.0*msuse))/np.sum(msuse)
            
            #Make a plot of the fit:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(mylinewidth)

            ax.minorticks_on()
            ax.tick_params('both', length=10, width=2, which='major')
            ax.tick_params('both', length=5, width=1, which='minor')
            plt.xticks(fontsize=myfontsize)
            plt.yticks(fontsize=myfontsize)

            plt.xlabel(r'v\,[km/s]',fontsize=myfontsize)
            plt.ylabel(r'frequency',\
                fontsize=myfontsize)

            vhistmax = np.max(np.abs(vzuse))
            nbins_use = np.int(Nbin/5.0)
            if (nbins_use < 10):
                nbins_use = 10
            if (nbins_use > 25):
                nbins_use = 25
            n, bins, patches = plt.hist(vzuse,nbins_use,weights=msuse,\
                                        facecolor='g',\
                                        alpha=0.75)
            vplot = np.linspace(-vhistmax,vhistmax,np.int(500))
            vperr = np.zeros(len(vplot))+\
                np.sum(vzerruse*msuse)/np.sum(msuse)
            pdf = velpdfuse(vplot,vperr,p0vbest)
            plt.plot(vplot,pdf/np.max(pdf)*np.max(n),\
                     linewidth=mylinewidth,color='red')

            #Overplot just background histogram:
            backampplt = p0vbest[3]
            backmeanplt = np.abs(p0vbest[4])
            backsigplt = np.sqrt(p0vbest[5]**2.0 + vperr**2.0)
            backpdfplt = backampplt/(np.sqrt(2.0*np.pi)*backsigplt)*\
                np.exp(-0.5*(vplot-backmeanplt)**2.0/backsigplt**2.0)
            plt.plot(vplot,backpdfplt/np.max(pdf)*np.max(n),linewidth=mylinewidth,\
                     linestyle='dashed',color='blue')
            backampplt = p0vbest[6]
            backmeanplt = -np.abs(p0vbest[7])
            backsigplt = np.sqrt(p0vbest[8]**2.0 + vperr**2.0)
            backpdfplt = backampplt/(np.sqrt(2.0*np.pi)*backsigplt)*\
                         np.exp(-0.5*(vplot-backmeanplt)**2.0/backsigplt**2.0)
            plt.plot(vplot,backpdfplt/np.max(pdf)*np.max(n),linewidth=mylinewidth,\
                     linestyle='dashed',color='blue')
            
            plt.xlim([-vhistmax,vhistmax])
            plt.savefig(outfile+'hist_%d.png' % (cnt),\
                        bbox_inches='tight',dpi=300)

            #Calculate bin radius:
            if (cnt == 0):
                rbin[cnt] = right_bin_edge[cnt]/2.0
            else:
                rbin[cnt] = \
                    (right_bin_edge[cnt] + right_bin_edge[cnt-1])/2.0
        
            #Output the fit values (with non-para comparison):
            print('Bin: %d | Nbin: %f | radius: %f | vztwo %.2f(%.2f)+%.2f-%.2f | vzfour %.2f(%.2f)+%.2f-%.2f' \
                  % (cnt,Nbin,rbin[cnt],\
                     vztwobin[cnt],vztwo_nonpara,\
                     vztwobinhi[cnt]-vztwobin[cnt],\
                     vztwobin[cnt]-vztwobinlo[cnt],\
                     vzfourbin[cnt],vzfour_nonpara,\
                     vzfourbinhi[cnt]-vzfourbin[cnt],\
                     vzfourbin[cnt]-vzfourbinlo[cnt]),\
            )

            #Move on to the next bin:
            jsum = 0.0
            js = 0
            cnt = cnt + 1                                    
            
    #Cut back the output arrays:
    rbin = rbin[:cnt]
    vzmeanbin = vzmeanbin[:cnt]
    vzmeanbinlo = vzmeanbinlo[:cnt]
    vzmeanbinhi = vzmeanbinhi[:cnt]
    vztwobin = vztwobin[:cnt]
    vztwobinlo = vztwobinlo[:cnt]
    vztwobinhi = vztwobinhi[:cnt]
    vzfourbin = vzfourbin[:cnt]
    vzfourbinlo = vzfourbinlo[:cnt]
    vzfourbinhi = vzfourbinhi[:cnt]
    backampbin1 = backampbin1[:cnt]
    backampbinlo1 = backampbinlo1[:cnt]
    backampbinhi1 = backampbinhi1[:cnt]
    backmeanbin1 = backmeanbin1[:cnt]
    backmeanbinlo1 = backmeanbinlo1[:cnt]
    backmeanbinhi1 = backmeanbinhi1[:cnt]
    backsigbin1 = backsigbin1[:cnt]
    backsigbinlo1 = backsigbinlo1[:cnt]
    backsigbinhi1 = backsigbinhi1[:cnt]
    backampbin2 = backampbin2[:cnt]
    backampbinlo2 = backampbinlo2[:cnt]
    backampbinhi2 = backampbinhi2[:cnt]
    backmeanbin2 = backmeanbin2[:cnt]
    backmeanbinlo2 = backmeanbinlo2[:cnt]
    backmeanbinhi2 = backmeanbinhi2[:cnt]
    backsigbin2 = backsigbin2[:cnt]
    backsigbinlo2 = backsigbinlo2[:cnt]
    backsigbinhi2 = backsigbinhi2[:cnt]                                
    
    #Calculate the VSPs with uncertainties. This
    #assumes negligible error in the surface density 
    #profile as compared to the velocity uncertainties.
    #This is usually fine, but something to bear in mind.
    ranal = np.logspace(-5,3,np.int(5e4))
    surfden = threeplumsurf(ranal,p0best[0],p0best[1],p0best[2],\
                            p0best[3],p0best[4],p0best[5])

    #This assumes a flat or linearly falling relation
    #beyond the last data point:
    vsp1 = np.zeros(nsamples)
    vsp2 = np.zeros(nsamples)
    vsp1_int = np.zeros(7)
    vsp2_int = np.zeros(7)
    vzfourstore = np.zeros((nsamples,len(ranal)))
    for i in range(nsamples):
        vzfour_thissample = vzfour_pdf[i,:cnt]
        vzfour = vzfourfunc(ranal,rbin,vzfour_thissample)
        vzfourstore[i,:] = vzfour
        vsp1[i] = integrator(surfden*vzfour*ranal,ranal)
        vsp2[i] = integrator(surfden*vzfour*ranal**3.0,ranal)
    vsp1_int[0], vsp1_int[1], vsp1_int[2], vsp1_int[3], \
        vsp1_int[4], vsp1_int[5], vsp1_int[6] = \
        calcmedquartnine(vsp1)
    vsp2_int[0], vsp2_int[1], vsp2_int[2], vsp2_int[3], \
        vsp2_int[4], vsp2_int[5], vsp2_int[6] = \
        calcmedquartnine(vsp2)
    
    return rbin,vzmeanbin,vzmeanbinlo,vzmeanbinhi,\
        vztwobin,vztwobinlo,vztwobinhi,\
        vzfourbin,vzfourbinlo,vzfourbinhi,\
        backampbin1,backampbinlo1,backampbinhi1,\
        backmeanbin1,backmeanbinlo1,backmeanbinhi1,\
        backsigbin1,backsigbinlo1,backsigbinhi1,\
        backampbin2,backampbinlo2,backampbinhi2,\
        backmeanbin2,backmeanbinlo2,backmeanbinhi2,\
        backsigbin2,backsigbinlo2,backsigbinhi2,\
        vsp1_int[0],vsp1_int[1],vsp1_int[2],\
        vsp2_int[0],vsp2_int[1],vsp2_int[2],\
        ranal,vzfourstore,vsp1,vsp2

#Functions:
def lnprob_vel(theta, y, yerr, ms, p0vin_min, p0vin_max):
    lp = lnprior_set_vel(theta,p0vin_min,p0vin_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_vel(theta, y, yerr, ms)

def lnlike_vel(theta, y, yerr, ms):
    modelpdf = velpdfuse(y,yerr,theta)
    lnlike_out = np.sum(np.log(modelpdf)*ms)
    
    if (lnlike_out != lnlike_out):
        lnlike_out = -np.inf
    
    return lnlike_out
    
def lnprior_set_vel(theta,p0vin_min,p0vin_max):
    ndims = len(theta)
    minarr = np.zeros(ndims)
    maxarr = np.zeros(ndims)
    for i in range(ndims):
        minarr[i] = p0vin_min[i]
        maxarr[i] = p0vin_max[i]
       
    if all(minarr < thetau < maxarr for minarr,thetau,maxarr in \
           zip(minarr,theta,maxarr)):
        return 0.0
    return -np.inf

def velfitbin(vz,vzerr,ms,p0vin_min,p0vin_max,nsamples,nprocs):
    #Fit the model velocity pdf to a single bin:

    #Emcee parameters:
    nwalkers = 500
    nmodels = 10000

    #Starting guess
    ndims = len(p0vin_min)
    pos = np.zeros((nwalkers, ndims), dtype='float')
    p0vin_startmin = p0vin_min
    p0vin_startmax = p0vin_max    
    for i in range(ndims):
        pos[:,i] = np.random.uniform(p0vin_startmin[i],\
            p0vin_startmax[i],nwalkers)

    #Run chains:
    with Pool(processes = nprocs) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndims, lnprob_vel, \
                    args=(vz, vzerr, ms, p0vin_min, p0vin_max), \
                    pool=pool)
        sampler.run_mcmc(pos, nmodels, progress = False)

    #Extract results + errors:
    burn = np.int(0.75*nmodels)
    chisq = -2.0 * \
            sampler.get_log_prob(discard=burn, flat=True)
    par_test = sampler.get_chain(discard=burn, flat=True)

    #Store best fit model:
    index = np.argsort(chisq)
    p0best = par_test[index[0],:]

    #Choose number of models to draw from the chains:
    min_chisq = np.min(chisq)
    index = np.where(chisq < min_chisq*500.0)[0]
    sample_choose = index[np.random.randint(len(index), \
                          size=nsamples)]

    #Set up arrays to store med,68%,95%,99% confidence intervals:
    vzmean_int = np.zeros(7)
    vzmean_store = np.zeros(nsamples)
    vztwo_int = np.zeros(7)
    vztwo_store = np.zeros(nsamples)
    vzfour_int = np.zeros(7)
    vzfour_store = np.zeros(nsamples)
    backamp1_int = np.zeros(7)
    backamp1_store = np.zeros(nsamples) 
    backmean1_int = np.zeros(7)
    backmean1_store = np.zeros(nsamples)
    backsig1_int = np.zeros(7)
    backsig1_store = np.zeros(nsamples)
    backamp2_int = np.zeros(7)
    backamp2_store = np.zeros(nsamples)
    backmean2_int = np.zeros(7)
    backmean2_store = np.zeros(nsamples)
    backsig2_int = np.zeros(7)
    backsig2_store = np.zeros(nsamples)                    

    for i in range(nsamples):
        theta = par_test[sample_choose[i],:]
        vzmean_store[i] = theta[0]
        vztwo_store[i] = vztwo_calc(theta)
        vzfour_store[i] = vzfour_calc(theta)
        backamp1_store[i] = theta[3]
        backmean1_store[i] = theta[4]
        backsig1_store[i] = theta[5]
        backamp2_store[i] = theta[6]
        backmean2_store[i] = theta[7]
        backsig2_store[i] = theta[8]

    #Solve for confidence intervals:
    vzmean_int[0], vzmean_int[1], vzmean_int[2], vzmean_int[3], \
        vzmean_int[4], vzmean_int[5], vzmean_int[6] = \
        calcmedquartnine(vzmean_store)
    vztwo_int[0], vztwo_int[1], vztwo_int[2], vztwo_int[3], \
        vztwo_int[4], vztwo_int[5], vztwo_int[6] = \
        calcmedquartnine(vztwo_store)
    vzfour_int[0], vzfour_int[1], vzfour_int[2], vzfour_int[3], \
        vzfour_int[4], vzfour_int[5], vzfour_int[6] = \
        calcmedquartnine(vzfour_store)
    backamp1_int[0], backamp1_int[1], backamp1_int[2], backamp1_int[3], \
        backamp1_int[4], backamp1_int[5], backamp1_int[6] = \
        calcmedquartnine(backamp1_store)
    backmean1_int[0], backmean1_int[1], backmean1_int[2], backmean1_int[3], \
        backmean1_int[4], backmean1_int[5], backmean1_int[6] = \
        calcmedquartnine(backmean1_store)
    backsig1_int[0], backsig1_int[1], backsig1_int[2], backsig1_int[3], \
        backsig1_int[4], backsig1_int[5], backsig1_int[6] = \
        calcmedquartnine(backsig1_store)
    backamp2_int[0], backamp2_int[1], backamp2_int[2], backamp2_int[3], \
        backamp2_int[4], backamp2_int[5], backamp2_int[6] = \
        calcmedquartnine(backamp2_store)
    backmean2_int[0], backmean2_int[1], backmean2_int[2], backmean2_int[3], \
        backmean2_int[4], backmean2_int[5], backmean2_int[6] = \
        calcmedquartnine(backmean2_store)
    backsig2_int[0], backsig2_int[1], backsig2_int[2], backsig2_int[3], \
        backsig2_int[4], backsig2_int[5], backsig2_int[6] = \
        calcmedquartnine(backsig2_store)

    #Output median and 68% confidence intervals.
    #Pass back full vzfour distribution for VSP 
    #calculation.
    return vzmean_int[0],vzmean_int[1],vzmean_int[2],\
           vztwo_int[0],vztwo_int[1],vztwo_int[2],\
           vzfour_int[0],vzfour_int[1],vzfour_int[2],\
           backamp1_int[0],backamp1_int[1],backamp1_int[2],\
           backmean1_int[0],backmean1_int[1],backmean1_int[2],\
           backsig1_int[0],backsig1_int[1],backsig1_int[2],\
           backamp2_int[0],backamp2_int[1],backamp2_int[2],\
           backmean2_int[0],backmean2_int[1],backmean2_int[2],\
           backsig2_int[0],backsig2_int[1],backsig2_int[2],\
           vzfour_store,p0best
