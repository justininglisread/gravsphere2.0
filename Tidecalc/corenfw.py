'''
   #------------------------------------------------------------------------
   # corenfw.py | version 0.0 | Justin Read 2018 
   #------------------------------------------------------------------------

   Corenfw mass model:
    - pars[0] = M200
    - pars[1] = c200
    - pars[2] = n
    - pars[3] = rc
    - pars[4] = rhocrit
    - pars[5] = G
    - pars[6] = intpnts (for projection)
'''

from __future__ import division
import numpy as np
import scipy.integrate as si
from Tidecalc.numfuncs import numsurf
import sys

#-----------------------------------------------------------------------------
# Functions
#-----------------------------------------------------------------------------

#Global vars:
oden = 200.0

def cosmo_cfunc(M200,h):
    c = 10.0**(0.905 - 0.101*np.log10(M200/(1e12/h)))
    return c

def cosmo_Rhalf(M200,c,rhocrit):
    #Calculate Rhalf a la Kravtsov et al. 2013:
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    Rhalf = 0.015*rv
    return Rhalf

def cosmo_r200(M200,c,rhocrit):
    #Calculate r200:
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    return rv

def cosmo_rs(M200,c,rhocrit):
    #Calculate rs:
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c
    return rs

def den(r,pars):    
    '''Calculates density'''
    M200 = pars[0]
    c = pars[1]
    n = pars[2]
    rc = pars[3]
    rhocrit = pars[4]

    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c
    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    if (n > 0):
        x = r/rc
        f = np.tanh(x)
        my_manal = manal*f**n
        my_rhoanal = rhoanal*f**n + \
            1.0/(4.*np.pi*r**2.*rc)*manal*(1.0-f**2.)*n*f**(n-1.0)
    else:
        my_rhoanal = rhoanal

    return my_rhoanal

def denrt(r,pars,rt):
    rho = den(r,pars)
    rho[r > rt] = 0.0
    return rho

def surf(r,pars):
    '''Calculates surface density'''
    return numsurf(r,pars,den)

def cummass(r,pars):    
    '''Calculates cumulative mass'''
    M200 = pars[0]
    c = pars[1]
    n = pars[2]
    rc = pars[3]
    rhocrit = pars[4]

    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c
    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    x = r/rc
    f = np.tanh(x)
    my_manal = manal*f**n

    return my_manal

def pot(r,pars):    
    '''Calculates gravitational potential'''
    M200 = pars[0]
    c = pars[1]
    n = pars[2]
    rc = pars[3]
    G = pars[5]
    print('Potential not implemented ... ')
    return np.zeros(len(r))

def fr(r,pars):    
    '''Calculates radial force'''
    G = pars[5]
    return -G*cummass(r,pars)/r**2.0

#-----------------------------------------------------------------------------
# Test the functions. This runs if the python module is directly called.
#-----------------------------------------------------------------------------
if __name__ == "__main__":

    #Import plots library:
    from pylab import loglog, plot, legend, show, figure
    from numpy import linspace
    
    rmin = 0.01
    rmax = 10
    pnts = 1000
    r = linspace(rmin,rmax,num=pnts)
    G = 6.67e-11
    Msun = 1.989e30
    kpc = 3.086e19
    rhocrit = 135.05
    M200 = 1.0e11
    h = 0.7
    nsig = 5.0
    c200 = 10.0**(np.log10(cosmo_cfunc(M200,h))-nsig*0.1)
    Rhalf = 5.0
    print('M200:', M200)
    print('c200:', c200)
    print('Rhalf:', Rhalf)
    pars = [M200,c200,1.0,Rhalf*1.75,\
            rhocrit,G,1000]
    distance = 23.0
    r_max = np.array([0.25,1.0e8])
    surface = surf(r,pars)
    density = den(r,pars)
    density_rt = denrt(r,pars,1.0)
    mass = cummass(r,pars)
    pot = pot(r,pars)
    fr = fr(r,pars)
    fr2 = -G*mass/r**2

    figure()
    loglog(r,density,label='density')
    plot(r,density_rt,label='density_rt')
    plot(r,surface,label='surface density')
    legend()

    figure()
    loglog(r,mass,label='cumulative mass')
    legend()

    figure()
    plot(r,pot,label='potential')
    legend() 

    figure()
    plot(r,np.sqrt(G*mass*Msun/(r*kpc))/1000.0,label='rotation curve')
    legend()

    figure()
    plot(r,fr,label='radial force')
    plot(r,fr2,label='G*M(r)/r**2')
    legend()
 
    show()
