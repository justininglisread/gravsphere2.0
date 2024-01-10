import numpy as np
from scipy.integrate import simps as integrator
from scipy.misc.common import derivative
from scipy.special import gamma
from scipy.integrate import quad, dblquad
from constants import *
import disSat as dis
multimode = 'normal'

###########################################################
#For setting cosmology priors on mass model parameters.
def cosmo_cfunc(M200,h):
    #From Dutton & Maccio 2014. Requires as input masses 
    #defined in 200c system in units of Msun:
    c = 10.**(0.905 - 0.101 * (np.log10(M200*h)-12.))
    return c
    
def cosmo_cfunc_WDM(M200,h,OmegaM,rhocrit,mWDM):
    #Use formula in https://arxiv.org/pdf/1112.0330.pdf
    #to modify CDM M200-c200 relation to the WDM 
    #one. Assumes mWDM in keV, dimensionless h
    #M200 in Msun and rhocrit in Msun kpc^-3.
    cCDM = cosmo_cfunc(M200,h)
    gamma1 = 15.0
    gamma2 = 0.3
    lamfseff = 0.049*(mWDM)**(-1.11)*\
        (OmegaM/0.25)**(0.11)*(h/0.7)**(1.22)*1000.0
    lamhm = 13.93*lamfseff
    Mhm = 4.0/3.0*np.pi*rhocrit*(lamhm/2.0)**3.0
    cWDM = cCDM * (1.0 + gamma1*Mhm / M200)**(-gamma2)
    return cWDM

def dadt(axp_t,O_mat_0,O_vac_0,O_k_0):
    dadt = (1.0/axp_t) * \
        ( O_mat_0 + \
              O_vac_0 * axp_t*axp_t*axp_t + \
              O_k_0   * axp_t )
    dadt = np.sqrt(dadt)
    return dadt

def dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0):
    dadtau = axp_tau*axp_tau*axp_tau * \
        ( O_mat_0 + \
              O_vac_0 * axp_tau*axp_tau*axp_tau + \
              O_k_0   * axp_tau )
    dadtau = np.sqrt(dadtau)

    return dadtau

def friedmann(O_mat_0,O_vac_0,O_k_0,alpha,axp_min):
    # This subroutine assumes that axp = 1 at z = 0 (today)
    # and that t and tau = 0 at z = 0 (today).
    # axp is the expansion factor, hexp the Hubble constant
    # defined as hexp=1/axp*daxp/dtau, tau the conformal
    # time, and t the look-back time, both in unit of 1/H0.
    # alpha is the required accuracy and axp_min is the
    # starting expansion factor of the look-up table.
    # ntable is the required size of the look-up table.

    #Set up variables:
    axp_tau = 1.0
    axp_t = 1.0
    tau = 0.0
    t = 0.0
    nstep = 0

    while (axp_tau > axp_min or axp_t >= axp_min):
        dtau = alpha * axp_tau / dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)
        axp_tau_pre = axp_tau - dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)*dtau/2.0
        axp_tau = axp_tau - dadtau(axp_tau_pre,O_mat_0,O_vac_0,O_k_0)*dtau
        tau = tau - dtau
        dt = alpha * axp_t / dadt(axp_t,O_mat_0,O_vac_0,O_k_0)
        axp_t_pre = axp_t - dadt(axp_t,O_mat_0,O_vac_0,O_k_0)*dt/2.0
        axp_t = axp_t - dadt(axp_t_pre,O_mat_0,O_vac_0,O_k_0)*dt
        t = t - dt
        nstep = nstep + 1

    age_tot=-t
    
    #Set ntable = nstep and set up output arrays accordingly:
    ntable = nstep
    axp_out = np.zeros(ntable)
    hexp_out = np.zeros(ntable)
    tau_out = np.zeros(ntable)
    t_out = np.zeros(ntable)
    nskip=nstep/ntable

    axp_t = 1.0
    t = 0.0
    axp_tau = 1.0
    tau = 0.0
    nstep = 0
    nout = 0
    t_out[nout]=t
    tau_out[nout]=tau
    axp_out[nout]=axp_tau
    hexp_out[nout]=dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)/axp_tau
  
    while (axp_tau >= axp_min or axp_t >= axp_min):
        dtau = alpha * axp_tau / dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)
        axp_tau_pre = axp_tau - dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)*dtau/2.0
        axp_tau = axp_tau - dadtau(axp_tau_pre,O_mat_0,O_vac_0,O_k_0)*dtau
        tau = tau - dtau
        dt = alpha * axp_t / dadt(axp_t,O_mat_0,O_vac_0,O_k_0)
        axp_t_pre = axp_t - dadt(axp_t,O_mat_0,O_vac_0,O_k_0)*dt/2.0
        axp_t = axp_t - dadt(axp_t_pre,O_mat_0,O_vac_0,O_k_0)*dt
        t = t - dt

        if(np.mod(nstep,nskip)==0):
            t_out[nout]=t
            tau_out[nout]=tau
            axp_out[nout]=axp_tau
            hexp_out[nout]=dadtau(axp_tau,O_mat_0,O_vac_0,O_k_0)/axp_tau
            nout = nout + 1
        nstep = nstep + 1

    return t_out, tau_out, axp_out, hexp_out

def time_from_redshift(z,Omegam,OmegaL,h):
    #Convert redshift, z, to physical time given
    #cosmological parameters. Output in Gyrs.
    Hub = h * 100. * kms / Mpc
    invHub_Gyr = 1.0/Hub/Gyr
    t_out, tau_out, axp_out, hexp_out = \
        friedmann(Omegam,OmegaL,0.0,1e-4,1e-3)
    t_out = -t_out*invHub_Gyr*1.0e9
    index = np.argsort(axp_out)
    a = 1.0/(1.0+z)
    t = np.interp(a,axp_out[index],np.max(t_out)-t_out[index])
    return t/1.0e9

def redshift_from_time(t,Omegam,OmegaL,h):
    #This calculates the inverse of the
    #time_from_redshift routine. Input
    #time in Gyrs.
    zarr = np.logspace(-1,4,np.int(1e4))
    tarr = time_from_redshift(zarr,Omegam,OmegaL,h)
    index = np.argsort(tarr)
    z = np.interp(t,tarr[index],zarr[index])
    return z

def estimate_M200abund(Mstarlo, Mstarhi, tSFlow, tSFhigh):
    #Load in the abundance matching
    #relation (non-paramteric) from
    #Read & Erkal 2019:
    f = open('./Data/mSFR-mhalo-field.txt','r')
    data = np.genfromtxt(f)
    f.close()
    M200lowarr = data[:,0][np.where(data[:,1])][::-1]
    SFRlow = data[:,1][np.where(data[:,1])][::-1]
    M200higharr = data[:,0][np.where(data[:,2])][::-1]
    SFRhigh = data[:,2][np.where(data[:,2])][::-1]

    #Calculate the abundance matching mass:
    mSFRlo = Mstarlo / (tSFhigh*1.0e9)
    mSFRhi = Mstarhi / (tSFlow*1.0e9)
    M200_SFRhi = np.interp(mSFRhi,SFRlow,M200lowarr)
    M200_SFRlo = np.interp(mSFRlo,SFRhigh,M200higharr)

    return M200_SFRlo, M200_SFRhi

def findJsq(apo,peri,host,hostpars):
    '''Finds the SPECIFIC angular momentum squared (Jsq) for
       an orbit with apo and peri in a spherical potential with
       force function host and properties hostpars.'''  
    if apo-peri == 0:
        Jsq = apo**3.0*np.abs(host.fr(peri,hostpars))
    else:
        Jsq = (host.pot(peri,hostpars)-host.pot(apo,hostpars))*\
              (1.0/2.0/apo**2.0-1.0/2.0/peri**2.0)**(-1.0)
    return Jsq

def rtfunc(d,peri,apo,host,hostpars,sat,satpars,alpha,satmass,x):
    xs = d - x
    satpars[0] = satmass
    Jsq = findJsq(apo,peri,host,hostpars)
    Omega = np.sqrt(Jsq/d**4)
    Omega_s = np.sqrt(np.abs(sat.fr(x,satpars))/x)
    F = -host.fr(d,hostpars)+host.fr(xs,hostpars)-sat.fr(x,satpars)-\
        2*alpha*Omega*Omega_s*x-Omega*Omega*x
    return F

def findrt(d,peri,apo,host,hostpars,sat,satpars,alpha):
    '''This routine finds the tidal radius by solving eq. 7 of Read et al.
    2006. Inputs are:
      - d        : current sat position 
      - peri     : orbit pericentre
      - apo      : orbit apocentre
      - host     : host galaxy potential
      - hostpars : host galaxy potential parameters (depends on host)
      - sat      : sat galaxy potential
      - satpars  : sat galaxy pot pars
      - alpha    : sat internal orbit parameter [-1,0,1] for ret/rad/pro
    '''

    from scipy.optimize.zeros import bisect
    
    # Check inputs are sensible:
    assert d >= peri,\
           'satellite position %f < peri %f' % (d,peri)
    assert d <= apo,\
           'satellite position %f > apo %f' % (d,apo)
    
    # Root find to get rt:
    xmin = d/1.0e4 + d/1.0e4*1e-6
    xmax = d - d*1.0e-6
    rt = bisect(lambda x: rtfunc(d,peri,apo,host,hostpars,sat,
                                 satpars,alpha,satpars[0],x),xmin,xmax)
    return rt

def mleftmaxcalc(M200,rp,ra,galmodel):
    #This routine takes a host galaxy model (e.g. MW/M31)
    #and calculates the mass likely to be stripped given
    #a peri, rp, and apo, ra of the orbit.

    #Set the host galaxy model:
    if (galmodel == 'MW'):
        #Milky Way based on Read 08:
        import Tidecalc.galaxy as host
        hostpars = [3.0e10*Msun,2.5*kpc,0.25*kpc,0.0*Msun,\
                    1.0*kpc,1.0e12*Msun,
                    12.16*kpc,17.0,G,999]
    elif (galmodel == 'M31'):
        #M31 based on Geehan et al. 2006:
        hostpars = [8.4e10*Msun,5.4*kpc,0.26*kpc,3.3e10*Msun,\
                    0.61*kpc,6.8e11*Msun,
                    8.18*kpc,22,G,999]
    else:
        print('%s mass model not yet implemented! Oops' % (galmodel))
        print('Bye bye')
        sys.exit(0)

    #Set the satellite galaxy model to be NFW:
    c200 = cosmo_cfunc(M200,h)
    import Tidecalc.corenfw as sat
    satpars = [M200*Msun,c200,0.0,\
               1.0*kpc,rhocrit*Msun/kpc**3.0,G,999]

    #Assume the satellite is at pericentre and use the
    #retrograde stripping radius (max stripping):
    d = rp
    alpha = -1

    #Then calculate stripping radius as in Read+06:
    rt = findrt(d*kpc,rp*kpc,ra*kpc,host,hostpars,\
                sat,satpars,alpha)
    print('Tidal radius found (kpc):', rt/kpc)
    
    #Now assume all mass beyond rt is lost, giving
    #us mleft:
    mleft = sat.cummass(rt,satpars)
    print('Satellite mass within rt (1e9 Msun):', \
          mleft/1.0e9/Msun)

    return mleft/Msun


###########################################################
#For constraining particle DM models:
def rhoNFW(r,rhos,rs):
    return rhos/((r/rs)*(1.+(r/rs))**2.)

def sidm_novel(rc,M200,c,oden,rhocrit):
    #Calculate SIDM model parameters from the coreNFWtides
    #model fit. For this to be valid, the coreNFWtides fit
    #should assume a pure-core model, with n=1. See
    #Read et al. 2018 for further details.
    #Returns cross section/particle mass in cm^2 / g.
    GammaX = 0.005/(1e9*year)
    Guse = G*Msun/kpc
    rho_unit = Msun/kpc**3.0
    rc = np.abs(rc)*10.0
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c
    rhos=rhocrit*deltachar

    rhorc = rhoNFW(rc,rhos,rs)
    r = np.logspace(np.log10(rc),np.log10(rs*5000.0),50000)
    rho = rhoNFW(r,rhos,rs)
    mass = M200*gcon*(np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))
    sigvtworc = Guse/rhorc*integrator(mass*rho/r**2.0,r)
    sigvrc = np.sqrt(sigvtworc)

    sigm = np.sqrt(np.pi)*GammaX/(4.0*rhorc*rho_unit*sigvrc)
    return sigm*100.0**2.0/1000.0

def radius_dsph(s, b, distance):
    return np.sqrt((distance * np.sin(b))**2. + s*s)

def integrand(s, b, distance, rho, Mpars):
    value = np.sin(b) * rho(np.array([radius_dsph(s, b, distance)]), Mpars)**2
    return value

def integrand_D(s, b, distance, rho, Mpars):
    value = np.sin(b) * rho(np.array([radius_dsph(s, b, distance)]), Mpars)
    return value

def get_J(rho, Mpars, distance, r_max):
    """
    Compute the J factor.
    :param distance: the distance of the galaxy in kpc
    :param r_max: the maximum radius over which to integrate
                  [this gives an integration angle of
                   alpha = r_max/distance (rads)]
    :param r: the radius array for the density profile in kpc
    :param rho: the density array for the density profile in Msun/kpc^3
    :return: the J factor in in GeV c^-4 cm^-5
    """
    
    #Min/max integration angles in radians:
    b_min = 0.0
    b_max = np.arcsin(r_max/distance)
    
    #This is an appropriate choice for Dwarf galaxies but
    #should be reconsidered for large mass systems:
    Rmaximum = 250.0
    
    #Upper/lower limits:
    s_min_bound = lambda b :  -(Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    s_max_bound = lambda b : (Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    
    #Computation J_max:
    Acc_arr = 1.0e-8
    J_max = dblquad(integrand,b_min,b_max,s_min_bound,\
                    s_max_bound,args=(distance,rho,Mpars),\
                    epsabs=Acc_arr,epsrel=Acc_arr)
    J_max = J_max[0]*kpccm*2.*np.pi*Msunkpc3toGeVcm3**2.0

    #Error checking:
    if (J_max == np.inf):
        print('Argh! Infinite J_max!! Bye bye...')
        sys.exit(0)
        
    if (J_max < 0):
        print('Argh! Negative J_max!! Bye bye...')
        sys.exit(0)

    return J_max  # in GeV^2 c^-4 cm^-5

def get_D(rho, Mpars, distance, r_max):
    """
    Compute the D factor.
    :param distance: the distance of the galaxy in kpc
    :param r_max: the maximum radius over which to integrate
                  [this gives an integration angle of
                   alpha = r_max/distance (rads)]
    :param r: the radius array for the density profile in kpc
    :param rho: the density array for the density profile in Msun/kpc^3
    :return: the D factor in in GeV c^-2 cm^-2
    """

    # Min/max integration angles in radians:
    r_min = 0.0
    b_min = np.arcsin(r_min/distance)
    b_max = np.arcsin(r_max/distance)
                        
    #This is an appropriate choice for Dwarf galaxies but
    #should be reconsidered for large mass systems:
    Rmaximum = 250.0
    
    #Upper/lower limits:
    s_min_bound = lambda b :  -(Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    s_max_bound = lambda b : (Rmaximum**2 - (distance*np.sin(b))**2 )**0.5
    
    #Computation J_max:
    Acc_arr = 1.0e-8          
    D_max = dblquad(integrand_D,b_min,b_max,s_min_bound,\
                       s_max_bound,args=(distance,rho,Mpars),\
                       epsabs=Acc_arr,epsrel=Acc_arr)
    D_max = D_max[0]*kpccm*2.*np.pi*Msunkpc3toGeVcm3

    #Error checking:
    if (D_max == np.inf):
        print('Argh! Infinite D_max!! Bye bye...')
        sys.exit(0)
    if (D_max < 0):
        print('Argh! Negative D_max!! Bye bye...')
        sys.exit(0)
        
    return D_max  # in GeV c^-2 cm^-2


###########################################################
#For DM mass profile:
def cosmo_profile_mass(r,M200,nsig_c200,tSF,mWDM,sigmaSI,\
                       mleft,zin):
    #Assume M200 here defined at zin which is the redshift
    #of subhalo infall. This can be fit as a free parameter
    #or fixed by a prior.

    #**TO DO**
    #1. improve dark matter heating model, fit
    #on latest data using DarkLight!
    #2. allow dark matter heating and sigmaSI to work
    #together
    
    #Hard code choice of mass-concentration relation for
    #now. Can be changed here if wanted.
    cNFW_method='d15'
    csig = dis.dark_matter.concentrations.Diemer19.scatter()
    
    #Calculate c200 from nsig_c200. This will depend
    #on the choice of M200-c200 relation, and the
    #cosmology. In this case, set by mWDM.
    c200mean = dis.genutils.cNFW(M200,z=zin,method=cNFW_method,\
                        wdm=True,mWDM=mWDM)
    c200 = c200mean*10.0**(nsig_c200*csig)

    #Bit of hack here too for now ***WARNING***.
    #If sigmaSI > 1.0e-4 then assume we're fitting SIDM.
    #In this case, no dark matter heating will occur. This
    #needs to be fixed/tested/implemented. Otherwise if
    #sigmaSI < 1.0e-4 then assume no SIDM and dark matter
    #heating will be set by the tSF parameter (that is
    #ignored if fitting SIDM):
    if (sigmaSI < 1.0e-4):
        menc = dis.vutils.menc(r, M200, 'coreNFW', c200=c200,\
                        cNFW_method='d15',zin=zin,tSF=tSF,\
                        wdm=True,mWDM=mWDM,sigmaSI=None,\
                        mleft=mleft)
    else:
        menc = dis.vutils.menc(r, M200, 'sidm', c200=c200,\
                        cNFW_method='d15',zin=zin,tSF=None,\
                        wdm=True,mWDM=mWDM,sigmaSI=sigmaSI,\
                        mleft=mleft)

    return menc

def cosmo_profile_den(r, M200,nsig_c200,tSF,mWDM,sigmaSI,\
                      mleft,zin):
    return np.zeros(len(r))

def cosmo_profile_dlnrhodlnr(r, M200,nsig_c200,tSF,mWDM,sigmaSI,\
                             mleft,zin):
    return np.zeros(len(r))

def corenfw_tides_den(r,M200,c,rc,n,rt,delta):
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c

    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    x = r/np.abs(rc)
    f = np.tanh(x)
    my_manal = manal*f**n
    my_rhoanal = rhoanal*f**n + \
        1.0/(4.*np.pi*r**2.*np.abs(rc))*manal*(1.0-f**2.)*n*f**(n-1.0)
    frt = np.tanh(rt/np.abs(rc))
    manal_rt = M200 * gcon * (np.log(1.0 + rt/rs)-rt/rs/(1.0+rt/rs))
    my_rhoanal_rt = rhos/((rt/rs)*(1.+(rt/rs))**2.)*frt**n + \
        1.0/(4.*np.pi*rt**2.*np.abs(rc))*manal_rt*(1.0-frt**2.)*n*frt**(n-1.0)

    my_rhoanal[r > rt] = my_rhoanal_rt * (r[r > rt]/rt)**(-delta)

    return my_rhoanal

def corenfw_tides_mass(r,M200,c,rc,n,rt,delta):
    gcon=1./(np.log(1.+c)-c/(1.+c))
    deltachar=oden*c**3.*gcon/3.
    rv=(3./4.*M200/(np.pi*oden*rhocrit))**(1./3.)
    rs=rv/c

    rhos=rhocrit*deltachar
    rhoanal = rhos/((r/rs)*(1.+(r/rs))**2.)
    manal = M200 * gcon * (np.log(1.0 + r/rs)-r/rs/(1.0+r/rs))

    x = r/np.abs(rc)
    f = np.tanh(x)
    my_manal = manal*f**n

    frt = np.tanh(rt/np.abs(rc))
    manal_rt = M200 * gcon * (np.log(1.0 + rt/rs)-rt/rs/(1.0+rt/rs))
    my_rhoanal_rt = rhos/((rt/rs)*(1.+(rt/rs))**2.)*frt**n + \
        1.0/(4.*np.pi*rt**2.*np.abs(rc))*manal_rt*(1.0-frt**2.)*n*frt**(n-1.0)
    Mrt = manal_rt*frt**n

    my_manal[r > rt] = Mrt + \
        4.0*np.pi*my_rhoanal_rt*rt**3.0/(3.0-delta)*\
        ((r[r > rt]/rt)**(3.0-delta)-1.0)

    return my_manal

def corenfw_tides_dlnrhodlnr(r,M200,c,rc,n,rt,delta):
    dden = derivative(\
        lambda x: corenfw_tides_den(x,M200,c,rc,n,rt,delta),\
        r,dx=1e-6)
    dlnrhodlnr = dden / corenfw_tides_den(r,M200,c,rc,n,rt,delta) * r
    return dlnrhodlnr
    

###########################################################
#For DM mass profile and VSPs of GC mocks (for overlaying 
#true solution):
def alpbetgamden(r,rho0,r0,alp,bet,gam):
    return rho0*(r/r0)**(-gam)*(1.0+(r/r0)**alp)**((gam-bet)/alp)

def alpbetgamdlnrhodlnr(r,rho0,r0,alp,bet,gam):
    return -gam + (gam-bet)*(r/r0)**alp*(1.0+(r/r0)**alp)**(-1.0)

def alpbetgammass(r,rho0,r0,alp,bet,gam):
    den = rho0*(r/r0)**(-gam)*(1.0+(r/r0)**alp)**((gam-bet)/alp)
    mass = np.zeros(len(r))
    for i in range(3,len(r)):
        mass[i] = integrator(4.0*np.pi*r[:i]**2.*den[:i],r[:i])
    return mass
    
def alpbetgamsigr(r,rho0s,r0s,alps,bets,gams,rho0,r0,alp,bet,gam,ra):
    nu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    mass = alpbetgammass(r,rho0,r0,alp,bet,gam)
    gf = gfunc_osipkov(r,ra)
    sigr = np.zeros(len(r))
    for i in range(len(r)-3):
        sigr[i] = 1.0/nu[i]/gf[i] * \
            integrator(Guse*mass[i:]*nu[i:]/r[i:]**2.0*\
                       gf[i:],r[i:])
    return sigr

def osipkov(r,r0):
    return r**2.0/(r**2.0+r0**2.0)
    
def gfunc_osipkov(r,r0):
    n0 = 2.0
    bet0 = 0.0
    betinf = 1.0
    gfunc = r**(2.0*betinf)*\
        ((r0/r)**n0+1.0)**(2.0/n0*(betinf-bet0))
    return gfunc

def alpbetgamvsp(rho0s,r0s,alps,bets,gams,rho0,r0,alp,bet,gam,ra):
    intpnts = np.int(1e4)
    r = np.logspace(np.log10(r0s/50.0),np.log10(500.0*r0s),\
                    np.int(intpnts))
    nu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    massnu = alpbetgamden(r,rho0s,r0s,alps,bets,gams)
    mass = alpbetgammass(r,rho0,r0,alp,bet,gam)
    sigr = alpbetgamsigr(r,rho0s,r0s,alps,bets,gams,rho0,\
                         r0,alp,bet,gam,ra)
    bet = osipkov(r,ra)
    sigstar = np.zeros(len(r))
    for i in range(1,len(r)-3):
        sigstar[i] = 2.0*integrator(nu[i:]*r[i:]/\
                               np.sqrt(r[i:]**2.0-r[i-1]**2.0),\
                               r[i:])
 
    #Normalise similarly to the data:
    norm = integrator(sigstar*2.0*np.pi*r,r)
    nu = nu / norm
    sigstar = sigstar / norm

    #VSPs:
    vsp1 = \
        integrator(2.0/5.0*Guse*mass*nu*(5.0-2.0*bet)*\
            sigr*r,r)/1.0e12
    vsp2 = \
        integrator(4.0/35.0*Guse*mass*nu*(7.0-6.0*bet)*\
            sigr*r**3.0,r)/1.0e12
        
    #Richardson & Fairbairn zeta parameters:
    Ntotuse = integrator(sigstar*r,r)
    sigint = integrator(sigstar*r**3.0,r)
    zeta_A = 9.0/10.0*Ntotuse*integrator(Guse*mass*nu*(\
        5.0-2.0*bet)*sigr*r,r)/\
        (integrator(Guse*mass*nu*r,r))**2.0
    zeta_B = 9.0/35.0*Ntotuse**2.0*\
        integrator(Guse*mass*nu*(7.0-6.0*bet)*sigr*r**3.0,r)/\
        ((integrator(Guse*mass*nu*r,r))**2.0*sigint)
    return vsp1, vsp2, zeta_A, zeta_B

#Richardson-Fairbairn VSP estimators:
def richfair_vsp(vz,Rkin,mskin):
    vsp1_RF = 1.0/(np.pi*2.0)*\
        np.sum(vz**4.0*mskin)/np.sum(mskin)
    vsp2_RF = 1.0/(np.pi*2.0)*\
        np.sum(vz**4.0*mskin*Rkin**2.0)/np.sum(mskin*Rkin**2.0)
    return vsp1_RF, vsp2_RF


###########################################################
#For optional central dark mass (e.g. remnants, black hole):
def plumden(r,pars):
    return 3.0*pars[0]/(4.*np.pi*pars[1]**3.)*\
        (1.0+r**2./pars[1]**2.)**(-5./2.)

def plummass(r,pars):
    return pars[0]*r**3./(r**2.+pars[1]**2.)**(3./2.)


###########################################################
#For Jeans modelling:
def sigp(r1,r2,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,\
         betpars,\
         Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,G,rmin,rmax,intpnts):
    #Calculate projected velocity dispersion profiles
    #given input *functions* nu(r); M(r); beta(r); betaf(r).
    #Also input is an array Mstar_prof(Mstar_rad) describing the 3D 
    #cumulative stellar mass profile. This should be normalised 
    #so that it peaks at 1.0. The total stellar mass is passed in Mstar.

    #Set up theta integration array:
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.

    #Set up rint interpolation array: 
    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)

    #First calc sigr2(rint):
    sigr2 = np.zeros(len(rint))
    nur = nu(rint,nupars)
    betafunc = betaf(rint,betpars,Rhalf,Arot)
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
        if (Mstar > 0):
            Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
        nuq = nu(rq,nupars)
        betafuncq = betaf(rq,betpars,Rhalf,Arot)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)
    
    #And now the sigLOS projection: 
    Sig = Sigfunc(rint,nupars)
    sigLOS2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = nu(rq,nupars)
        sigr2q = np.interp(rq,rint,sigr2,left=0,right=0)
        betaq = beta(rq,betpars)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th)

    sigr2out = np.interp(r2,rint,sigr2,left=0,right=0)
    sigLOS2out = np.interp(r2,rint,sigLOS2,left=0,right=0)
    Sigout = np.interp(r1,rint,Sig,left=0,right=0)

    return sigr2out,Sigout,sigLOS2out

def sigp_vs(r1,r2,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,\
            betpars,\
            Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,G,rmin,rmax,intpnts):
    #Calculate projected velocity dispersion profiles
    #given input *functions* nu(r); M(r); beta(r); betaf(r).
    #Also input is an array Mstar_prof(Mstar_rad) describing the 3D
    #cumulative stellar mass profile. This should be normalised
    #so that it peaks at 1.0. The total stellar mass is passed in Mstar.
    #Finally, the routine calculates a dimensional version of the
    #fourth order "virial shape" parmaeters in Richardson & Fairbairn 2014
    #described in their equations 8 and 9.

    #Set up theta integration array:
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.

    #Set up rint interpolation array:
    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)

    #First calc sigr2(rint):
    sigr2 = np.zeros(len(rint))
    nur = nu(rint,nupars)
    betafunc = betaf(rint,betpars,Rhalf,Arot)
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
        if (Mstar > 0):
            Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
        nuq = nu(rq,nupars)
        betafuncq = betaf(rq,betpars,Rhalf,Arot)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)

    #And now the sigLOS projection:
    Sig = Sigfunc(rint,nupars)
    sigLOS2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = nu(rq,nupars)
        sigr2q = np.interp(rq,rint,sigr2,left=0,right=0)
        betaq = beta(rq,betpars)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th)

    #And now the dimensional fourth order "virial shape"
    #parameters:
    betar = beta(rint,betpars)
    Mr = M(rint,Mpars)+Mstar*np.interp(rint,Mstar_rad,Mstar_prof)
    vs1 = 2.0/5.0*integrator(nur*(5.0-2.0*betar)*sigr2*\
                             G*Mr*rint,rint)
    vs2 = 4.0/35.0*integrator(nur*(7.0-6.0*betar)*sigr2*\
                              G*Mr*rint**3.0,rint)

    sigr2out = np.interp(r2,rint,sigr2,left=0,right=0)
    sigLOS2out = np.interp(r2,rint,sigLOS2,left=0,right=0)
    Sigout = np.interp(r1,rint,Sig,left=0,right=0)

    return sigr2out,Sigout,sigLOS2out,vs1,vs2

def sigp_prop(r1,r2,r3,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,\
              betpars,\
              Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,G,rmin,rmax,intpnts):
    #Calculate projected velocity dispersion profiles
    #given input *functions* nu(r); M(r); beta(r); betaf(r).
    #Also input is an array Mstar_prof(Mstar_rad) describing the 3D
    #cumulative stellar mass profile. This should be normalised
    #so that it peaks at 1.0. The total stellar mass is passed in Mstar.

    #Set up theta integration array:
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.

    #Set up rint interpolation array:
    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)

    #First calc sigr2(rint):
    sigr2 = np.zeros(len(rint))
    nur = nu(rint,nupars)
    betafunc = betaf(rint,betpars,Rhalf,Arot)
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
        if (Mstar > 0):
            Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
        nuq = nu(rq,nupars)
        betafuncq = betaf(rq,betpars,Rhalf,Arot)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)
 
    #And now the sigLOS, sigpmr and sigpmt projections:
    Sig = Sigfunc(rint,nupars)
    sigLOS2 = np.zeros(len(rint))
    sigpmr2 = np.zeros(len(rint))
    sigpmt2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = nu(rq,nupars)
        sigr2q = np.interp(rq,rint,sigr2,left=0,right=0)
        betaq = beta(rq,betpars)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmr2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq+betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmt2[i] = 2.0*rint[i]/Sig[i]*\
            integrator((1.0-betaq)*nuq*sigr2q/cth2,th)

    sigr2out = np.interp(r2,rint,sigr2,left=0,right=0)
    sigLOS2out = np.interp(r2,rint,sigLOS2,left=0,right=0)
    sigpmr2out = np.interp(r3,rint,sigpmr2,left=0,right=0)
    sigpmt2out = np.interp(r3,rint,sigpmt2,left=0,right=0)
    Sigout = np.interp(r1,rint,Sig,left=0,right=0)
    
    return sigr2out,Sigout,sigLOS2out,sigpmr2out,sigpmt2out

def sigp_prop_vs(r1,r2,r3,nu,Sigfunc,M,Mcentral,beta,betaf,nupars,Mpars,\
                 betpars,\
                 Mstar_rad,Mstar_prof,Mstar,Arot,Rhalf,G,rmin,rmax,intpnts):
    #Calculate projected velocity dispersion profiles
    #given input *functions* nu(r); M(r); beta(r); betaf(r).
    #Also input is an array Mstar_prof(Mstar_rad) describing the 3D
    #cumulative stellar mass profile. This should be normalised
    #so that it peaks at 1.0. The total stellar mass is passed in Mstar.
    
    #Set up theta integration array:
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    cth2 = cth**2.

    #Set up rint interpolation array:
    rint = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)
    
    #First calc sigr2(rint):
    sigr2 = np.zeros(len(rint))
    nur = nu(rint,nupars)
    betafunc = betaf(rint,betpars,Rhalf,Arot)
    for i in range(len(rint)):
        rq = rint[i]/cth
        Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
        if (Mstar > 0):
            Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
        nuq = nu(rq,nupars)
        betafuncq = betaf(rq,betpars,Rhalf,Arot)
        sigr2[i] = 1./nur[i]/rint[i]/betafunc[i] * \
            integrator(G*Mq*nuq*betafuncq*sth,th)

    #And now the sigLOS, sigpmr and sigpmt projections:
    Sig = Sigfunc(rint,nupars)
    sigLOS2 = np.zeros(len(rint))
    sigpmr2 = np.zeros(len(rint))
    sigpmt2 = np.zeros(len(rint))
    for i in range(len(rint)):
        rq = rint[i]/cth
        nuq = nu(rq,nupars)
        sigr2q = np.interp(rq,rint,sigr2,left=0,right=0)
        betaq = beta(rq,betpars)
        sigLOS2[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((1.0-betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmr2[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((1.0-betaq+betaq*cth2)*nuq*sigr2q/cth2,th)
        sigpmt2[i] = 2.0*rint[i]/Sig[i]*\
                     integrator((1.0-betaq)*nuq*sigr2q/cth2,th)
        
    sigr2out = np.interp(r2,rint,sigr2,left=0,right=0)
    sigLOS2out = np.interp(r2,rint,sigLOS2,left=0,right=0)
    sigpmr2out = np.interp(r3,rint,sigpmr2,left=0,right=0)
    sigpmt2out = np.interp(r3,rint,sigpmt2,left=0,right=0)
    Sigout = np.interp(r1,rint,Sig,left=0,right=0)

    #And now the dimensional fourth order "virial shape"
    #parameters:
    betar = beta(rint,betpars)
    Mr = M(rint,Mpars)+Mstar*np.interp(rint,Mstar_rad,Mstar_prof)
    vs1 = 2.0/5.0*integrator(nur*(5.0-2.0*betar)*sigr2*\
                             G*Mr*rint,rint)
    vs2 = 4.0/35.0*integrator(nur*(7.0-6.0*betar)*sigr2*\
                              G*Mr*rint**3.0,rint)
    
    return sigr2out,Sigout,sigLOS2out,sigpmr2out,sigpmt2out,\
        vs1,vs2

def beta(r,betpars):
    bet0star = betpars[0]
    betinfstar = betpars[1]
    r0 = 10.**betpars[2]
    n = betpars[3]

    #Ensure stability at beta extremities:
    if (bet0star > 0.98): 
        bet0star = 0.98
    if (bet0star < -0.95):
        bet0star = -0.95
    if (betinfstar > 0.98):
        betinfstar = 0.98
    if (betinfstar < -0.95):
        betinfstar = -0.95
    bet0 = 2.0*bet0star / (1.0 + bet0star)
    betinf = 2.0*betinfstar / (1.0 + betinfstar)

    beta = bet0 + (betinf-bet0)*(1.0/(1.0 + (r0/r)**n))
    return beta

def betaf(r,betpars,Rhalf,Arot):
    bet0star = betpars[0]
    betinfstar = betpars[1]
    r0 = 10.**betpars[2]
    n = betpars[3]

    #Ensure stability at beta extremities:
    if (bet0star > 0.98):
        bet0star = 0.98
    if (bet0star < -0.95):
        bet0star = -0.95
    if (betinfstar > 0.98):
        betinfstar = 0.98
    if (betinfstar < -0.95):
        betinfstar = -0.95
    bet0 = 2.0*bet0star / (1.0 + bet0star)
    betinf = 2.0*betinfstar / (1.0 + betinfstar)

    betafn = r**(2.0*betinf)*((r0/r)**n+1.0)**(2.0/n*(betinf-bet0))*\
             np.exp(-2.0*Arot*r/Rhalf)
    
    return betafn

def accelminmax_calc(r,M,Mcentral,Mpars,\
                     Mstar_rad,Mstar_prof,Mstar,G,rmin,rmax,intpnts):

    #Calculate acceleration as a function of radius (SI units):
    rq = np.logspace(np.log10(rmin),np.log10(rmax),intpnts)
    Mq = M(rq,Mpars)+Mcentral(rq,Mpars)
    if (Mstar > 0):
        Mq = Mq+Mstar*np.interp(rq,Mstar_rad,Mstar_prof)
    aq = G*Mq/rq**2.0/kpc

    #Now at each projected radius, scan alone line
    #of sight to cacl min/max acceleration:
    Rproj = r
    thmin = 0.
    bit = 1.e-5
    thmax = np.pi/2.-bit
    th = np.linspace(thmin,thmax,intpnts)
    sth = np.sin(th)
    cth = np.cos(th)
    aminmax = np.zeros(len(Rproj))
    for i in range(len(Rproj)):
        #Define angle th s.t. th = pi points towards us
        #and th = -pi points away. Acc as a function of
        #theta is: alos = np.abs(aq(Rproj[i]/cth)*sth)
        #By symmetry, only need to consider th=[0,pi/2]:
        aqq = np.interp(Rproj[i]/cth,rq,aq)
        alos = np.abs(aqq*sth)

        #The above (appropriately normalised) is the
        #likelihood of a line of sight aceleration.
        #Here, we want min/max:
        aminmax[i] = np.max(alos)
    return aminmax

###########################################################
#For data binning:
def binthedata(R,ms,Nbin):
    #Nbin is the number of particles / bin:
    index = np.argsort(R)
    right_bin_edge = np.zeros(len(R))
    norm = np.zeros(len(R))
    cnt = 0
    jsum = 0

    for i in range(len(R)):
        if (jsum < Nbin):
            norm[cnt] = norm[cnt] + ms[index[i]]
            right_bin_edge[cnt] = R[index[i]]
            jsum = jsum + ms[index[i]]
        if (jsum >= Nbin):
            jsum = 0.0
            cnt = cnt + 1
    
    right_bin_edge = right_bin_edge[:cnt]
    norm = norm[:cnt]
    surfden = np.zeros(cnt)
    rbin = np.zeros(cnt)
    
    for i in range(len(rbin)):
        if (i == 0):
            surfden[i] = norm[i] / \
                (np.pi*right_bin_edge[i]**2.0)
            rbin[i] = right_bin_edge[i]/2.0
        else:
            surfden[i] = norm[i] / \
                (np.pi*right_bin_edge[i]**2.0-\
                 np.pi*right_bin_edge[i-1]**2.0)
            rbin[i] = (right_bin_edge[i]+right_bin_edge[i-1])/2.0
    surfdenerr = surfden / np.sqrt(Nbin)
    
    #Calculate the projected half light radius &
    #surface density integral:
    Rhalf, Menc_tot = surf_renorm(rbin,surfden)

    #And normalise the profile:
    surfden = surfden / Menc_tot
    surfdenerr = surfdenerr / Menc_tot

    return rbin, surfden, surfdenerr, Rhalf

def surf_renorm(rbin,surfden):
    #Calculate the integral of the surface density
    #so that it can then be renormalised.
    #Calcualte also Rhalf.
    ranal = np.linspace(0,10,np.int(5000))
    surfden_ranal = np.interp(ranal,rbin,surfden,left=0,right=0)
    Menc_tot = 2.0*np.pi*integrator(surfden_ranal*ranal,ranal)
    Menc_half = 0.0
    i = 3
    while (Menc_half < Menc_tot/2.0):
        Menc_half = 2.0*np.pi*\
            integrator(surfden_ranal[:i]*ranal[:i],ranal[:i])
        i = i + 1
    Rhalf = ranal[i-1]
    return Rhalf, Menc_tot


###########################################################
#For calculating confidence intervals: 
def calcmedquartnine(array):
    index = np.argsort(array,axis=0)
    median = array[index[np.int(len(array)/2.)]]
    sixlowi = np.int(16./100. * len(array))
    sixhighi = np.int(84./100. * len(array))
    ninelowi = np.int(2.5/100. * len(array))
    ninehighi = np.int(97.5/100. * len(array))
    nineninelowi = np.int(0.15/100. * len(array))
    nineninehighi = np.int(99.85/100. * len(array))

    sixhigh = array[index[sixhighi]]
    sixlow = array[index[sixlowi]]
    ninehigh = array[index[ninehighi]]
    ninelow = array[index[ninelowi]]
    nineninehigh = array[index[nineninehighi]]
    nineninelow = array[index[nineninelowi]]

    return median, sixlow, sixhigh, ninelow, ninehigh,\
        nineninelow, nineninehigh


###########################################################
#For fitting the surface brightness:
def Sig_addpnts(x,y,yerr):
    #If using neg. Plummer component, add some more
    #"data points" at large & small radii bounded on
    #zero and the outermost data point. This
    #will disfavour models with globally
    #negative tracer density.
    addpnts = len(x)
    xouter = np.max(x)
    youter = np.min(y)
    xinner = np.min(x)
    yinner = np.max(y)
    xadd_right = np.logspace(np.log10(xouter),\
                             np.log10(xouter*1000),addpnts)
    yadd_right = np.zeros(addpnts) + youter/2.0
    yerradd_right = yadd_right
    xadd_left = np.logspace(np.log10(xinner),\
                            np.log10(xinner/1000),addpnts)
    yadd_left = np.zeros(addpnts) + yinner
    yerradd_left = yadd_left/2.0
    x = np.concatenate((x,xadd_right))
    y = np.concatenate((y,yadd_right))
    yerr = np.concatenate((yerr,yerradd_right))
    x = np.concatenate((xadd_left,x))
    y = np.concatenate((yadd_left,y))
    yerr = np.concatenate((yerradd_left,yerr))
    return x,y,yerr

#For stellar and tracer profiles:
def multiplumden(r,pars):
    Mpars = pars[0:np.int(len(pars)/2.0)]
    apars = pars[np.int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            3.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            (1.0+r**2./aparsu**2.)**(-5./2.)
    return multplum

def multiplumsurf(r,pars):
    Mpars = pars[0:np.int(len(pars)/2.0)]
    apars = pars[np.int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            Mpars[i]*aparsu**2.0 / \
            (np.pi*(aparsu**2.0+r**2.0)**2.0)
    return multplum

def multiplumdlnrhodlnr(r,pars):
    Mpars = pars[0:np.int(len(pars)/2.0)]
    apars = pars[np.int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplumden = np.zeros(len(r))
    multplumdden = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplumden = multplumden + \
            3.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            (1.0+r**2./aparsu**2.)**(-5./2.)
        multplumdden = multplumdden - \
            15.0*Mpars[i]/(4.*np.pi*aparsu**3.)*\
            r/aparsu**2.*(1.0+r**2./aparsu**2.)**(-7./2.)
    return multplumdden*r/multplumden

def multiplummass(r,pars):
    Mpars = pars[0:np.int(len(pars)/2.0)]
    apars = pars[np.int(len(pars)/2.0):len(pars)]
    nplum = len(Mpars)
    multplum = np.zeros(len(r))
    for i in range(len(Mpars)):
        if (multimode == 'seq'):
            if (i == 0):
                aparsu = apars[0]
            else:
                aparsu = apars[i] + apars[i-1]
        else:
            aparsu = apars[i]
        multplum = multplum + \
            Mpars[i]*r**3./(r**2.+aparsu**2.)**(3./2.)
    return multplum

def threeplumsurf(r,M1,M2,M3,a1,a2,a3):
    return multiplumsurf(r,[M1,M2,M3,\
                            a1,a2,a3])
def threeplumden(r,M1,M2,M3,a1,a2,a3):
    return multiplumden(r,[M1,M2,M3,\
                           a1,a2,a3])
def threeplummass(r,M1,M2,M3,a1,a2,a3):
    return multiplummass(r,[M1,M2,M3,\
                            a1,a2,a3])

def Rhalf_func(M1,M2,M3,a1,a2,a3):
    #Calculate projected half light radius for
    #the threeplum model:
    ranal = np.logspace(-3,1,np.int(500))
    Mstar_surf = threeplumsurf(ranal,M1,M2,M3,a1,a2,a3)

    Menc_half = 0.0
    i = 3
    while (Menc_half < (M1+M2+M3)/2.0):
        Menc_half = 2.0*np.pi*\
            integrator(Mstar_surf[:i]*ranal[:i],ranal[:i])
        i = i + 1
    Rhalf = ranal[i-1]
    return Rhalf


###########################################################
#For setting up walkers for fitting:
def blobcalc(lowin,highin):
    bitscale = (highin-lowin)/2.0
    low = (lowin+highin)/2.0-bitscale*0.1
    high = (lowin+highin)/2.0+bitscale*0.1
    return low, high


###########################################################
#For fitting the velocity distribution in each bin [no errors]:
def monte(func,a,b,n):
    #Function to perform fast 1D Monte-Carlo integration
    #for convolution integrals:
    xrand = np.random.uniform(a,b,n)
    integral = func(xrand).sum()
    return (b-a)/np.float(n)*integral

def velpdf_noerr(vz,theta):
    vzmean = theta[0]
    alp = theta[1]
    bet = theta[2]
    backamp = theta[3]
    backmean = theta[4]
    backsig = theta[5]

    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
        np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
        backamp/(np.sqrt(2.0*np.pi)*backsig)*\
        np.exp(-0.5*(vz-backmean)**2.0/backsig**2.0)
    return pdf

#For fitting the velocity distribution in each bin [fast]
#Uses an approximation to the true convolution integral.
def velpdffast(vz,vzerr,theta):
    vzmean = theta[0]
    bet = theta[2]
    fgamma = gamma(1.0/bet)/gamma(3.0/bet)
    alp = np.sqrt(theta[1]**2.0+vzerr**2.0*fgamma)    
    backamp = theta[3]
    backmean = theta[4]
    backsig = np.sqrt(theta[5]**2.0 + vzerr**2.0)
    
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
        np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
        backamp/(np.sqrt(2.0*np.pi)*backsig)*\
        np.exp(-0.5*(vz-backmean)**2.0/backsig**2.0)
    return pdf

#Version used if you want to make sure the "background"
#has either positive or negative mean, but not zero. This is
#useful for weeding out contamination from tidal tails
#wrapped over the line of sight. See e.g. de Leo et al. 2023
#on modelling the SMC for further details.
def velpdffastswitch(vz,vzerr,theta):
    vzmean = theta[0]
    bet = theta[2]
    fgamma = gamma(1.0/bet)/gamma(3.0/bet)
    alp = np.sqrt(theta[1]**2.0+vzerr**2.0*fgamma)
    backamp = theta[3]
    if (theta[6] > 0):
        backmean = np.abs(theta[4])
    else:
        backmean = -np.abs(theta[4])
    backsig = np.sqrt(theta[5]**2.0 + vzerr**2.0)
    
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
          np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
          backamp/(np.sqrt(2.0*np.pi)*backsig)*\
          np.exp(-0.5*(vz-backmean)**2.0/backsig**2.0)
    return pdf

#Version used if you want to represent the background
#with two Gaussians; one with positive and one with
#negative mean. This is important for modelling tidal
#tails wrapped back on the line of sight. See e.g. de Leo
#et al. 2023 on modelling the SMC for further details.
def velpdfdoubleback(vz,vzerr,theta):
    vzmean = theta[0]
    bet = theta[2]
    fgamma = gamma(1.0/bet)/gamma(3.0/bet)
    alp = np.sqrt(theta[1]**2.0+vzerr**2.0*fgamma)
    backamp1 = theta[3]
    backmean1 = np.abs(theta[4])
    backsig1 = np.sqrt(theta[5]**2.0 + vzerr**2.0)
    backamp2 = theta[6]
    backmean2 = -np.abs(theta[7])
    backsig2 = np.sqrt(theta[8]**2.0 + vzerr**2.0)
    backamp = backamp1 + backamp2
    
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
          np.exp(-(np.abs(vz-vzmean)/alp)**bet) + \
          backamp1/(np.sqrt(2.0*np.pi)*backsig1)*\
          np.exp(-0.5*(vz-backmean1)**2.0/backsig1**2.0) + \
          backamp2/(np.sqrt(2.0*np.pi)*backsig2)*\
          np.exp(-0.5*(vz-backmean2)**2.0/backsig2**2.0)
    return pdf
    
def velpdf_func(vz,vzerr,vzint,theta):
    #Inner integral function for convolving
    #velpdf with a Gaussian error PDF. Change
    #this function to implement non-Gaussian
    #errors.
    vzmean = theta[0]
    alp = theta[1]
    bet = theta[2]
    backamp = theta[3]
    backmean = theta[4]
    backsig = theta[5]
    pdf = (1.0-backamp)*bet/(2.0*alp*gamma(1.0/bet))*\
                np.exp(-(np.abs(vzint-vzmean)/alp)**bet)*\
                1.0/(np.sqrt(2.0*np.pi)*vzerr)*\
                np.exp(-0.5*(vz-vzint)**2.0/vzerr**2.0)+\
                backamp/(np.sqrt(2.0*np.pi)*backsig)*\
                np.exp(-0.5*(vzint-backmean)**2.0/backsig**2.0)*\
                1.0/(np.sqrt(2.0*np.pi)*vzerr)*\
                np.exp(-0.5*(vz-vzint)**2.0/vzerr**2.0)
    return pdf

#For fitting the velocity distribution in each bin with
#full (expensive) convolution integral:
def velpdf(vz,vzerr,theta):
    #Generalised Gaussian + Gaussian convolved with
    #vzerr, assuming Gaussian errors:
    vzmean = theta[0]
    sig = vztwo_calc(theta)
    vzlow = -sig*10+vzmean
    vzhigh = sig*10+vzmean
    if (type(vz) == np.ndarray):
        pdf = np.zeros(len(vz))
        for i in range(len(vz)):
            pdf_func = lambda vzint : velpdf_func(vz[i],\
                vzerr[i],vzint,theta)
            pdf[i] = quad(pdf_func,vzlow,vzhigh)[0]
    else:
        pdf_func = lambda vzint : velpdf_func(vz,\
            vzerr,vzint,theta)
        pdf = quad(pdf_func,vzlow,vzhigh)[0]
    return pdf

def velpdfmonte(vz,vzerr,theta):
    #Generalised Gaussian + Gaussian convolved with
    #vzerr, assuming Gaussian errors:
    npnts = np.int(500)
    vzmean = theta[0]
    sig = vztwo_calc(theta)
    vzlow = -sig*10+vzmean
    vzhigh = sig*10+vzmean
    if (type(vz) == np.ndarray):
        pdf = np.zeros(len(vz))
        for i in range(len(vz)):
            pdf_func = lambda vzint : velpdf_func(vz[i],\
                vzerr[i],vzint,theta)
            pdf[i] = monte(pdf_func,vzlow,vzhigh,npnts)
    else:
        pdf_func = lambda vzint : velpdf_func(vz,\
            vzerr,vzint,theta)
        pdf = monte(pdf_func,vzlow,vzhigh,npnts)
    return pdf

def vztwo_calc(theta):
    #Calculate <vlos^2>^(1/2) from
    #generalised Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    return np.sqrt(alp**2.0*gamma(3.0/bet)/gamma(1.0/bet))
    
def vzfour_calc(theta):
    #Calculate <vlos^4> from
    #generalised Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    sig = vztwo_calc(theta)
    kurt = gamma(5.0/bet)*gamma(1.0/bet)/(gamma(3.0/bet))**2.0
    return kurt*sig**4.0
    
def kurt_calc(theta):
    #Calculate kurtosis from generalised
    #Gaussian parameters:
    alp = theta[1]
    bet = theta[2]
    kurt = gamma(5.0/bet)*gamma(1.0/bet)/(gamma(3.0/bet))**2.0
    return kurt

def vzfourfunc(ranal,rbin,vzfourbin):
    #Interpolate and extrapolate
    #vzfour(R) over and beyond the data:
    vzfour = np.interp(ranal,rbin,vzfourbin)
    return vzfour
    
#For calculating the Likelihood from the vsp array:
def vsppdf_calc(vsp):
    #First bin the data:
    nbins = 50
    bins_plus_one = np.linspace(np.min(vsp),np.max(vsp),nbins+1)
    bins = np.linspace(np.min(vsp),np.max(vsp),nbins)
    vsp_pdf, bins_plus_one = np.histogram(vsp, bins=bins_plus_one)
    vsp_pdf = vsp_pdf / np.max(vsp_pdf)
    binsout = bins[vsp_pdf > 0]
    vsp_pdfout = vsp_pdf[vsp_pdf > 0]
    return binsout, vsp_pdfout

def vsp_pdf(vsp,bins,vsp_pdf):
    return np.interp(vsp,bins,vsp_pdf,left=0,right=0)

