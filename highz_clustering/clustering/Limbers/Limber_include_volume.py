import os
import sys
import numpy as np
from astropy.io import fits as pf
from sklearn.neighbors import KernelDensity as kde
from scipy import integrate
import camb
from camb import model
from scipy.special import j0
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax3d
from skmonaco import mcquad
from skmonaco import mcmiser
import time

#Import SpIES / SHELA data
data = '/Users/johntimlin/Clustering/Combine_SpIES_Shela/KDE_RF_candidates/Limber_eval_fits/Matched_candidates_wz.fits'
obs = pf.open(data)[1].data
gdx = ((obs.Z >= 2.9)&(obs.Z <= 5.1))
#gdx = obs.Z>0
#Set up a KDE for dNdz
tmpz = obs.Z[gdx][:, np.newaxis] #change the array from row shape (1) to column shape (1,)
print np.shape(tmpz)
sample_range = np.linspace(min(tmpz[:, 0]), max(tmpz[:, 0]), len(tmpz[:, 0]))[:, np.newaxis]
est = kde(bandwidth=0.1,kernel='gaussian') #Set up the Kernel
histkde = est.fit(tmpz).score_samples(sample_range) #fit the kernel to the data and find the density of the grid
#Interpolate (you get the same function back) to plug in any z in the range (as opposed to set z values)
dNdz = interpolate.interp1d(sample_range.flatten(),np.exp(histkde))

'''
#Plot the KDE dndz
plt.plot(sample_range[:,0],np.exp(histkde))
#plt.plot(sample_range[:,0],dNdz(sample_range[:,0]))
#plt.plot(bins[:-1],num,linestyle = 'steps-mid')

ZE = np.linspace(min(obs.Z),max(obs.Z),100)
xo=integrate.quad(dNdz,min(sample_range),max(sample_range)) #quad(f(x),xlower,xupper, args)
print xo

plt.show() 
'''


#First define Planck 2015 cosmological parameters
H = 70 #H0. 
oc = 0.1125 #physical density of CDM 
ob = 0.022 #physical density of baryons

#Conversion to density param: Omega_Matter = (oc+ob)/(H0/100.)**2

#Set up parameters in CAMB
pars = camb.CAMBparams()
#H0 is hubble parameter at z=0, ombh2 is the baryon density (physical), omch2 is the matter density (physical)
#mnu is sum of neutrino masses, omk is curvature parameter (set to 0 for flat), meffsterile is effective mass of sterile neutrinos
pars.set_cosmology(H0=H,ombh2=ob, omch2=oc,omk=0)#,mnu=0,meffsterile=0) 
pars.set_dark_energy()

#Set parameters using standard power law parameterization.If nt=None, uses inflation consistency relation.
#ns is scalar speectral index
pars.InitPower.set_params(ns=0.960)
camb.set_halofit_version(version='original') #uses the Smith 2003 halo model
ze=np.linspace(0,20,150)
ka=np.logspace(-3,2,len(ze))#np.linspace(0,10,100)
#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
#pars: input parameters, zs: redshift range, nonlinear: generate nonlinear power spectrum, hubble_units=True: output as Mpc^-3 
#instead of Mpc/h^-3 
PK = camb.get_matter_power_interpolator(pars,zs = ze,zmax = ze[-1], nonlinear=True, hubble_units=True, kmax = ka[-1])
#Generate the power using the interpolator and the z and k arrays
#Power = PK.P(z,k)
#PK2 = camb.get_matter_power_interpolator(pars,zs = ze,zmax = ze[-1], nonlinear=True, hubble_units=False, kmax = ka[-1])

def dimpower(Pk,z,k):
    delta = Pk.P(z,k) * k**3/(2*np.pi**2)
    return delta

def domega(k,z,theta,cambpars,H0,dndz,Power,OmegaM,OmegaL,evalint=False):
    if evalint == True:
    	#Use this if integrating ln(10)k dlog(k)
    	#start = time.time()
        k=kz[0]
        z=kz[1]
        bkg = camb.get_background(cambpars)
        x = 10**k * (theta/60./180.*np.pi) * bkg.comoving_radial_distance(z)
        om = (H0/3.0e5) * 10**(-k) * dimpower(Power,z,10**k) * dndz(z)**2 * j0(x) * (OmegaM*(1+z)**3+OmegaL)**0.5*np.log(10)
        #end = time.time()
        #print end-start
        ## USe this if integrating dk 
        #x = k * theta * bkg.comoving_radial_distance(z)
        #om = (H0/3.0e5) * k**-2 * dimpower(Power,z,k) * dndz(z)**2 * j0(x) * (OmegaM*(1+z)**3+OmegaL)**0.5
    if evalint == False:
        #project the z array onto new axis to output a matrix when evaluating in k and z. This allows
        #me to plot a wireframe 3d plot
        #k=kz[0]
        #z=kz[1]
        z = np.array(z) 
        z = z[:,np.newaxis]
        bkg = camb.get_background(cambpars)
        x = k * theta * bkg.comoving_radial_distance(z)
        om = (H0/3.0e5) * k**-2 * dimpower(Power,z,k) * dndz(z)**2 * j0(x) * (OmegaM*(1+z)**3+OmegaL)**0.5
    
    return om



#parameters if integrate == False
theta = 1./60./180.*np.pi # radians = arcmin/60/180*pi
z = np.linspace(3.05,5,100)
k = np.logspace(-3,2,100)
Vol = 1265237462.94
omegaM = (oc+ob)/(H/100.)**2
omegaL= 1.0-omegaM
#Generate the surface under which to integrate
surf = domega(k,z,theta,pars,H,dNdz,PK,omegaM,omegaL)
#Set up meshgrid such that z interates over the columns and k iterates over the row
K,Z = np.meshgrid(k,z)

plt.figure(4)
plt.plot(K[0],surf[0])
plt.xscale('log')
plt.xlabel('k')
plt.ylabel(r'$\delta^2$w')

plt.figure(5)
plt.plot(Z[:,0],surf[:,0])
plt.xscale('linear')
plt.xlabel('z')
plt.ylabel(r'$\delta^2$w')

fig = plt.figure(6)
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(Z,np.log10(K),surf)
ax.set_xlabel('z')
ax.set_ylabel('log10(k)')
ax.set_zlabel(r'$\delta^2$w')

plt.show()
