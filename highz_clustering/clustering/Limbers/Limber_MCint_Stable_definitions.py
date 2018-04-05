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
from multiprocessing import Pool
from functools import partial

#Import SpIES / SHELA data
data = '../Data_Sets/HZLZ_combined_all_hzclassifiers_wphotoz_zspecflg.fits'
obs = pf.open(data)[1].data
Z = obs.zphotNW

gdx = (obs.dec>=-1.2) & (obs.dec<=1.2)

Z = Z[gdx]


def Zhist(zarray, zlow, zhigh):
	gdx = (Z >= zlow)&(Z <zhigh) 
	
	#Set up a KDE for dNdz
	tmpz = Z[gdx][:, np.newaxis] #change the array from row shape (1) to column shape (1,)
	print np.shape(tmpz)
	sample_range = np.linspace(min(tmpz[:, 0]), max(tmpz[:, 0]), len(tmpz[:, 0]))[:, np.newaxis]
	est = kde(bandwidth=0.1,kernel='epanechnikov') #Set up the Kernel
	histkde = est.fit(tmpz).score_samples(sample_range) #fit the kernel to the data and find the density of the grid
	#Interpolate (you get the same function back) to plug in any z in the range (as opposed to set z values)
	dNdz = interpolate.interp1d(sample_range.flatten(),np.exp(histkde))
	print sample_range.flatten()
	print 'done'
	ZE = np.linspace(min(Z),max(Z),100)
	xo=integrate.quad(dNdz,min(sample_range),max(sample_range)) #quad(f(x),xlower,xupper, args)
	print xo
	return dNdz


dNdz = Zhist(Z,2.9,3.4)


'''
#Plot the KDE dndz
plt.plot(sample_range[:,0],np.exp(histkde))
plt.xlabel('z')
#plt.plot(sample_range[:,0],dNdz(sample_range[:,0]))
#plt.plot(bins[:-1],num,linestyle = 'steps-mid')
ZE = np.linspace(min(Z),max(Z),100)
xo=integrate.quad(dNdz,min(sample_range),max(sample_range)) #quad(f(x),xlower,xupper, args)
print xo
plt.savefig('dndz.png')
plt.show() 
'''
	# Compute the matter power spectrum from CAMB and Generate the P(z,k) function to output the power at any given redshift
#and wavenumber

#First define Planck 2015 cosmological parameters
H = 70 #H0. 
oc = 0.229 #physical density of CDM 
ob = 0.046 #physical density of baryons

#Conversion to density param: Omega_Matter = (oc+ob)/(H0/100.)**2

#Set up parameters in CAMB
pars = camb.CAMBparams()
#H0 is hubble parameter at z=0, ombh2 is the baryon density (physical), omch2 is the matter density (physical)
#mnu is sum of neutrino masses, omk is curvature parameter (set to 0 for flat), meffsterile is effective mass of sterile neutrinos
#pars.set_cosmology(H0=H,ombh2=ob, omch2=oc,omk=0)#,mnu=0,meffsterile=0) 

#Hard code the cosmolgy params
pars.H0=H #hubble param (No h!!)
pars.omegab=ob #Baryon density parameter
pars.omegac=oc #CDM density parameter
pars.omegav=0.725 #Vacuum density parameter
pars.set_dark_energy()

#Set parameters using standard power law parameterization.If nt=None, uses inflation consistency relation.
#ns is scalar speectral index
pars.InitPower.set_params(ns=0.960)
camb.set_halofit_version(version='original') #uses the Smith 2003 halo model
ze=np.linspace(0,20,150)
ka=np.logspace(-4,2,len(ze))#np.linspace(0,10,100)
#Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
#pars: input parameters, zs: redshift range, nonlinear: generate nonlinear power spectrum, hubble_units=True: output as Mpc/h^3 
#instead of Mpc^3 
PK = camb.get_matter_power_interpolator(pars,zs = ze,zmax = ze[-1], nonlinear=True, hubble_units=True, k_hunit=True, kmax = ka[-1])
#Generate the power using the interpolator and the z and k arrays
#Power = PK.P(z,k)

cosmo = [oc,ob, H]



def dimpower(Pk,z,k):
    delta = Pk.P(z,k) * k**3/(2*np.pi**2)
    return delta
    
def domega(kz,theta,cambpars,H0,dndz,Power,OmegaM,OmegaL,evalint=False):
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


'''
#parameters if integrate == False
theta = 1./60./180.*np.pi # radians = arcmin/60/180*pi
z = np.linspace(2.91,5.1,100)
k = np.logspace(-3,2,100)
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
'''

'''
def processBAG(Xin,count):
    val = []
    for i in count:
        pred = bag.predict(Xin[i])
        if pred[0] == 0:
            val.append(i)
    return val
'''




#Integrate using mcmiser
def MCint(theta, integrand, cosmology, dNdz, npoints, count, klow = -4.0, khigh = 1.0, zlow = 2.91, zhigh = 5.15):
	newtheta = theta
	domega = integrand
	oc = cosmology[0]
	ob = cosmology[1]
	H = cosmology[2]
	omegaM = (oc+ob)#/(H/100.)**2
	omegaL= 1.0-omegaM

	print H,omegaM, omegaL, omegaM+omegaL
	print 'begin integration'    
	s= time.time()
	#mcquad(fn,integrand xl=[0.,0.],xu=[1.,1.], lower and upper limits of integration npoints=100000  number of points,args)
	#newtheta = np.logspace(-1.3,2.5,20)
	mclimber = []
	for i in range(len(newtheta)):
		thetas = newtheta[i]
		test = mcmiser(domega, xl=[klow,zlow], xu=[khigh,zhigh], npoints=npoints, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True),nprocs = 4)
		mclimber.append(test[0])
		
	return mclimber


newtheta =  np.logspace(-1.3,2.5,20)
nproc = 1  #Number of processors on which to run
vals = np.arange(len(newtheta))  #an array of integers the length of the data to split for multiprocessing
ev=len(vals)/nproc #Define an equal split of data...NOTE!!!! THIS MUST BE INT DIVISION FOR THIS TO WORK!
sploc=[]
for i in range(nproc-1):
	sploc.append((i+1)*ev)
ct = np.split(vals,sploc)



#Call the Pool command to multiprocess
if __name__ == '__main__':
	p = Pool(nproc)
	func = partial(MCint, newtheta, domega, cosmo, dNdz, 1e3)
	int = p.map(func, ct) #Map the random_loop routine to each processor with a single array in ct to run over


print int


'''
latest runs:
##Allz model
mcmiser(domega, xl=[-4.0,2.91], xu=[1.0,5.15], npoints=5e4, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))

[0.0022200027263388844, 0.0021083064888461562, 0.0018467475863396263, 0.0014342192153634847, 0.001141302503987034, 0.0010579793518872139, 0.00091899832049186425, 0.00073453119035888227, 0.00058345951998180017, 0.00043836510368147147, 0.00028878694417141306, 0.0001758557419443639, 9.7841320226084067e-05, 4.6472244769074968e-05, 1.0978963947506515e-05, 3.7284308615191285e-06, -4.9169059752185104e-06, -4.8756992517851984e-06, 3.3819373489096673e-07, -2.6877690505571225e-06]

##Lowz Model
mcmiser(domega, xl=[-4.0,2.91], xu=[1.0,3.39], npoints=1e4, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))

[0.0050141695829086395, 0.0047247497919663929, 0.004107606040191919, 0.0031743960785829322, 0.0025213933945704261, 0.002321023537877059, 0.0020075074778027001, 0.0016238288515034115, 0.001299353059799276, 0.00092267941013242068, 0.00066212327173974131, 0.00037116355664063911, 0.00020990846383551042, 8.4628699051503305e-05, 2.6627785834102073e-05, 2.4337199080482871e-06, -1.3200134692599123e-05, 4.7733567463223787e-07, -1.8895459264216058e-06, 1.0722363064538035e-07]


##Highz Model


'''


