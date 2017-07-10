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

#Import the Adam's data as a lowz test
#data = '/Users/johntimlin/Clustering/Myers2006/Myers2006_dr1_test.fits'
#obs = pf.open(data)[1].data
#gdx = ((obs.ZSPEC >= 0.4)&(obs.ZSPEC <= 3))
#gdx = ((obs.zphot <= 2.1) & (obs.zphot >= 0.4) & (obs.ZSPEC > 0))
#Compute the redshift percentiles for the Friedmann-Diaconis rule for bin width
#q75, q25 = np.percentile(obs.ZSPEC[gdx], [75 ,25])
#iqr = q75 - q25
#FD = 2*iqr /(len(obs.ZSPEC[gdx]))**(1/3.0)
#Set up the bin range from using the Friedmann Diaconis bin width
#bins = np.arange(min(obs.ZSPEC[gdx]),max(obs.ZSPEC[gdx]),FD)
#num,bins = np.histogram(obs.ZSPEC[gdx],bins,normed=True)

#Import SpIES / SHELA data
data = '../Data_Sets/QSO_Candidates_allcuts_with_errors_visualinsp.fits'
obs = pf.open(data)[1].data
Z = obs.zphotNW
gdx = ((Z >= 3.4)&(Z <= 5.2) & (obs.Good_obj == 0)) & (obs.dec>=-1.2) & (obs.dec<=1.2)
#gdx = Z>0
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
PK = camb.get_matter_power_interpolator(pars,zs = ze,zmax = ze[-1], nonlinear=True, hubble_units=False, k_hunit=True, kmax = ka[-1])
#Generate the power using the interpolator and the z and k arrays
#Power = PK.P(z,k)


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

#Integrate using mcmiser

omegaM = (oc+ob)#/(H/100.)**2
omegaL= 1.0-omegaM

print H,omegaM, omegaL, omegaM+omegaL
print 'begin integration'    
s= time.time()
#mcquad(fn,integrand xl=[0.,0.],xu=[1.,1.], lower and upper limits of integration npoints=100000  number of points,args)
newtheta = np.logspace(-1.3,2.5,20)
mclimber = []
for i in range(len(newtheta)):
	thetas = newtheta[i]
	test = mcmiser(domega, xl=[-4.0,3.41], xu=[1.0,5.1], npoints=1e3, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))
	mclimber.append(test[0])
	
print mclimber
e=time.time()

print e-s



'''
latest run:

mcmiser(domega, xl=[-3.0,3.45], xu=[2.0,5.0], npoints=1e6, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))

[0.0018763493756045195, 0.0015591052537067829, 0.0013261541719343291, 0.0011664782432483816, 0.0010404309744665909, 0.00091741906231659518, 0.00078667114128277277, 0.00064789973106323866, 0.0005049509301372051, 0.00036797906601997838, 0.00024422862731641093, 0.00014404571216926446, 7.2933582496721974e-05, 2.9223826003039019e-05, 7.8230852216102688e-06, 2.9890491694937377e-06, -2.307437559147607e-06, -9.1226385750823894e-07, -3.9755941765663542e-07, 1.3928717601483434e-08]

141181.353475 s 

new candidates, 10**3points
[0.0019430243038571534, 0.0016349397131697643, 0.0015559643190088466, 0.0011592312843893796, 0.001045982603488736, 0.00095526409517522886, 0.00093113611560497887, 0.0005889401612489372, 0.00053144714843557936, 0.00038853567370124737, 0.00025666765171879647, 0.00016544957819145055, 9.8265639739552113e-05, 3.3731282373988794e-05, 8.4752026179249433e-06, 2.2529810568760694e-06, 9.1571876941527249e-06, -7.5021177212707544e-07, 4.2410939833994758e-06, 3.9566810872630768e-06]


shen candidates: newtheta = np.logspace(-1.3,2.5,20)
mcmiser(domega, xl=[-3.0,2.91], xu=[2.0,5.17], npoints=1e6, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))

[0.0018358807532616219, 0.0015034895403743954, 0.0012276746859320596, 0.0010369278499846939, 0.00090159800775010729, 0.00078828444848061288, 0.00067568885621950841, 0.00055784990591065565, 0.00043864978763109299, 0.00032197840731266829, 0.00021621673957789532, 0.0001293993054038773, 6.6678330899456382e-05, 2.7563877682033188e-05, 7.9067731028462201e-06, 2.9283435400988902e-06, -2.2004904973685094e-06, -8.6505180997999433e-07, -3.2480646807619417e-07, 7.9393559384844712e-08]

10^3 points changing to z = 5.1 on the upper limit of the integral
[0.0031072154820804372, 0.0024773982340020656, 0.0022069831939996406, 0.0018249231279954346, 0.0016802887745281424, 0.0014562986726930265, 0.0012651874250202608, 0.0010400616665105426, 0.00080494654363068101, 0.0005982830063258948, 0.00038513590577395919, 0.00026714928016567424, 0.00014338341872873156, 4.9450665812679637e-05, 1.782600514223763e-05, 5.0932795884699636e-06, 1.4925594883012705e-05, -4.9547953675508698e-06, 4.5836346273833925e-06, 3.8992113097562235e-06]




[0.0022529517590044938, 0.0021280436475443168, 0.0018731349354724374, 0.0015033947078234348, 0.0013070363461209996, 0.0011766368407685472, 0.0010140398851336263, 0.00083560744525899085, 0.00065508092975343803, 0.00047837963299693522, 0.0003187878517245088, 0.0001885268241591017, 9.5947162744095565e-05, 3.876918723162803e-05, 1.0394048795964464e-05, 3.7806009976488573e-06, -3.1705205023784285e-06, -1.3079909197198175e-06, -1.1571078800356848e-06, 2.4679274288594045e-07]

'''


