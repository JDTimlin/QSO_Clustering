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
gdx = ((Z >= 2.9)&(Z <= 5.2) & (obs.Good_obj == 0)) & (obs.dec>=-1.2) & (obs.dec<=1.2)
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
PK = camb.get_matter_power_interpolator(pars,zs = ze,zmax = ze[-1], nonlinear=True, hubble_units=True, k_hunit=True, kmax = ka[-1])
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
	test = mcmiser(domega, xl=[-4.0,2.91], xu=[1.0,5.1], npoints=1e6, args=(thetas,pars,H,dNdz,PK,omegaM,omegaL,True))
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


#Hardcoding the parameters and making Pk a function of k instead of k/h

[0.0030956793820641412, 0.0027001551429969941, 0.0021353189839345265, 0.001790039722132996, 0.0013612067871075671, 0.0012479792469265196, 0.0011293243751453769, 0.0010201520138515918, 0.00078023155239201141, 0.0006165545753964686, 0.00049494210490758604, 0.00028094430057569984, 0.00011526598607507839, 0.00012602558144464475, 2.822815995276027e-05, -1.5568326725917217e-07, 1.2774478847150886e-05, 2.3142143107741176e-06, -9.4281027585867707e-06, 7.8030145012187476e-06]

#Hardcoding the parameters and making Pk a function of k/h instead of k

[0.0054108478363561379, 0.0044045068478766923, 0.0038189680213347073, 0.0034810972150242153, 0.0025577190360265151, 0.0023227121222531111, 0.0020201086777665586, 0.0017062477714488468, 0.0013214858785841105, 0.0010855411706507374, 0.00064019479752570408, 0.00043256757849160913, 4.9111200306399008e-05, 3.4720619711821213e-05, 3.699043749743926e-05, 2.4799228538414688e-05, -3.6601804281396227e-05, -7.1267878927119301e-06, -1.1604522676434193e-05, 7.091453933877462e-06]

#Hardcoding the parameters and making Pk a function of k instead of k/h 10^4 points

[0.0030956613799786314, 0.0025779964271445601, 0.0021029657234330279, 0.0017208104705162669, 0.0014829017651772074, 0.001291891938700023, 0.0011312136982037958, 0.00096203935433271522, 0.00077351346105401276, 0.0006013410787982414, 0.00045069825791954122, 0.00027639761652180106, 0.00016009130016169398, 8.5148393016386341e-05, 2.8444485933087399e-05, 7.0336114482887848e-06, -4.2714741490801106e-07, -4.9012425561183614e-06, -2.0104411593932462e-06, -1.9040916494153032e-06]

#Hardcoding the parameters and making Pk a function of k instead of k/h 10^5 points

[0.0030871031110961773, 0.002583500912924171, 0.0021039881369522826, 0.0017284335351482937, 0.0014744576140542753, 0.0012848358155449862, 0.0011217350067921229, 0.00095300045064317985, 0.00078222379097313245, 0.00060592408990305839, 0.00043471808346221935, 0.00029212372203551166, 0.00017155591390591307, 8.3468829065576263e-05, 3.2130352344337625e-05, 7.3325936106881484e-06, 1.0606063834311036e-06, -1.0297691089622345e-06, 2.1656385967507094e-07, -2.3980477815096683e-07]

#Hardcoding the parameters and making Pk a function of k instead of k/h 10^6 points

[0.0030904509488304138, 0.0025838375570460337, 0.0021039114087752051, 0.001729659710015563, 0.0014706427586624517, 0.0012859273083670021, 0.00112291761631512, 0.0009574350035407008, 0.00078529262388130137, 0.00060803221716398828, 0.0004392807513616327, 0.00029010795684795911, 0.00016935666860958326, 8.3277492886939307e-05, 3.2492263851320268e-05, 8.0403511554626647e-06, 1.768148727929703e-06, -2.5917300603660223e-06, -5.0718445643122303e-07, -4.5098257814992347e-07]

######
#  NEW RUNS
#######


High-z - 3.4<=z<=5.12 
Integral of dN/nz = 
0.9427463789734734
10^6 points
[0.0021772390344967922, 0.0017981963415267294, 0.0015138723477458136, 0.0013240929523615694, 0.0011788998120912685, 0.001040545230087426, 0.00089445988488793908, 0.00073791556201553894, 0.00057756954058320212, 0.00042075495013100297, 0.00027990070965034242, 0.00016570146848930912, 8.4108479728399395e-05, 3.3515081431901641e-05, 8.5771703413743857e-06, 3.1984295859179773e-06, -2.9496370893133368e-06, -9.1885279782602005e-07, -4.1999550373301511e-07, 2.4053576412830735e-08]

lowz - 2.9<=z<=3.4
Integral of dN/nz = 0.8598222911589789
10^6 points
[0.0044448383465912596, 0.0036227146683768833, 0.0029069235471847099, 0.0024042816344065111, 0.0020639606554351299, 0.0017995581817883723, 0.001544215049633319, 0.0012809174789394379, 0.0010115453784482319, 0.00074573832197666573, 0.00050417339194476236, 0.00030523243537030806, 0.00015850818219234416, 6.676916854321649e-05, 2.010649680773994e-05, 7.6066575051453202e-06, -4.5024669170149373e-06, -2.0881465945823146e-06, -8.7397875471468789e-07, 1.9624113161221976e-07]


allz -  2.9<=z<=5.12 
Integral of dN/nz = 0.9625851508136403
10^6 points

[0.0018901069830166064, 0.0015469541773802007, 0.0012618003172005974, 0.0010646848849014826, 0.00092659609065711736, 0.0008121011076479759, 0.00069717733505905916, 0.00057620099028505515, 0.00045441356648772189, 0.0003334100976949317, 0.00022407216949898401, 0.00013460720655439581, 6.8890660458582111e-05, 2.8945118539565589e-05, 8.5993807832771192e-06, 2.6124134574857943e-06, -2.353262803259482e-06, -1.0507113239217212e-06, -4.7148295416947918e-07, -9.8152287015064401e-08]


#######
##FINAL RUNS Fixed k/h
#######


allz -  2.9<=z<=5.12 
Integral of dN/nz =(0.9632788939076756, 5.1965754854739e-09)
10^6 points

[0.0010484572835645872, 0.00087628995129873937, 0.0007139119401947258, 0.0005873233316463013, 0.00049977023241733881, 0.00043705689938899347, 0.00038169008839822392, 0.00032568170044474358, 0.00026669689684474584, 0.00020721279189903552, 0.00014950192470338821, 9.8161861381187499e-05, 5.6801859334655952e-05, 2.8278037258969553e-05, 1.0868929558997933e-05, 2.9120215105104667e-06, 3.971100085074141e-07, -8.5076328058170249e-07, -1.7364306366937422e-07, -5.601478452769336e-08] 

'''
'''
allz -  2.9<=z<=5.12 
Integral of dN/nz =(0.9836035642677904, 1.3408349275794555e-08)
10^6 points
[0.0011388948127420094, 0.00095220743879588542, 0.00077542750131884774, 0.00063670313475368832, 0.00054079980828579941, 0.00047226359609191438, 0.00041237507773595058, 0.00035190297626006077, 0.00028873129689813566, 0.00022319453928122332, 0.00016096801825516989, 0.00010624811262385081, 6.1690875743870583e-05, 3.0489106558991979e-05, 1.2011847906804716e-05, 2.8349458746808996e-06, 5.9157674753200637e-07, -9.0380548798434433e-07, -8.407675166883347e-08, -9.4363569593502628e-08]
'''




