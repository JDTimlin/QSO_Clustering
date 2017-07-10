import os
import warnings
import numpy as np
from modified_astroml_code import *
from sklearn.neighbors import BallTree
from astropy.io import fits as pf
import matplotlib.pyplot as plt

#Set up the theta values over which to compute the correlation function
th=np.linspace(np.log10(0.04),np.log10(250),22)
#th=np.linspace(np.log10(0.1),np.log10(350),40)
thetarad=10**th


#Triton path to the data and random files
#file1='../Data_Sets/Candidates_photoz_SpSh_shenzrange_tocluster.fits'
file1='../../data/QSO_Candidates_allcuts_with_errors_and_zspec.fits'
file2='../../data/rand100x_kderf_samp.fits'

#Names of the output files
fullsamp = "./SpSh_angcor_obj2.txt"
jackknife = "./SpSh_angcor_obj2_JK.txt"


#Open the data with fits
data=pf.open(file1)[1].data
data2=pf.open(file2)[1].data
# Separate the data and random points and stack so that I have an array of [x,y,z] coordinates for each quasar


fdx = ((data.ra>=344.1) | (data.ra < 330))&  (data.Ag<=0.21)# & (data.zphotNW>=2.9) & (data.zphotNW<=5.4)

classification = (data.ypredBAG+data.ypredSVM+data.ypredRFC)
imag = 22.5-2.5*np.log10(data.iflux)

#dx = ((data.zphotNW>=2.9) & (data.zphotNW<=5.4)) & ((fdx) |(classification <2)) & (data.Highprob ==1) & (abs(data.zphotNW - data.zPDE)<1) #& (imag>20.2)

dx = ((data.zphotNW>=2.9) & (data.zphotNW< 5.4)) & ((fdx) | (classification < 2)) #& (data.Highprob ==1) & (abs(data.zphotNW - data.zPDE)<0.3) #& (imag>20.2)

datra = data.ra[dx]
datdec = data.dec[dx]

datvect = np.concatenate((np.array([datra]).T,np.array([datdec]).T),axis = 1)

rdx = () #((data2.RA>=344.1) | (data2.RA < 330)) & (data2.DEC>=-1.5) & (data2.DEC<=1.5) # 

randra = data2.RA[rdx]
randdec = data2.DEC[rdx]

randvect = np.concatenate((np.array([randra]).T,np.array([randdec]).T),axis = 1)

print len(randvect),len(datvect)

factor = len(randvect)/float(len(datvect))

dd,rr,dr,corr = two_point_angular(datvect, randvect, thetarad, method='landy-szalay')

for i in range(len(thetarad)-1):
	DD = dd[i]
	DR = dr[i]
	RR = rr[i]
	XI = corr[i]
	#Write/Append to file depending on whether file exists or not
	if os.path.exists(fullsamp):
		infile=open(fullsamp,"a")
		THavg1 = (thetarad[i] + thetarad[i+1])/2.0
		infile.write(str(THavg1)+ ' '+ str(DD)+ ' ' + str(DR) + ' ' + str(RR) + ' '+str(XI)+'\n')
		infile.close()
	else:
		infile=open(fullsamp ,"w")
		infile.write("#Theta DD DR RR XI factor="+str(factor)+" \n")
		THavg1 = (thetarad[i] + thetarad[i+1])/2.0
		infile.write(str(THavg1)+ ' '+ str(DD)+ ' ' + str(DR) + ' ' + str(RR) + ' '+str(XI)+'\n')
		infile.close()
    		


print 'Begin Jackknife'

jknum = 10
D = datvect[datvect[:,0].argsort()]
jksplit = np.arange(len(datvect))  #an array of integers the length of the data to split for multiprocessing
evensplit=len(jksplit)/jknum #Define an equal split of data...NOTE!!!! THIS MUST BE INT DIVISION FOR THIS TO WORK!
splitloc=[]
for i in range(jknum-1):
	splitloc.append((i+1)*evensplit)
jklist = np.split(D,splitloc)

R =randvect[randvect[:,0].argsort()]

print jklist[0][:,0]
print R[:,0][(R[:,0] > 0.0) & (R[:,0] < max(jklist[0][:,0]))]

jkrand = []
jkdat = []
for i in range(len(jklist)):
	if i == 0:
		cutd = (D[:,0] > 0.0) & (D[:,0] <= min(jklist[i+1][:,0]))
		cutr = (R[:,0] > 0.0) & (R[:,0] <= min(jklist[i+1][:,0]))
		dcut = np.invert(cutd)
		rcut = np.invert(cutr)
		darr = D[dcut]
		rarr = R[rcut]
		jkdat.append(darr)
		jkrand.append(rarr)
		print 'first', i
		print len(jkrand[i])/float(len(randvect))
		print len(jkdat[i])/float(len(datvect))
	elif i == jknum-1:
		cutd = (D[:,0] >= min(jklist[i][:,0])) & (D[:,0] < 360.0)
		cutr = (R[:,0] >= min(jklist[i][:,0])) & (R[:,0] < 360.0)
		dcut = np.invert(cutd)
		rcut = np.invert(cutr)
		darr = D[dcut]
		rarr = R[rcut]
		jkdat.append(darr)
		jkrand.append(rarr)
		print 'last', i
		print len(jkrand[i])/float(len(randvect))
		print len(jkdat[i])/float(len(datvect))
	else:
		cutd = (D[:,0] > max(jklist[i-1][:,0])) & (D[:,0] <= min(jklist[i+1][:,0]))
		cutr = (R[:,0] > max(jklist[i-1][:,0])) & (R[:,0] <= min(jklist[i+1][:,0]))
		dcut = np.invert(cutd)
		rcut = np.invert(cutr)
		darr = D[dcut]
		rarr = R[rcut]
		jkdat.append(darr)
		jkrand.append(rarr)
		print 'mid', i
		print len(jkrand[i])/float(len(randvect))
		print len(jklist[i])/float(len(datvect))

'''
for i in range(jknum):
	#plt.subplot(111,projection = 'aitoff')
	plt.scatter(jkrand[i][:,0],jkrand[i][:,1],s = 1,color='g')
	plt.scatter(jkdat[i][:,0],jkdat[i][:,1],s = 1,color='r')
	plt.show()
'''

for i in range(len(jkrand)):
	print len(jkrand[i])
	print len(randvect)
	print len(jkrand[i])/float(len(randvect))
	
for i in range(jknum):

	dvect = jkdat[i]
	rvect = jkrand[i]

	ddjk,rrjk,drjk,corrjk = two_point_angular(dvect, rvect, thetarad, method='landy-szalay')

	for i in range(len(thetarad)-1):
		DD = ddjk[i]
		DR = drjk[i]
		RR = rrjk[i]
		XI = corrjk[i]
		#Write/Append to file depending on whether file exists or not
		if os.path.exists(jackknife):
			infile=open(jackknife ,"a")
			THavg1 = (thetarad[i] + thetarad[i+1])/2.0
			infile.write(str(THavg1)+ ' '+ str(DD)+ ' ' + str(DR) + ' ' + str(RR) + ' '+str(XI)+'\n')
			infile.close()
		else:
			infile=open(jackknife,"w")
			infile.write("#Theta DD DR RR XI factor="+str(factor)+" "+"jknum="+" "+str(jknum)+ "\n")
			THavg1 = (thetarad[i] + thetarad[i+1])/2.0
			infile.write(str(THavg1)+ ' '+ str(DD)+ ' ' + str(DR) + ' ' + str(RR) + ' '+str(XI)+'\n')
			infile.close()
    
	infile=open(jackknife,"a")
	infile.write("STEP_DONE \n")
	infile.close()










                     
