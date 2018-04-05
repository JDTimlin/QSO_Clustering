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
#file1='/Users/johntimlin/Catalogs/QSO_candidates/photoz/jupyter_notebooks/QSO_Candidates_40prob_maxz.fits'
file1='../Data_Sets/HZLZ_combined_all_hzclassifiers_wphotoz_zspecflg.fits'
#file1='../Data_Sets/Test_good_data.fits'

#file1 = './HZLZ_bright_faint_zmatched.fits'

file2='../Data_Sets/Short_randoms_spsh.fits'
#file2='../Data_Sets/Test_good_rands.fits'

#file2 = '../Data_Sets/rand_obj_newset.fits'

#Names of the output files
fullsamp = "./Final_Clustering_All.txt"
jackknife = "./Final_Clustering_All_JK.txt"


#Open the data with fits
data=pf.open(file1)[1].data
data2=pf.open(file2)[1].data
# Separate the data and random points and stack so that I have an array of [x,y,z] coordinates for each quasar


#fdx = ((data.ra>=344.1) | (data.ra < 330)) &  (data.Ag<=0.21) # & (data.zphotNW>=2.9) & (data.zphotNW<=5.4)

#classification = (data.ypredBAG+data.ypredSVM+data.ypredRFC)



#dx = ((data.zphotNW>=2.9) & (data.zphotNW<=5.4) & (data.Good_obj == 0) & (data.dec>=-1.2) & (data.dec<=1.2) &  (data['imag']>=20.2)) & ((fdx) | (classification < 2))

magi = -2.5/np.log(10) * (np.arcsinh((data.iflux/1e9)/(2*1.8e-10))+np.log(1.8e-10))
print magi

#Restrict to only the 'good' Data
dx = (data.dec>=-1.2) & (data.dec<=1.2) & (data.zbest>=2.9) & ((data.ra>=344.4)|(data.ra<60)) #& (magi>=20.2)

datra = data.ra[dx]
datdec = data.dec[dx]



datvect = np.concatenate((np.array([datra]).T,np.array([datdec]).T),axis = 1)

#Map the data restriction to the randoms as well
rdx = ((data2.DEC>=-1.2) & (data2.DEC<=1.2)) & ((data2.RA>=344.4)|(data2.RA<60))

randra = data2.RA[rdx]
randdec = data2.DEC[rdx]

randvect = np.concatenate((np.array([randra]).T,np.array([randdec]).T),axis = 1)

print len(randvect),len(datvect)

#Plot the field
#plt.scatter(randra,randdec,s=1,color = 'c')
#plt.scatter(datra,datdec,s=1)

#plt.show()


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
print D
print datvect

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










                     
