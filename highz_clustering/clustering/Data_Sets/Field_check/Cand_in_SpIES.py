from astropy.io import fits as pf
import numpy as np


dat = open('Test_newunif.txt','r')


dat.readline()

RA=[]
DEC=[]
ID = []

for i in dat.readlines():
	print i.split(), len(i.split())
	if len(i.split()) > 2:
		RA.append(float(i.split()[0]))
		DEC.append(float(i.split()[1]))
		ID.append(float(i.split()[2]))
	else:
		continue
		


tbhdu=pf.BinTableHDU.from_columns([pf.Column(name='RA',format='D',array=np.asarray(RA)), pf.Column(name='DEC',format='D',array=np.asarray(DEC)),pf.Column(name='ID',format='D',array=np.asarray(ID))])
		

prihdr=pf.Header()
prihdr['COMMENT']="Master QSO's in the SpIES mask"
prihdu=pf.PrimaryHDU(header=prihdr)
	
hdulist=pf.HDUList([prihdu,tbhdu])
hdulist.writeto('Test_newunif.fits')
