from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Read in training file
#data = Table.read('GTR-ADM-QSO-ir-testhighz_findbw_lup_2016_starclean.fits')

#My path to the Training file
#data = Table.read('../Training_set/GTR-ADM-QSO-ir-testhighz_findbw_lup_2016_starclean.fits')
#data = Table.read('../Training_set/GTR-ADM-QSO-ir-testhighz_findbw_lup_2016_starclean_with_shenlabel.fits')

data = Table.read('../Training_set/GTR-ADM-QSO-Trainingset-with-McGreer-VVDS-DR12Q_splitlabel_VCVcut_best.fits')
#data = data.filled()

Xtrain = np.vstack([np.asarray(data[name]) for name in ['ug', 'gr', 'ri', 'iz', 'zs1', 's1s2']]).T

#Xtrain = np.vstack([ data['ug'], data['gr'], data['ri'], data['iz'], data['zs1'], data['s1s2']]).T
#ytrain = np.array(data['labels'])
ytrain = np.array(data['lzlabel'])

# Read in test file
#data2 = Table.read('GTR-ADM-QSO-ir_good_test_2016.fits')

#My path to the test file
data2 = Table.read('../Final_S82_candidates_full/GTR-ADM-QSO-ir_good_test_2016_out_Stripe82all.fits')

##USE BELOW FILE FOR COMPLETENESS MEASURE
#data2 = Table.read('./Add_to_Timlin17_Candidates.fits')

print 'read'
# Limit test file to Stripe 82 area
ramask = ( ( (data2['ra']>=300.0) & (data2['ra']<=360.0) ) | ( (data2['ra']>=0.0) & (data2['ra']<=60.0) ) )
decmask = ((data2['dec']>=-1.5) & (data2['dec']<=1.5))
dataS82 = data2[ramask & decmask]
#dataS82 = dataS82.filled()


# Create test input for sklearn
#Xtest = np.vstack([ dataS82['ug'], dataS82['gr'], dataS82['ri'], dataS82['iz'], dataS82['zs1'], dataS82['s1s2']]).T
Xtest = np.vstack([np.asarray(dataS82[name]) for name in ['ug', 'gr', 'ri', 'iz', 'zs1', 's1s2']]).T

print 'Scaling'
# "Whiten" the data (both test and training)
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(Xtrain)  # Use the full training set now
XStrain = scaler.transform(Xtrain)
XStest = scaler.transform(Xtest)

print 'Begin RF train'
# Instantiate the RF classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, max_depth=15, min_samples_split=2, n_jobs=4, random_state=42)
rfc.fit(XStrain, ytrain)

print 'Begin RF predict'
# Determine RF classifications for the test set
ypredRFC = rfc.predict(XStest)

print 'Begin SVM train'
# Instantiate the SVM classifier
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(XStrain,ytrain)

# Use dask to determing the SVM classifications for the test set
print 'Begin SVM predict'
## JT's version of SVM by multiprocessing
from multiprocessing import Pool
from functools import partial

####MUST RESHAPE THE TEST SET DATA
XStest_reshape = [x.reshape(1,-1) for x in XStest]

def processSVM(Xin,count):
    val = []
    for i in count:
        pred = svm.predict(Xin[i])
        if pred[0] ==0:
            val.append(i)
    return val

nproc = 12  #Number of processors on which to run
vals = np.arange(len(XStest_reshape))  #an array of integers the length of the data to split for multiprocessing
ev=len(vals)/nproc #Define an equal split of data...NOTE!!!! THIS MUST BE INT DIVISION FOR THIS TO WORK!
sploc=[]
for i in range(nproc-1):
	sploc.append((i+1)*ev)
ct = np.split(vals,sploc)


#Call the Pool command to multiprocess
if __name__ == '__main__':
	p = Pool(nproc)
	func = partial(processSVM,XStest_reshape)
	svmpredict = p.map(func, ct) #Map the random_loop routine to each processor with a single array in ct to run over
	ypredSVM = np.ones((len(XStest_reshape)))
	idx = np.hstack(svmpredict)
	for i in idx:
		ypredSVM[i] = 0


'''
from dask import compute, delayed
import dask.threaded
import dask.multiprocessing
#Gordons version of SVM using dask
def processSVM2(Xin):
    return svm.predict(Xin)

# Create dask objects
dobjsSVM = [delayed(processSVM2)(x.reshape(1,-1)) for x in XStest]

# Actually determine the SVM classifications
ypredSVM = compute(*dobjsSVM, get=dask.threaded.get)

# Reformat the SVM classification output.
ypredSVM = np.array(ypredSVM).reshape(1,-1)[0]

gdx = (ypredSVM == 0)

print np.asarray(range(len(ypredSVM)))[gdx]
'''

print 'Begin Bagging train'

# Instantiate the bagging classifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bag = BaggingClassifier(KNeighborsClassifier(n_neighbors=5), max_samples=0.5, max_features=1.0, n_jobs=1)
bag.fit(XStrain, ytrain)
print 'Begin Bagging predict'
'''
start1 = time()

from dask import compute, delayed
import dask.threaded
import dask.multiprocessing
# Use dask to determing the bagging classifications for the test set
def processBAG(Xin):
    return bag.predict(Xin)

# Create dask objects
dobjsBAG = [delayed(processBAG)(x.reshape(1,-1)) for x in XStest]

# Actually determine the bagging classifications
gypredBAG = compute(*dobjsBAG, get=dask.multiprocessing.get)

# Reformat the bagging classification output.
gypredBAG = np.array(gypredBAG).reshape(1,-1)[0]


bdx = (gypredBAG == 0)

print np.asarray(range(len(gypredBAG)))[bdx]
end1 = time()
print 'GTR way=',end1-start1
'''
start2 = time()

### JT WAY OF MULITPROCESSING
def processBAG(Xin,count):
    val = []
    for i in count:
        pred = bag.predict(Xin[i])
        if pred[0] == 0:
            val.append(i)
    return val
nproc = 12  #Number of processors on which to run
vals = np.arange(len(XStest_reshape))  #an array of integers the length of the data to split for multiprocessing
ev=len(vals)/nproc #Define an equal split of data...NOTE!!!! THIS MUST BE INT DIVISION FOR THIS TO WORK!
sploc=[]
for i in range(nproc-1):
	sploc.append((i+1)*ev)
ct = np.split(vals,sploc)


#Call the Pool command to multiprocess
if __name__ == '__main__':
	p = Pool(nproc)
	func = partial(processBAG,XStest_reshape)
	bagpredict = p.map(func, ct) #Map the random_loop routine to each processor with a single array in ct to run over
	ypredBAG = np.ones((len(XStest_reshape)))
	bagidx = np.hstack(bagpredict)
	for i in bagidx:
		ypredBAG[i] = 0


end2=time()

print 'JT way=', end2-start2

print 'Writing'
# Add classifications to data array and write output file
dataS82['ypredRFC'] = ypredRFC
dataS82['ypredSVM'] = ypredSVM
dataS82['ypredBAG'] = ypredBAG
dataS82.write('Lzall_novcvtrain.fits', format='fits')
