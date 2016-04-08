import pandas
import pandas as pd
import sklearn.ensemble
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
#matplotlib.get_backend()
fileinp=pandas.read_csv('regressionSet.csv')
sLength=fileinp.shape

fileinp['random']=pandas.Series(np.random.rand(sLength[0]))
    
trainset=fileinp[fileinp['random']<0.8]
testset=fileinp[fileinp['random']>=0.8]

trainlabel=trainset[['zspec']]
trainfeat=trainset[[1,2,3,4,5]]

testlabel=testset[['zspec']]
testfeat=testset[[1,2,3,4,5]]

forest=sklearn.ensemble.RandomForestRegressor(n_estimators=100)
fit=forest.fit(trainfeat,np.ravel(trainlabel))
res=forest.predict(testfeat)

plt.scatter(res,testlabel)
plt.ylabel('z_phot')
plt.xlabel('z_spec')

plt.show()
plt.clf()
plt.cla()
plt.close()

res=np.reshape(res,[res.shape[0],1])

	
dif=res-testlabel
mean=np.mean(dif)
std=np.std(dif)
print mean[0],std[0]
#matr=pandas.DataFrame(res[0],testlabel)
#print res.shape,testlabel.shape
