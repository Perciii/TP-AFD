#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score 


def displayData(C0, C1,figure=""):
    x0,y0=zip(*C0)
    plt.figure(figure)
    plt.scatter(x0,y0,c='red')
    
    x1,y1=zip(*C1)
    plt.scatter(x1,y1,c='blue')
    
    plt.title('Nuage de points')         

    plt.show()
    
    
def myLDA(C0,C1,x):
    n0=len(C0)
    n1=len(C1)
    xApp,yApp=zip(*np.concatenate((C0,C1)))
   
    sigma=np.cov(xApp,yApp)
   
    mu0=np.mean(C0,axis=0)
    mu1=np.mean(C1,axis=0)
    
    
    pi0=n0/(n0+n1)
    pi1=n1/(n0+n1)
    xT=np.transpose(x)
    mu0T=np.transpose(mu0)
    mu1T=np.transpose(mu1)
    sigmaInv=np.linalg.inv(sigma)
    
    w=np.dot(sigmaInv,np.subtract(mu0,mu1))
    b=np.dot(1/2,np.dot(np.transpose(np.subtract(mu0,mu1)),np.dot(sigmaInv,np.add(mu0,mu1))))+np.log(pi0/pi1)
    decision=np.subtract(np.dot(xT,w),b)
    
    if(decision>0):
        return 0
    else:
    
        return 1
        



# return the accuracy of good prediction    
def testMyLDA(C0App,C1App,C0Test,C1Test):
    goodPrediction=0
    for i in C0Test:   
        if myLDA(C0App,C1App,i)==0:
            goodPrediction+=1
    for i in C1Test:
        if myLDA(C0App,C1App,i)==1 :
            goodPrediction+=1
    
    return goodPrediction/(len(C0Test)+len(C1Test))
    
    
#initial Data


mu0=np.array([0,0])
mu1=np.array([3,2])
sigma=np.array([[1,1/2],[1/2,1]])


 
#generate data for apprendiship    
np.random.seed(1)
C0App=np.random.multivariate_normal(mu0, sigma,10) 
C1App=np.random.multivariate_normal(mu1, sigma,10)

appData=np.concatenate((C0App,C1App))


#generate data for test
C0Test=np.random.multivariate_normal(mu0, sigma,1000) 
C1Test=np.random.multivariate_normal(mu1, sigma,1000)


testData=np.concatenate((C0Test,C1Test))

#display appData
displayData(C0App,C1App,"app")


#display testData
displayData(C0Test,C1Test,"test")

#print(testMyLDA(C0App,C1App,C0App,C1App))
print(testMyLDA(C0App,C1App,C0Test,C1Test))

#compare our LDA with sklearn for appData
classApp=np.concatenate(([0]*len(C0App),[1]*len(C1App)))

classifier=LDA()
classifier.fit(appData,classApp)

#print(classifier.score(appData,classApp))

#compare our LDA with sklearn for testData

classTest=np.concatenate(([0]*len(C0Test),[1]*len(C1Test)))

classifier2=LDA()
classifier2.fit(appData,classApp)
print(classifier2.score(testData,classTest))
x,y=zip(*appData)




