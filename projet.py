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
    
    
def myLDA(C0,C1,lamb=-1):
    n0=len(C0)
    n1=len(C1)
   
    
    if(lamb>=0 and lamb<=1):
        sigma0=np.cov(zip(*C0))
        sigma1=np.cov(zip(*C1))
        
        sigma=lamb*np.add(np.dot(n0/(n0+n1),sigma0),np.dot(n1/(n0+n1),sigma1))+(1-lamb)*np.identity(2)
    else:
        xApp,yApp=zip(*np.concatenate((C0,C1)))
        sigma=np.cov(xApp,yApp)
    
   
    mu0=np.mean(C0,axis=0)
    mu1=np.mean(C1,axis=0)
    
    
    pi0=n0/(n0+n1)
    pi1=n1/(n0+n1)
    
    mu0T=np.transpose(mu0)
    mu1T=np.transpose(mu1)
    sigmaInv=np.linalg.inv(sigma)
    
    w=np.dot(sigmaInv,np.subtract(mu0,mu1))
    b=np.dot(-1/2,np.dot(np.transpose(np.subtract(mu0,mu1)),np.dot(sigmaInv,np.add(mu0,mu1))))+np.log(pi0/pi1)
    
    
    return w,b
    

        
#apply a LDA from w and b params and return the class of x

def applyLDA(w,b,x):
    xT=np.transpose(x)
    decision=np.add(np.dot(xT,w),b)
    
    if decision>0:
        return 0
    else:
        return 1


# return the accuracy of good prediction    
def testMyLDA(C0App,C1App,C0Test,C1Test,lamb=-1):
    w,b=myLDA(C0App,C1App)
    goodPrediction=0
    for i in C0Test:   
        if applyLDA(w,b,i)==0:
            goodPrediction+=1
    for i in C1Test:
        if applyLDA(w,b,i)==1 :
            goodPrediction+=1
    
    return goodPrediction/(len(C0Test)+len(C1Test))
    
# return the accuracy of good prediction    
def testMyGeneralizedLDA(C0App,C1App,C0Test,C1Test):
    w,b=myLDA(C0App,C1App)
    goodPrediction=0
    for i in C0Test:   
        if applyLDA(w,b,i)==0:
            goodPrediction+=1
    for i in C1Test:
        if applyLDA(w,b,i)==1 :
            goodPrediction+=1
    
    return goodPrediction/(len(C0Test)+len(C1Test))

# generate  two arrays of x data from two covariance and two mean
def generateData(x,sigma0,sigma1,mu0,mu1):
    C0App=np.random.multivariate_normal(mu0, sigma0,x) 
    C1App=np.random.multivariate_normal(mu1, sigma1,x)
    
    return C0App,C1App
    
#for question 4 test myLda with different cov (en modifiant la variance)
def testMyLDAwithVaryingCov():
    result=[]
    variances=[]
    sigma=np.array([[1,1/2],[1/2,1]])
    mu0=np.array([0,0])
   
    mu1=np.array([3,2])
    C0AppBis=np.random.multivariate_normal(mu0, sigma,10) 
    C0Test=np.random.multivariate_normal(mu0, sigma,1000) 

    for i in range(0,100):
        
        variances.append(i/10)
        sigma2=np.array([[1+i/10,1/2],[1/2,1+i/10]])
               
        C1AppBis=np.random.multivariate_normal(mu1, sigma2,10) 
        C1Test=np.random.multivariate_normal(mu1, sigma2,1000) 
        result.append(testMyLDA(C0AppBis,C1AppBis,C0Test,C1Test))
    
    return result,variances
        
        


    
###############################
    
#initial Data

#########################


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
print("the rate of good answer for my LDA on apprendiship data")
print(testMyLDA(C0App,C1App,C0App,C1App))
print("the rate of good answer for my LDA on test data")
print(testMyLDA(C0App,C1App,C0Test,C1Test))

#compare our LDA with sklearn for appData
classApp=np.concatenate(([0]*len(C0App),[1]*len(C1App)))

classifier=LDA()
classifier.fit(appData,classApp)

print("the rate of good answer with sklearn on app data")
print(classifier.score(appData,classApp))

#compare our LDA with sklearn for testData

classTest=np.concatenate(([0]*len(C0Test),[1]*len(C1Test)))

print("the rate of good answer with sklearn on test data")
print(classifier.score(testData,classTest))

#test MyLDA with a new C0App (replace first value by [-10,-10]

C0AppBis=np.copy(C0App)
C0AppBis[0]=np.array([-10,-10])

print("data (modified) App with myLDA:")
print(testMyLDA(C0AppBis,C1App,C0AppBis,C1App))

appDataBis=np.concatenate((C0AppBis,C1App))
print("data test with myLDA:")
print(testMyLDA(C0AppBis,C1App,C0Test,C1Test))


classifierBis=LDA()
classifierBis.fit(appDataBis,classApp)
print("rate of good prediction with sklearn")
print(classifierBis.score(appDataBis,classApp))
print("rate of good prediction with sklearn:")
print(classifierBis.score(testData,classTest))


w,b=myLDA(C0App,C1App)
easyPointsX=np.transpose([0,-b/w[0]])
easyPointsY=np.transpose([-b/w[1],0])
print(easyPointsX)
print(easyPointsY)
plt.plot(easyPointsX, easyPointsY, 'y-')
plt.show



print(testMyLDAwithVaryingCov())
plt.figure("bonjour")
x,y=testMyLDAwithVaryingCov()
plt.plot(y,x)

plt.show() 

