#!/usr/bin/python3
import numpy as np
def displayData(C0, C1):
    plt.figure(nomGraphique)
    plt.scatter(C0)
    plt.title('Nuage de points')
    plt.grid(True)
   
    
    #label axe des abscisses
    if axeX is None:
        plt.xlabel("X")
    else:
        plt.xlabel(axeX)
    
    #label axe des ordonnées
    if axeY is None:
        plt.ylabel("Y")
    else:
        plt.ylabel(axeY)

    plt.show()
    
C0=5
print(C0)

mu0=np.array([0,0])
mu1=np.array([3,2])
sigma=np.array([[1,1/2],[1/2,1]])


 
#generate data for apprendiship    
np.random.seed(0)
C0App=np.random.multivariate_normal(mu0, sigma,10) 
C1App=np.random.multivariate_normal(mu1, sigma,10)

appData=np.concatenate((C0App,C1App))


#generate data for test

C0Test=np.random.multivariate_normal(mu0, sigma,1000) 
C1Test=np.random.multivariate_normal(mu1, sigma,1000)


testData=np.concatenate((C0Test,C1Test))

displayData(C0App,C1App)

