# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:50:09 2018

@author: Asus
"""

from bayes_bis import ensemble_bayes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import sys
from scipy.stats import norm



def gausscop_pdf(u,theta):
    """
    Computes the combined predictions based on those returned by classifiers.
    
    Parameters
    ----------    
     - u : numpy array containing the conditional cdf of each classifier 
    - theta : the classifier covariance parameter. This parameter lives in the 
    open interval (-1/(len(u)-1);1).
    
    Returns
    -------
    - cop : scalar (copula value)
    """
    invmat = 1/(1-theta) * (np.eye(len(u)) - theta/(1+(len(u)-1)*theta)*np.ones((len(u),len(u))) )
    det = (1+(len(u)-1)*theta)*np.power(1-theta,len(u)-1)
    precision = sys.float_info.epsilon *1e6
    if (len(u[np.where(u>precision)])>0):
        mini = np.min(u[np.where(u>0)])*1e-6 #something a lot smaller than the minimal positive value in u
        mini = max(mini,precision)
    else: # all entries in u are null or negative
        mini = precision
    if (len(u[np.where(u<1-precision)])>0):
        maxi = 1-(1-np.max(u[np.where(u<1)]))*1e-6 #something a lot larger than the maximal value in u that is below 1
        maxi = min(maxi,1-precision)
    else: # all entries in u are 1 or something bigger
        maxi = 1 - precision
    u[np.where(u<=0)] = mini #avoid -inf
    u[np.where(u>=1)] = maxi #avoid +inf     
    if (np.sum(u>=1)):
        print(u)
        raise ValueError('stop')
    v = norm.ppf(u)
    
    #cop = 1/np.sqrt(det) * np.exp(np.dot(-0.5*v, np.dot(invmat-np.eye(len(u)),v) ))
    #print(det)
    log_cop = np.dot(-0.5*v, np.dot(invmat-np.eye(len(u)),v) ) - 0.5*np.log(det)
    return log_cop

"""
Defining the copula function for two random variables
"""
def copule(u,theta,family='Gaussian'):
    """
    FGM copula density for a pair of random variables
    """
    if (family == 'FGM'):
        return 1 + theta*(1-2*u[0])*(1-2*u[1])
    elif (family == 'Gaussian'):
        return np.exp(gausscop_pdf(u,theta))
    else:
        raise ValueError('Unknown copula family.')

def pca_split(X,y,n_class,Ns,inds_in):
    """
    This function splits the dataset into Ns pieces. 
    The split is deterministic. The feature space is divided into Ns regions.
    Each region is separated from another by a hyerplane that is orthogonal
    to the eigenvector number dim returned by a PCA.
    """   
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    u = pca.components_[0].T
    ind_loc = []
    for i in range(n_class):
        ind_class = np.where(y==i)[0]
        X_class = X[ind_class]
        v_loc = np.dot(X_class,u)
        inds = np.argsort(v_loc.flatten())
        width = int(len(v_loc)/Ns)
        for j in range(Ns):
            ind = inds[j*width:(j+1)*width]
            if (i==0):
                ind_loc.append(inds_in[ind_class[ind]])
            else:
                ind_loc[j] = np.hstack((ind_loc[j],inds_in[ind_class[ind]]))
    return ind_loc
"""
fonction qui permet de calculer la Proba(D_val/theta,c)
"""
def produit(theta):
    return ens.prod_int(y_pred1,probas,np.array([0.5,0.5]),theta)

"""
fonction qui permet de sampler selon beta et faire montecarlo
"""
def estim_with_sampling(u,num_ech):
    samples = (beta.rvs(a = 3, b =3 , size = num_ech))*1.5 - 0.5  
    z = np.zeros(num_ech)
    x = np.zeros(num_ech)
    for i in range(num_ech):
        z[i] = copule(u,samples[i])*produit(samples[i])[0]
        x[i] = copule(u,samples[i])*produit(samples[i])[1]
    return (np.mean(z),np.mean(x))

dataset = make_moons(600, noise=0.2, random_state=1995)
X = dataset[0]
y = dataset[1]
n_class = y.max()-y.min()+1

"""
initiate 2 classifiers
"""
clf = []
clf = []
#clf.append( LogisticRegression(random_state=1))
#clf.append( LogisticRegression(random_state=1))
#clf.append( LogisticRegression(random_state=1))
clf.append( LogisticRegression(random_state=1))
#clf.append( LogisticRegression(random_state=1))
clf.append(RandomForestClassifier())
clf.append(KNeighborsClassifier(3))
#clf.append( LinearDiscriminantAnalysis())
#clf.append( QuadraticDiscriminantAnalysis())
Ns = len(clf)

ens = ensemble_bayes(copula_bis = copule, clf = clf,sampling = estim_with_sampling,num_ech = 10)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
"""
Devide the test set in 2 val sets
"""
X_test1 = X_test[:99,:]
y_test1 = y_test[:99]
X_test2 = X_test[99:,:]
y_test2 = y_test[99:]

ind_train = pca_split(X_train,y_train,n_class,Ns,np.arange(y_train.size))
for i in range(Ns):
    plt.figure()
    plt.scatter(X_train[ind_train[i], 0], X_train[ind_train[i], 1], marker='o', c=y_train[ind_train[i]], s=25,cmap='tab10')
    plt.xlim(np.min(X_train[:,0]),np.max(X_train[:,0]))
    plt.ylim(np.min(X_train[:,1]),np.max(X_train[:,1]))
    
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train, s=25,cmap='tab10')
plt.xlim(np.min(X_train[:,0]),np.max(X_train[:,0]))
plt.ylim(np.min(X_train[:,1]),np.max(X_train[:,1]))


ens.fit(X_train,y_train,ind_train)
"""
using the first test set to estimate the probas
"""
y_pred1 = ens.predict(X_test1)
probas = ens.get_probas(y_test1,y_pred1)

"""
Predicting the labels of the second testset using the aggregated classifier
"""
y_pred2 = ens.predict(X_test2)
agg_pred = ens.decision(y_pred2,probas,np.array([0.5,0.5]))
"""
Accuracies of the base classifiers and the aggregated classifier
"""
accuracy = ens.accuracy_score_ens(X_test2,y_test2,agg_pred,True)
print('accuracy of ensemble is:',accuracy)

theta_range = np.linspace(-1/(Ns-1),1,50)
theta_range = theta_range[1:len(theta_range)-1] #eliminating theta=-1
theta_range = np.hstack((theta_range,np.asarray([0.0])))
theta_range.sort()
acc =  np.zeros((len(theta_range,)))
for i in range(len(theta_range)):
    ens.theta=theta_range[i]
    y_pred1 = ens.predict(X_test1)
    probas = ens.get_probas(y_test1,y_pred1)
    y_pred2 = ens.predict(X_test2)
    agg_pred = ens.decision(y_pred2,probas,np.array([0.5,0.5]))
    acc[i] = ens.accuracy_score_ens(X_test2,y_test2,agg_pred)
    
plt.figure()
plt.plot(theta_range,acc)

"""
la performance est mieux quand la prior est concentrée autour de O, O.2 , donc il faut bien
choisir cette prior pour atteindre la performance maximale,
intuition : si theta = 0 est un pic de performance , (independance des classifieurs)
ça ne sert a rien de poser une prior puisque ça va diminuer la performance
idée : si independance , theta = 0 , sinon prior sur theta (tests sur classif dependants)

"""

