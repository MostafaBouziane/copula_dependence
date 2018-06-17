# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 17:10:58 2018

@author: Asus
"""
from ensemble import ensemble
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import make_moons
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.svm import LinearSVC

"""
Defining the copula function for two random variables
"""
def copule(u,theta):
    return 1 + theta*(1-2*u[0])*(1-2*u[1])

dataset = make_moons(10000, noise=0.2, random_state=1995)
X = dataset[0]
y = dataset[1]

"""
initiate 2 classifiers
"""
clf1 = LogisticRegression(random_state=1)
clf2 = LinearDiscriminantAnalysis()

ens = ensemble(clf = [clf1,clf2],copula = copule)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
"""
Devide the test set in 2 val sets
"""
X_test1 = X_test[:1650,:]
y_test1 = y_test[:1650]
X_test2 = X_test[1650:,:]
y_test2 = y_test[1650:]

ens.fit(X_train,y_train)
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
accuracy = ens.accuracy_score_ens(X_test2,y_test2,agg_pred)

