# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 22:49:02 2018

@author: Asus
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 22:51:56 2018

@author: Asus
"""

from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np
from sklearn.preprocessing import normalize

class ensemble_bayes:
    
    """
    This class is used to aggregate classifiers using the copulas method.
    
    Parameters : 
        clf : a list of classifiers to aggrgate (default value : empty list)
        copula : the copula function used for to estimate joint distribution (takes as parameters an array of the probabilities and the theta parameters)
        theta : the copula parameter
    """
    
    def __init__(self,copula_bis, clf = [], sampling = None, theta = 0,num_ech = 10):
        
        self.clf = clf
        self.sampling = sampling
        self.num_ech= num_ech
        self.copula_bis = copula_bis
        
    def fit(self,X_train, y_train,ind=None):
        """
        Fitting all the classifiers to the train (X_train,y_train) data.
        Returns a list of the classifiers fitted.
        """
        p = []
        i = 0
        for cl in self.clf:
            if (ind == None):
                p.append(cl.fit(X_train,y_train))
            else:
                p.append(cl.fit(X_train[ind[i]],y_train[ind[i]]))
                i+=1
        return p
    
    
    def predict(self,X_test):
        """
        Predicting the labels of the test data (X_test)
        """
        preds = []
        for cl in self.clf:
            preds.append(cl.predict(X_test))
        return preds
    
    
    def get_probas(self,y_true,y_pred):
        """
        Getting the confusion matrix probabilities using the true labels (y_true) and the predicted labels (y_pred)
        Returns a list of each classifier's confusion matrix.
        """
        conf_mx = []
        for i in range(len(y_pred)):
            cm = confusion_matrix(y_true,y_pred[i]) + 1
            conf_mx.append(cm / cm.sum(axis=1))
        return conf_mx


    def prod_int(self,y_pred,probas, props,theta):
        
        """
        Computes the aggregation rule. 
        Takes as parameters the predicted labels (y_pred), the estimated probabilities(probas) and the propotions of the problem classes 
        Returns the labels predicted by the aggregated classifier.
        
        """
        
        """
        Ajuster les probabilités pour que le produit P(D_val/theta,c) ne tend pas vers 0.
        0.25 pour la plus petite proba 
        et 0.75 pour la plus grande
        """
        def adjust(arr):
            new = np.zeros(arr.shape)
            for i in range(len(arr[0])):
                if arr[0,i]<arr[1,i]:
                    new[0,i] = 0.25
                    new[1,i] = 0.75
                else:
                    new[0,i] = 0.75
                    new[1,i] = 0.25
            return new
        
        cumul = []
        m = len(self.clf)
        for item in probas:
            cumul.append(np.cumsum(item,axis =1))
        prob = []
        for item in range(len(set(y_pred[0]))):
            for j in range(len(y_pred[0])):
                p = []
                q = []
                for i in range(len(self.clf)):
                    p.append(cumul[i][y_pred[i][j],item])
                    q.append(probas[i][y_pred[i][j],item])
                prob.append(self.copula_bis(np.array(p),theta))
        probs = [prob[i:i+len(y_pred[0])] for i  in range(0, len(prob), len(y_pred[0]))]
        decision = np.concatenate([probs])
        decision = np.prod(adjust(decision/np.sum(decision,axis=0)),axis=1)
        return decision
    
    
    def decision(self,y_pred,probas, props):
        
        """
        Computes the aggregation rule. 
        Takes as parameters the predicted labels (y_pred), the estimated probabilities(probas) and the propotions of the problem classes 
        Returns the labels predicted by the aggregated classifier.
        
        """
        cumul = []
        m = len(self.clf)
        for item in probas:
            cumul.append(np.cumsum(item,axis =1))
        prob = []
        for item in range(len(set(y_pred[0]))):
            for j in range(len(y_pred[0])):
                p = []
                q = []
                for i in range(len(self.clf)):
                    p.append(cumul[i][y_pred[i][j],item])
                    q.append(probas[i][y_pred[i][j],item])
                prob.append(self.sampling(np.array(p),self.num_ech)[item]*np.prod(np.array(q))*props[item])
        probs = [prob[i:i+len(y_pred[0])] for i  in range(0, len(prob), len(y_pred[0]))]
        decision = np.concatenate([probs])
        cop_result = np.argmax(decision,axis=0)
        return cop_result
    
    def accuracy_score_ens (self,X_test,y_true,y_pred,verbose=False):
        """
        Returns the accuracy score.
        """
        if (verbose == True):
            for cl in self.clf:
                print("accuracy score for the base classifier",cl.score(X_test,y_true))
        return accuracy_score(y_true,y_pred)