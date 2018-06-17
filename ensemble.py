# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:32:44 2018

@author: Mostafa BOUZIANE
"""
from sklearn.metrics import confusion_matrix,accuracy_score
import numpy as np

class ensemble:
    
    """
    This class is used to aggregate classifiers using the copulas method.
    
    Parameters : 
        clf : a list of classifiers to aggrgate (default value : empty list)
        copula : the copula function used for to estimate joint distribution (takes as parameters an array of the probabilities and the theta parameters)
        theta : the copula parameter
    """
    
    def __init__(self, clf = [], copula = None, theta = 1):
        
        self.clf = clf
        self.copula = copula
        self.theta = theta
        
    def fit(self,X_train, y_train):
        """
        Fitting all the classifiers to the train (X_train,y_train) data.
        Returns a list of the classifiers fitted.
        """
        p = []
        for cl in self.clf:
            p.append(cl.fit(X_train,y_train))
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
            cm = confusion_matrix(y_true,y_pred[i])
            conf_mx.append(cm / cm.sum(axis=1))
        return conf_mx
    
    
    
    def decision(self,y_pred,probas, props):
        
        """
        Computes the aggregation rule. 
        Takes as parameters the predicted labels (y_pred), the estimated probabilities(probas) and the propotions of the problem classes 
        Returns the labels predicted by the aggregated classifier.
        
        """
        cumul = []
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
                prob.append(self.copula(np.array(p),self.theta)*np.prod(np.array(q))*props[item])
        probs = [prob[i:i+len(y_pred[0])] for i  in range(0, len(prob), len(y_pred[0]))]
        decision = np.concatenate([probs])
        cop_result = np.argmax(decision,axis=0)
        return cop_result
    
    def accuracy_score_ens (self,y_true,y_pred):
        """
        Returns the accuracy score.
        """
        return accuracy_score(y_true,y_pred)
            
        