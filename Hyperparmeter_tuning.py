# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:22:24 2023

@author: Valdemar
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import laplacian_kernel
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import functools

class KNN:
    def __init__(self, X, Y, k):
        self.comp_x = X
        self.comp_y = Y
        self.k = k
        self.n_classes = np.max(Y) + 1 #Assuming classes structured as {i}_i=0 ^N-1
    
    def dist(self, x1,x2):
        return np.dot(x1-x2,x1-x2)
    
    def predict(self, X):
        predictions = []
        i=1
        for x in X:
            distances = []
            l = []
            for x_known, y in zip(self.comp_x, self.comp_y):
                d = self.dist(x,x_known)
                distances.append(d)
                l.append(y)
            idx = np.argsort(distances)
            l = np.array(l)
            l = labels[idx]
            l = l[:self.k]
            prediction_weights = [0]*self.n_classes
            for y in l:
                prediction_weights[y] += 1
            predictions.append(np.argmax(prediction_weights))
        return predictions
    
    def score(self, X, Y):
        Y_hat = self.predict(X)
        corr=0
        tot = Y.shape[0]
        for yhat, y in zip(Y_hat, Y):
            if y == yhat:
                corr += 1
        return corr/tot
    
data = pd.read_csv("project_train.csv")
features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].to_numpy()
labels = data['Label'].to_numpy()
parameters = []
accs_per_set = []
features, X_test, labels, y_test = train_test_split(features, labels, test_size = .2)


for i in range(10):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size = .2)
    criterions = ["gini", "entropy", "log_loss"]
    accs_tot = []
    opt_leafs = []
    opt_splits = []
    opt_depths = []
    opt_estimators = []
    
    for crit in criterions:
        nests = np.arange(10, 500, 20)
        accs = []
        for n in nests:
            clf = RandomForestClassifier(n_estimators = n,criterion = crit, random_state=42)
            clf.fit(X_train, y_train)       
            accs.append(clf.score(X_val, y_val))
        
        opt_estimators.append((nests)[np.argmax(accs)])
        
        depths = np.arange(2, 40, 2)
        accs = []
        for depth in depths:
            clf = RandomForestClassifier(n_estimators = opt_estimators[-1],criterion=crit, max_depth = depth,random_state=42)
            clf.fit(X_train, y_train)
            accs.append(clf.score(X_val, y_val))
        
        opt_depths.append((depths)[np.argmax(accs)])
        
        splits = np.arange(2, 20, 1)
        accs = []
        for split in splits:
            clf = RandomForestClassifier(n_estimators = opt_estimators[-1],criterion=crit, max_depth = opt_depths[-1],min_samples_split = split,random_state=42)
            clf.fit(X_train, y_train)
            accs.append(clf.score(X_val, y_val))
        
        opt_splits.append((splits)[np.argmax(accs)])
        
        leafs = np.arange(1, 30, 2)
        accs = []
        for leaf in leafs:
            clf = RandomForestClassifier(n_estimators = opt_estimators[-1],max_depth = opt_depths[-1], min_samples_split = opt_splits[-1], min_samples_leaf=leaf,criterion = crit, random_state=42)
            clf.fit(X_train, y_train)
            accs.append(clf.score(X_val, y_val))
        
        opt_leafs.append((leafs)[np.argmax(accs)])        
        accs_tot.append(np.max(accs))
    parameters.append((opt_estimators[np.argmax(accs_tot)], opt_depths[np.argmax(accs_tot)], opt_splits[np.argmax(accs_tot)], opt_leafs[np.argmax(accs_tot)], criterions[np.argmax(accs_tot)]))
    accs_per_set.append(np.max(accs_tot))
print(f"Possible parameters for random forest: {parameters}")
opt_parameters = []
majority_criterion = {"gini":0, "entropy":0, "log_loss":0}
for p in parameters:
    majority_criterion[p[4]] += 1
opt_criterion = max(majority_criterion, key=majority_criterion.get)
for p in parameters:
    if p[4] == opt_criterion:
        opt_parameters.append(p[:4])
opt_parameters_rf = np.mean(opt_parameters, axis=0).astype("int")
print(f"Mean value over parameters for Random Forest gives: {opt_parameters_rf}")
print(f"Most popular criterion: {opt_criterion}")

accs_per_set = []
parameters = []
for i in range(10):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size = .2)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    C = np.arange(1, 10, .1)
    acc = []
    for c in C:
        clf = SVC(C=c, kernel = laplacian_kernel)
        clf.fit(X_train, y_train)
        acc.append(clf.score(X_val, y_val))
    c_opt = C[np.argmax(acc)]
    acc = []
    gammas = np.arange(1e-3,1, 1e-3)
    for g in gammas:
        clf = SVC(C=c_opt,gamma=g,kernel=functools.partial(laplacian_kernel, gamma=g))
        clf.fit(X_train, y_train)
        acc.append(clf.score(X_val, y_val))
    parameters.append((c_opt, gammas[np.argmax(acc)]))
    accs_per_set.append(np.max(acc))
print(f"Possible parameters for SVM: {parameters}")
print(f"Mean value over parameters for SVM: {np.mean(parameters, axis=0)}")
opt_parameters_svm = np.mean(parameters, axis=0)

accs_per_set = []
parameters = []
for i in range(10):
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size = .2)
    Ks = np.arange(2, 30, 1)
    acc = []
    for k in Ks:
        clf = KNN(X_train, y_train, k)
        acc.append(clf.score(X_test, y_test))
    parameters.append(Ks[np.argmax(acc)])
    accs_per_set.append(np.argmax(acc))
print(f"Possible Ks for KNN: {parameters}")
print(f"Mean value of K: {np.round(np.mean(parameters, axis=0))}")
opt_k = int(np.round(np.mean(parameters, axis=0)))

#Test on artificial test set
clf = RandomForestClassifier(n_estimators = opt_parameters_rf[0],max_depth = opt_parameters_rf[1], min_samples_split = opt_parameters_rf[2], min_samples_leaf=opt_parameters_rf[3],criterion = opt_criterion, random_state=42)
clf.fit(features, labels)
print(f"Accuracy for optimized random forest: {clf.score(X_test, y_test)}")

clf = SVC(C=opt_parameters_svm[0], kernel=functools.partial(laplacian_kernel, gamma=opt_parameters_svm[1]))
clf.fit(features, labels)
print(f"Accuracy for optimized SVM: {clf.score(X_test, y_test)}")

clf = KNN(features, labels, opt_k)
print(f"Accuracy for optimized KNN: {clf.score(X_test, y_test)}")
    