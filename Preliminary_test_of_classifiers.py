# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:22:24 2023

@author: Valdemar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sklearn stuff
from sklearn.svm import SVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.model_selection import train_test_split

#Ensembles
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#Torch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

#Neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(11, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x= self.activation(self.fc1(x))
        x= self.activation(self.fc2(x))
        y= self.sigmoid(self.fc3(x))
        return y
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    tot_loss = []
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tot_loss.append(loss.item())
        
    return model, np.mean(tot_loss)
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#Kernels
def rbf(x1, x2, gamma=.5):
    exponent_term = -np.dot(x1-x2, x1-x2)*gamma
    return np.exp(exponent_term)

def poly(x1, x2, c=1, d=4):
    return (np.dot(x1,x2) + c)**d

def laplace(x1,x2,sigma=2):
    exponent_term = -np.linalg.norm(x1-x2, ord = 1)/(sigma)
    return np.exp(exponent_term)

def tanh(x1,x2, kappa = 1, c = -1):
    return np.tanh(kappa*np.dot(x1,x2) + c)

#MMSE with kernel trick
class kernel_method_classifier:
    def __init__(self, kernel = "rbf", degree=2, lagrange=0):
        self.kernel = kernel
        if self.kernel == "rbf":
            self.ker = rbf
        elif self.kernel == "poly":
            self.ker = poly
        elif self.kernel == "laplace":
            self.ker = laplace
        elif self.kernel == "tanh":
            self.ker = tanh
            
        self.deg = degree
        self.lagrange = lagrange
    
    def fit(self, X,Y):
        lagrangian = self.lagrange*np.eye(X.shape[0])
        K = np.zeros(lagrangian.shape)
        for n in range(K.shape[0]):
            for m in range(K.shape[1]):
                K[n,m] = self.ker(X[n], X[m])
        inv = np.linalg.inv(K + lagrangian)
        a=[]
        for label in (set(Y.tolist())):
            idx = np.where(Y==label)
            temp_y = -np.ones(Y.shape)
            temp_y[idx] = 1
            a.append(np.dot(inv, temp_y))
        self.X_train = X
        self.a = np.array(a)
    
    def predict(self, X):
        predictions = []
        for n, xn in enumerate(X):
            outputs = []
            k = np.zeros(self.X_train.shape[0])
            for i,xm in enumerate(self.X_train):
                k[i] = self.ker(xn, xm)
            for a_val in self.a:
                outputs.append((np.dot(k, a_val)))
            predictions.append(np.argmax(outputs))
        return np.array(predictions)
    
    def generate_boundary(self):
        X_1 = np.arange(-2.5,2.5,1/20)
        X_2 = np.arange(-2.5,2.5,1/20)
        boundary = {0: [], 1: []}
        
        for x2 in X_2:
            for x1 in X_1:
                outs = []
                x = np.array([x1, x2])
                k = np.zeros(self.X_train.shape[0])
                for i,xm in enumerate(self.X_train):
                    k[i] = self.ker(x, xm)
                for a_val in self.a:
                    outs.append(np.dot(k,a_val))
                boundary[np.argmax(outs)].append(x)
        return boundary
    def score(self, X, Y):
        Y_hat = self.predict(X)
        corr=0
        tot = Y.shape[0]
        for yhat, y in zip(Y_hat, Y):
            if y == yhat:
                corr += 1
        return corr/tot

#KNN
def secondElement(a):
    return a[1]

class KNN:
    def __init__(self, X, Y, k):
        self.k = k
        self.X_known = X
        self.Y = Y
        self.n_classes = len(set(Y.tolist()))
        
    def distance(self, x1, x2):
        return np.dot(x1-x2, x1-x2)
    
    def predict(self, X):
        predictions = []
        for x1 in X:
            distance_class_pair = []
            for y,x2 in zip(self.Y,self.X_known):
                distance_class_pair.append((y,self.distance(x1,x2)))
            distance_class_pair.sort(key=secondElement)
            distance_class_pair = distance_class_pair[:self.k]
            candidates = [0]*self.n_classes
            for y,d in (distance_class_pair):
                candidates[y] += 1
            predictions.append(np.argmax(candidates))
        return predictions
    
    def score(self, X, y):
        correct = 0
        total = 0
        y_pred = self.predict(X)
        for yp, y in zip(y_pred, y):
            total+=1
            if yp == y:
                correct += 1
        return correct/total

#Counts how frequency of classes in a sphere (or any shape depending on norm) around the point to classify
#Classifies to the most frequent class
class sphere_count:
    def __init__(self, X, Y, r):
        self.r= r
        self.X_known = X
        self.Y = Y
        self.n_classes = len(set(Y.tolist()))
        
    def distance(self, x1, x2):
        return np.linalg.norm(x1-x2)
    
    def predict(self, X):
        predictions = []
        for x1 in X:
            class_count = [0,0]
            for xk, yk in zip(self.X_known, self.Y):
                if self.distance(x1,xk) < self.r:
                    class_count[yk] += 1
            predictions.append(np.argmax(class_count))
        return predictions
    
    def score(self, X, y):
        correct = 0
        total = 0
        y_pred = self.predict(X)
        for yp, y in zip(y_pred, y):
            total+=1
            if yp == y:
                correct += 1
        return correct/total

class weighted_neighbour:
    def __init__(self, X, Y):
        self.X_known = X
        self.Y = Y
        self.n_classes = len(set(Y.tolist()))
        
    def distance(self, x1, x2):
        return np.linalg.norm(x1-x2)
    
    def predict(self, X):
        predictions = []
        for x1 in X:
            class_count = [0,0]
            for xk, yk in zip(self.X_known, self.Y):
                if self.distance(x1,xk) == 0:
                    class_count[yk] += 999999999999
                    break
                class_count[yk] += 1/self.distance(x1,xk)
            predictions.append(np.argmax(class_count))
        return predictions
    
    def score(self, X, y):
        correct = 0
        total = 0
        y_pred = self.predict(X)
        for yp, y in zip(y_pred, y):
            total+=1
            if yp == y:
                correct += 1
        return correct/total

#Mixture model for classes
class GMM_classifier:
    def __init__(self, components = 3):
        self.components = components
    
    def fit(self, X, y):
        y_set = set(y.tolist())
        gmms = []
        for c in y_set:
            idx = np.where(y==c)[0]
            x_class = X[idx]
            gmm = BayesianGaussianMixture(n_components=self.components[c], tol=1e-6)
            gmm.fit(x_class)
            gmms.append(gmm)
        self.gmms = gmms
    
    def predict(self, X):
        predictions = []
        for xt in X:
            probas = []
            for gmm in self.gmms:
                probas.append(gmm.score(xt.reshape(1,-1)))
            predictions.append(np.argmax(probas))
        return predictions
    
    def score(self, X, y):
        correct = 0
        total = 0
        y_pred = self.predict(X)
        for yp, y in zip(y_pred, y):
            total+=1
            if yp == y:
                correct += 1
        return correct/total
            

data = pd.read_csv("project_train.csv")
#print(data.axes)
features = data[['danceability', 'energy',
                 'key', 'loudness', 
                 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 
       'liveness', 'valence', 'tempo']].to_numpy()
labels = data['Label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2)

scaling = StandardScaler()
scaling.fit(X_train)
X_train = scaling.transform(X_train)
X_test = scaling.transform(X_test)

#Kernel based classifiers
print("\n\n ####### Kernel based classifiers ####### \n\n")
clf = NuSVC(kernel = laplacian_kernel)
clf.fit(X_train, y_train)
print(f"Support vector machine accuracy: {clf.score(X_test, y_test)}\n")

kernels = ["rbf", "poly", "laplace"]
for kernel in kernels:
    clf = kernel_method_classifier(kernel = kernel, lagrange = 1)
    clf.fit(X_train, y_train)
    print(f"MMSE {kernel} kernel accuracy: {clf.score(X_test, y_test)}\n")

#Ensemble classifiers (bagging)
print("\n\n ####### Ensemble based classifiers ####### \n\n")
clf = RandomForestClassifier(n_estimators = 300)
clf.fit(X_train, y_train)
print(f"Randomforest accuracy: {clf.score(X_test, y_test)}\n")

clf = XGBClassifier(max_depth = 50, n_estimators=1000, reg_lambda = .25, objective = 'binary:logistic', gamma = .01)
clf.fit(X_train, y_train)
print(f"XGBoost accuracy: {clf.score(X_test, y_test)}\n")

clf = CatBoostClassifier(verbose=0, n_estimators=1000)
clf.fit(X_train, y_train)
print(f"catboost accuracy: {clf.score(X_test, y_test)}\n")

#Neighbourhood search classifiers
print("\n\n ####### Neighbourhood search based classifiers ####### \n\n")
clf = KNN(X_train, y_train, 5)
print(f"KNN accuracy: {clf.score(X_test, y_test)}\n")
    
clf = sphere_count(X_train, y_train,5.2)
print(f"Sphere count accuracy: {clf.score(X_test, y_test)}\n")

clf = weighted_neighbour(X_train, y_train)
print(f"Weighted neighbourhood count accuracy: {clf.score(X_test, y_test)}\n")

#Bayesian classifiers
print("\n\n ####### Probability based classifiers ####### \n\n")
clf = LogisticRegressionCV(max_iter = 1000)
clf.fit(X_train, y_train)
print(f"Logistic regression accuracy: {clf.score(X_test, y_test)}\n")

clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
print(f"QDA accuracy: {clf.score(X_test, y_test)} \n")

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
print(f"LDA accuracy: {clf.score(X_test, y_test)}\n")

clf = GaussianProcessClassifier(n_restarts_optimizer = 3)
clf.fit(X_train, y_train)
print(f"GP accuracy: {clf.score(X_test, y_test)}\n")

clf = GMM_classifier(components = [50, 50]) #More components -> more obscure music taste
clf.fit(X_train, y_train)
print(f"GMM accuracy: {clf.score(X_test, y_test)}\n")

#Neural network
y_train_1h = np.zeros((y_train.size, y_train.max() + 1))
y_train_1h[np.arange(y_train.size), y_train] = 1
y_test_1h = np.zeros((y_test.size, y_test.max() + 1))
y_test_1h[np.arange(y_test.size), y_test] = 1

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train_1h, dtype = torch.float32)
y_test_tensor = torch.tensor(y_test_1h, dtype = torch.float32)

trainset = TensorDataset(X_train_tensor, y_train_tensor)
testset = TensorDataset(X_test_tensor, y_test_tensor)
trainloader = DataLoader(trainset, batch_size = 64, shuffle = True)
testloader = DataLoader(testset, batch_size = 64, shuffle = True)

model = NeuralNetwork()
loss_fn = nn.BCELoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-2, 
                              weight_decay=1e-3, momentum_decay=.05)
losses = []
for i in range(100):
    model, l = train_loop(trainloader, model, loss_fn, optimizer)
    losses.append(l)
plt.plot(losses)
plt.show()
test_loop(testloader, model, loss_fn)