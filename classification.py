# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 00:22:24 2023

@author: Valdemar
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def list_to_string(l):
    s=""
    for c in l:
        s+=str(c)
    return s

data = pd.read_csv("project_train.csv")
#print(data.axes)
features = data[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].to_numpy()
labels = data['Label'].to_numpy()

data_test = pd.read_csv("project_test.csv")
test_features = data_test[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].to_numpy()

clf = RandomForestClassifier(n_estimators = 140, max_depth=14,min_samples_split=2, min_samples_leaf=7, criterion = "gini", random_state=42)
clf.fit(features, labels)
preds_rf = clf.predict(test_features)
preds_rf_string = list_to_string(preds_rf)