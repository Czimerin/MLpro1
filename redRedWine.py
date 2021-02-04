# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:59:16 2021

@author: nickk
"""
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib



redWine = pd.read_csv("red.csv")
whiteWine = pd.read_csv("white.csv")


redWine["color"] = "R"
whiteWine["color"] ="W"


bothRnW=pd.concat([redWine, whiteWine],axis=0)

bothRnW.head()


redWine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
whiteWine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
bothRnW.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)


data = bothRnW.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
labels = bothRnW.iloc[:,[12]]

xTrain, xTest, yTrain, yTest = train_test_split(data,labels, test_size=0.4, random_state=1 )

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xTrain,np.ravel(yTrain,order='C'))
result = knn.predict(xTest)


print(accuracy_score(yTest, result))
