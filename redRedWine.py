# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:59:16 2021

@author: nickk
         Fred Fikter
"""
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cmap_light = ListedColormap(['red', 'gray'])



distanceMetric = ["euclidean", "manhattan", "chebyshev"]
userDistanceMetric = input("Enter number of distacne metric to use: \n 1. euclidean \n 2. manhattan \n 3. chebyshev \n \n")
userDistanceMetric = int(userDistanceMetric)



userTestSize = input("Enter a number between 0 and 1 for the percentage of the data to be used in the testing pool \n \n")
testSize = float(userTestSize)

print("_______________________________________________________________________")
print("\nyou have chosen: " + distanceMetric[userDistanceMetric - 1])
print("\nfor the test percentage you have chosen: " + str(testSize*100) +"%\n")
print("\nfor the training percentage you have chosen: " + str((1 - testSize)*100) +"%\n")


#reads in the csv files
redWine = pd.read_csv("red.csv")
whiteWine = pd.read_csv("white.csv")

#adds a color column with the vaule R
redWine["color"] = "0"
#adds a color column with the vaule W
whiteWine["color"] ="1"

#adds both dataframes into one dataframe
bothRnW=pd.concat([redWine, whiteWine],axis=0)

#renames the columns with a "_" for the space inbetween two words
redWine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
whiteWine.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)
bothRnW.rename(columns={'fixed acidity': 'fixed_acidity','citric acid':'citric_acid','volatile acidity':'volatile_acidity','residual sugar':'residual_sugar','free sulfur dioxide':'free_sulfur_dioxide','total sulfur dioxide':'total_sulfur_dioxide'}, inplace=True)

print("\nattributes in the dataset: "+ str(len(bothRnW.columns)))

print("\nnumber of instances of red wine in the data: " + str(len(redWine.index)))

print("\nnumber of instances of white wine in the data: " + str(len(whiteWine.index)))

#takes the first 12 columns of data for trainig
data = bothRnW.iloc[:, [0, 8]]
#takes the last column color for trainig
labels = bothRnW.iloc[:, [12]]

dataMin = data.min()
dataMax = data.max()
h=0.2

xMin = dataMin.get(key = 'fixed_acidity')
xMax = dataMax.get(key = 'fixed_acidity')

yMin = dataMin.get(key = 'pH')
yMax = dataMax.get(key = 'pH')


#Split the data into randow train and test subsets
xTrain, xTest, yTrain, yTest = train_test_split(data,labels, test_size=testSize, random_state=1 )

knn = KNeighborsClassifier(n_neighbors=3, metric = distanceMetric[userDistanceMetric - 1])
knn.fit(xTrain,np.ravel(yTrain,order='C'))
result = knn.predict(xTest)

xx, yy = np.meshgrid(np.arange(xMin-1, xMax+1, h), np.arange(yMin-1, yMax+1, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6),facecolor='#aaa')



plt.contourf(xx, yy, Z, cmap=cmap_light)
plt.scatter(data.fixed_acidity, data.pH, 3, labels.color)
plt.ylabel('pH')
plt.xlabel('Fixed Acidity')
plt.title("Red and White wines")

plt.show()

print("\naccuracy of this run: " + str(accuracy_score(yTest, result)))
