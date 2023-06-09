import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

#https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data
cols = ['name','landmass','zone', 'area', 'population', 'language','religion','bars','stripes','colours',
'red','green','blue','gold','white','black','orange','mainhue','circles',
'crosses','saltires','quarters','sunstars','crescent','triangle','icon','animate','text','topleft','botright']
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data", names = cols)

#variable names to use as predictors
var = [ 'red', 'green', 'blue','gold', 'white', 'black', 'orange','bars','stripes', 'circles','crosses', 'saltires','quarters','sunstars','triangle','animate']


#Print number of countries by landmass, or continent
America = df[df['landmass']== 1]
S_America = df[df['landmass']== 2]
Europe = df[df['landmass']== 3]
Africa = df[df['landmass']== 4]
Asia = df[df['landmass']== 5]
Oceania = df[df['landmass']== 6]


#Create a new dataframe with only flags from Europe and Oceania
df_EuOc = pd.concat([Europe, Oceania], axis=0)

#Print the average values of the predictors for Europe and Oceania
print(Europe[var].mean())



#Create labels for only Europe and Oceania
labels = None

#Print the variable types for the predictors


#Create dummy variables for categorical predictors
# data = pd.get_dummies(None)

#Split data into a train and test set


#Fit a decision tree for max_depth values 1-20; save the accuracy score in acc_depth
depths = range(1, 21)
acc_depth = []


#Plot the accuracy vs depth


#Find the largest accuracy and the depth this occurs


#Refit decision tree model with the highest accuracy and plot the decision tree


#Create a new list for the accuracy values of a pruned decision tree.  Loop through
#the values of ccp and append the scores to the list


#Plot the accuracy vs ccp_alpha


#Find the largest accuracy and the ccp value this occurs


#Fit a decision tree model with the values for max_depth and ccp_alpha found above


#Plot the final decision tree
