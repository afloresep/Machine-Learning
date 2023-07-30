"""
Dealing with missing values in categorical data and one-hot-encoding categorical variables. 
We will convert an existing codebase to a pipeline like in the previous exercise. The two steps in detail are:

1. SimpleImputer() will be used again to fill missing values in the pipeline, but this time, the strategy parameter will need to be updated to most_frequent.
2. OneHotEncoder() will be used as the second step in the pipeline. The default setting in scikit-learn's OneHotEncoder() is that a sparse array will be
returned from this transform, so we will use sparse='False' to return a full array.
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)

y = df.age
X=df.drop(columns=['age'])
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(include=['object']).columns

######## ----- Existing code that we want to rewrite using Pipelines

#create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
x_train_cat = x_train[cat_cols]
#fill missing values with mode on categorical features only
x_train_fill_missing = x_train_cat.fillna(x_train_cat.mode().values[0][0])
#apply one hot encoding on x_train_fill_missing
ohe = OneHotEncoder(sparse=False, drop='first').fit(x_train_fill_missing)
#transform data after filling in missing values
x_train_fill_missing_ohe = ohe.transform(x_train_fill_missing)

#Now want to do the same thing on the test set! 
x_test_fill_missing = x_test[cat_cols].fillna(x_train_cat.mode().values[0][0])
x_test_fill_missing_ohe = ohe.transform(x_test_fill_missing)
####### ----------------------- 

    
#1. Rewrite using Pipelines!



#2. Fit the pipeline and transform the test data (categorical columns only!)


#3. Check if the two arrays are the same using np.array_equal()

#print('Are the arrays equal?')

