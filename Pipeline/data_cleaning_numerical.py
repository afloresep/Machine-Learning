import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

## Loading the dataset
columns = ["sex","length","diam","height","whole","shucked","viscera","shell","age"]
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",names=columns)
## Defining target and predictor variables
y = df.age
X = df.drop(columns=['age'])

## Numerical columns:
num_cols = X.select_dtypes(include=np.number).columns
## Categorical columns
cat_cols = X.select_dtypes(include=['object']).columns

## Create some missing values
for i in range(1000):
    X.loc[np.random.choice(X.index),np.random.choice(X.columns)] = np.nan

## Perform train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)


#####-------Imputation and Scaling: Code base to transform -----------------#####
# This is what we are changing...

## Numerical training data
x_train_num = x_train[num_cols]
# Filling in missing values with mean on numeric features only
x_train_fill_missing = x_train_num.fillna(x_train_num.mean())
## Fitting standard scaler on x_train_fill_missing
scale = StandardScaler().fit(x_train_fill_missing)
## Scaling data after filling in missing values
x_train_fill_missing_scale = scale.transform(x_train_fill_missing)
## Same steps as above, but on the test set:
x_test_fill_missing = x_test[num_cols].fillna(x_train_num.mean())
x_test_fill_missing_scale = scale.transform(x_test_fill_missing)
#####-------Imputation and Scaling: Code base to transform -----------------#####

#1. Rewrite using Pipelines!
# To define a pipeline, we pass a list of tuples of the form (name, transform/estimator) into a Pipeline object. 
# In this example for imputation with a SimpleImputer and scale our numerical variables with StandardScaler

pipeline = Pipeline([("imputer", SimpleImputer(strategy='mean')), ("scale", StandardScaler())])

#2. Fit pipeline on the test and compare results
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
pipeline.fit(x_train[num_cols])
x_transform = pipeline.transform(x_test[num_cols])

"""
x_test[num_cols]: This part of the code selects only the numerical columns from the x_test dataset. 
The num_cols variable contains the names of numerical columns, which were previously identified using the select_dtypes method.

pipeline.transform(...): This part applies the data preprocessing steps defined in the pipeline to the selected numerical columns. 
The pipeline consists of two steps:

a. SimpleImputer(strategy='mean'): The SimpleImputer is used to fill any missing values in the numerical columns. 
In this case, it is filling missing values with the mean of each column, as specified by the strategy='mean' argument.

b. StandardScaler(): The StandardScaler is used to scale the numerical columns, making sure they have a mean of 0 and a standard deviation of 1. 
Scaling the features is a common preprocessing step in machine learning to ensure that features with different ranges do not disproportionately 
influence the model.

pipeline.transform(x_test[num_cols]): After defining the pipeline with the two preprocessing steps, the transform method is applied 
to the selected numerical columns of the x_test dataset. This means that missing values are filled with the mean, and the numerical 
columns are scaled using the standardization procedure.

The result of this operation is the transformed x_test[num_cols] dataset, where missing values are replaced with the column means, 
and the numerical columns are scaled for further use in machine learning models.
"""

#3.  Verify pipeline transform test set is the same by using np.array_equal()
array_diff = np.array_equal(x_transform, x_test_fill_missing_scale)
print(array_diff)

#4. Change imputer strategy to median
pipeline_median =Pipeline([("imputer",SimpleImputer(strategy='median')), ("scale",StandardScaler())])
pipeline_median.fit(x_train[num_cols])

# 5 Compare results between the two pipelines
x_transform_median = pipeline_median.transform(x_test[num_cols])
new_array_diff = abs(x_transform-x_transform_median).sum()
print(new_array_diff)