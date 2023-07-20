import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Load the data set
cancer = load_breast_cancer()

# Split the data into training and testing sets
X = cancer.data
y = cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

# These are the hyperparameters that we will test.
# We'll try both 'l1' and 'l2' regularization.
# C is the inverse of regularization strength. Smaller C will result in stronger regularization.


# The logistic regression model
lr = LogisticRegression(solver = 'liblinear', max_iter = 1000)

''' 
LogisticRegression model has only two possible values: l1 and l2. We list them both
The hyperparameter C is the inverse of regularization strength. It can be any positive number, 
so we have to specify a probability distribution that allows us to randomly select a positive number.
 The scipy library has many probability distributions to choose from
For this example, we're using the uniform distribution. This allows us to randomly 
select numbers between loc and loc+scale (in this case, between 0 and 100).
''' 
distributions = {'penalty': ['l1', 'l2'], 'C': uniform(loc=0, scale=100)}

# Create a RandomizedSearchCV model
clf = RandomizedSearchCV(lr, distributions, n_iter = 8) # Set to test 8 hyperparameter combinations

# Fit the RandomizedSearchCV model
clf.fit(X_train, y_train)


# Evaluate:
# 1. See which hyperparameters performed the best
print(clf.best_estimator_)

#output: LogisticRegression(C=81.21687287754932, max_iter=1000, penalty='l1',               solver='liblinear'). Best was l1 penalty with c = 81

# 2.  Print the parameters and mean test score
print(clf.cv_results_['params'])
print(clf.cv_results_['mean_test_score'])

# 3. Create and print Pandas DataFrame
cv_table = pd.concat([pd.DataFrame(clf.cv_results_['params']), pd.DataFrame(clf.cv_results_['mean_test_score'], columns=['Accuracy'])], axis=1)
 
