import numpy as np
import pandas as pd
import codecademylib3
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('wine_quality.csv')
print(df.columns)
y = df['quality']
features = df.drop(columns = ['quality'])


## 1. Data transformation
from sklearn.preprocessing import StandardScaler
'''
Scaling the data to fit the variable features and then using transform() to get X 
to get the transformed input to our model
'''
standard_scaler_fit = StandardScaler().fit(features)
X = standard_scaler_fit.transform(features)

## 2. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

## 3. Fit a logistic regression classifier without regularization
from sklearn.linear_model import LogisticRegression
clf_no_reg = LogisticRegression(penalty = 'none') # classifier without regularization
clf_no_reg.fit(X_train, y_train) #fitting the data

## 4. Plot the coefficients
'''
Lines of code to get the ordered coefficients as a bar plot: 
'''
predictors = features.columns
coefficients = clf_no_reg.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
coef.plot(kind='bar', title = 'Coefficients (no regularization)')
plt.tight_layout()
plt.show()
plt.clf()


## 5. Training and test performance using F1_score (weighted mean of precision and recall)
'''
For classifiers, it is important that the classifier not only has high accuracy, but also high precision and recall, 
i.e., a low false positive and false negative rate.

A metric known as f1 score, which is the weighted mean of precision and recall, captures the performance of a 
classifier holistically. It takes values between 0 and 1 and the closer it is to 1, the better the classifier
'''


from sklearn.metrics import f1_score
y_pred_test = clf_no_reg.predict(X_test)
y_pred_train = clf_no_reg.predict(X_train)
print('Training Score', f1_score(y_train, y_pred_train))
print('Testing Score', f1_score(y_test, y_pred_test))

## 6. Default Implementation (L2-regularized!)
clf_default = LogisticRegression()
clf_default.fit(X_train, y_train)

## 7. Ridge Scores
y_pred_train = clf_default.predict(X_train)
y_pred_test = clf_default.predict(X_test)
 
print('Ridge-regularized Training Score', f1_score(y_train, y_pred_train))
print('Ridge-regularized Testing Score', f1_score(y_test, y_pred_test))

'''
Output: 
Training Score 0.7727598566308242
Testing Score 0.7266666666666667
Ridge-regularized Training Score 0.7727598566308242
Ridge-regularized Testing Score 0.7266666666666667
'''

'''

The scores remain the same! Does this mean that regularization did nothing? Indeed! This means that 
the constraint boundary for the regularization we performed is large enough to hold the original loss 
function minimum, thus rendering our model the same as the unregularized one.

How can we tune up the regularization? Recall that C is the inverse of the regularization strength (alpha), 
meaning that smaller values of C correspond to more regularization. The scikit-learn default for C is 1; therefore, 
in order to increase the amount of regularization, we need to consider values of C that are less than 1. 
But how far do we need to go? Lets try a coarse-grained search before performing a fine-grained one.
'''

## 8. Coarse-grained hyperparameter tuning
'''
 Get an array each for the training and test scores 
 corresponding to these values of C.
'''
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]
for x in C_array:
    clf = LogisticRegression(C = x )
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)
    training_array.append(f1_score(y_train, y_pred_train))
    test_array.append(f1_score(y_test, y_pred_test))

## 9. Plot training and test scores as a function of C
plt.plot(C_array,training_array)
plt.plot(C_array,test_array)
plt.xscale('log')
plt.show()
plt.clf()

## 10. Making a parameter grid for GridSearchCV
C_array  = np.logspace(-4, -2, 100)
#Making a dict to enter as an input to param_grid
tuning_C = {'C':C_array}

## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV
clf_gs = LogisticRegression()
gs = GridSearchCV(clf_gs, param_grid = tuning_C, scoring = 'f1', cv = 5)
gs.fit(X_train,y_train) 

## 12. Optimal C value and the score corresponding to it
print(gs.best_params_, gs.best_score_)


## 13. Validating the "best classifier"
'''
he score you got above reflects the mean f1-score on the 5 folds corresponding to the best classifier. Notice however that we haven’t yet used the test data, X_test, y_test from our original train-test split! This was done with good reason: the original test data can now be used as our validation dataset to validate whether our “best classifier” is doing as well as we’d like it to on essentially unknown data.

Define a new classifier clf_best_ridge that corresponds to the best C value you obtained in the previous task. Fit it to the training data and obtain the f1_score on the test data to validate the model.
'''
clf_best = LogisticRegression(C = gs.best_params_['C'])
clf_best.fit(X_train,y_train)
y_pred_best = clf_best.predict(X_test)
print(f1_score(y_test,y_pred_best))

## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
'''

We're now going to use a grid search cross-validation method to regularize the classifier, but with L1 regularization instead. Instead of using GridSearchCV, we’re going to use LogisticRegressionCV. The syntax here is a little different. The arguments to LogisticRegressionCV that are relevant to us:

Cs : A list/array of C values to check; choose values between 0.01 and 100 here.
cv : Number of folds (5 is a good choice here!)
penalty : Remember to choose 'l1' for this!
solver : Recall that L1 penalty requires that we specify the solver to be 'liblinear'.
scoring : 'f1' is still a great choice for a classifier.
Using the above, define a cross-validated classifier, clf_l1 and fit (X,y) here. (Note that we're not doing a train-test-validation split like last time!)
'''
from sklearn.linear_model import LogisticRegressionCV
C_array = np.logspace(-2,2,100)
clf_l1 = LogisticRegressionCV(Cs=C_array, cv = 5, penalty = 'l1', scoring = 'f1', solver = 'liblinear')
clf_l1.fit(X,y)

## 15. Optimal C value and corresponding coefficients
print('Best C value', clf_l1.C_)
print('Best fit coefficients', clf_l1.coef_)


## 16. Plotting the tuned L1 coefficients
coefficients = clf_l1.coef_.ravel()
coef = pd.Series(coefficients,predictors).sort_values()
 
plt.figure(figsize = (12,8))
coef.plot(kind='bar', title = 'Coefficients for tuned L1')
plt.tight_layout()
plt.show()
plt.clf()

'''
final output: 

Training Score 0.7727598566308242
Testing Score 0.7266666666666667
Ridge-regularized Training Score 0.7727598566308242
Ridge-regularized Testing Score 0.7266666666666667
{'C': 0.0019630406500402726} 0.7723336222647887
0.7407407407407408
Best C value [0.25950242]
Best fit coefficients [[ 0.11638766 -0.55633204 -0.17262426  0.0312286  -0.17328661  0.20056466
  -0.50586008  0.         -0.07044797  0.43222007  0.94102915]]
  
'''