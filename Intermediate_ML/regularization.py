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
For classifiers, it is important that the classifier not only has high accuracy, 
but also high precision and recall, i.e., a low false positive and false negative rate.

A metric known as f1 score, which is the weighted mean of precision and recall, 
captures the performance of a classifier holistically. It takes values between 0 and 1 and the closer it is to 1, 
the better the classifier
'''


from sklearn.metrics import f1_score
y_pred_test = clf_no_reg.predict(X_test)
y_pred_train = clf_no_reg.predict(X_train)
print('Training Score', f1_score(y_train, y_pred_train))
print('Testing Score', f1_score(y_test, y_pred_test))

## 6. Default Implementation (L2-regularized!)


## 7. Ridge Scores


## 8. Coarse-grained hyperparameter tuning
training_array = []
test_array = []
C_array = [0.0001, 0.001, 0.01, 0.1, 1]



## 9. Plot training and test scores as a function of C


## 10. Making a parameter grid for GridSearchCV


## 11. Implementing GridSearchCV with l2 penalty
from sklearn.model_selection import GridSearchCV


## 12. Optimal C value and the score corresponding to it


## 13. Validating the "best classifier"


## 14. Implement L1 hyperparameter tuning with LogisticRegressionCV
from sklearn.linear_model import LogisticRegressionCV


## 15. Optimal C value and corresponding coefficients



## 16. Plotting the tuned L1 coefficients
