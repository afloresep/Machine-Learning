import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6], drop_first=True)
y = df['accep']
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)

#1.Decision tree trained on  training set
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(x_train, y_train)
print(f'Accuracy score of DT on test set (trained using full set): {dt.score(x_test, y_test).round(4)}')
# output: 
'''
Output:
Accuracy score of DT on test set (trained using full set): 0.8588
'''

#2. New decision tree trained on bootstrapped sample
dt2 = DecisionTreeClassifier(max_depth=5)
#ids are the indices of the bootstrapped sample
ids = x_train.sample(x_train.shape[0], replace=True, random_state=0).index
dt2.fit(x_train.loc[ids], y_train[ids])
print(f'Accuracy score of DT on test set (trained using bootstrapped sample): {dt2.score(x_test, y_test).round(4)}')
# Remember that .loc in pandas allows the return of specified rows and/or columns from that DataFrame 
'''
Output:
Accuracy score of DT on test set (trained using bootstrapped sample): 0.8912
'''

#Repeat a decision tree build on 10 different bootstrapped samples using a for loop and aggregate results:
preds = []
random_state = 0
#Write for loop:
for i in range(10):
  ids = x_train.sample(x_train.shape[0], replace=True, random_state=i).index
  dt2.fit(x_train.loc[ids], y_train[ids])
  preds.append(dt2.predict(x_test))
ba_pred = np.array(preds).mean(0)
print(ba_pred)

# 4. Calculate accuracy of the bagged sample
ba_accuracy = accuracy_score(ba_pred>=0.5, y_test)
print(f'Accuracy score of aggregated 10 bootstrapped samples:{ba_accuracy.round(4)}')
