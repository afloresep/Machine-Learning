import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'accep'])
df['accep'] = ~(df['accep']=='unacc') #1 is acceptable, 0 if not acceptable
X = pd.get_dummies(df.iloc[:,0:6], drop_first=True)
y = df['accep']
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.25)
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
print("Accuracy score of DT on test set (trained using full feature set):")
accuracy_dt = dt.score(x_test, y_test)
print(accuracy_dt)

# 1. Create rand_features, 10 random samples from the set of features (15 original features)
rand_features = np.random.choice(x_train.columns,10)

# Make new decision tree trained on random sample of 10 features and calculate the new accuracy score
dt2 = DecisionTreeClassifier()
dt2.fit(x_train[rand_features], y_train)
print("Accuracy score of DT on test set (trained using random feature sample):")
accuracy_dt2 = dt2.score(x_test[rand_features], y_test)
print(accuracy_dt2)
# Dont use .loc/.iloc because we're selecting columns, not rows. 

# 2. Build decision trees on 10 different random samples 
predictions = []
for i in range(10):
     rand_features = np.random.choice(x_train.columns,10)
     dt2.fit(x_train[rand_features], y_train)
     predictions.append(dt2.predict(x_test[rand_features]))
print(predictions)

## 3. Get aggregate predictions and accuracy score

'''
We have ten decision trees’ worth of predictions now
To meaningfully combine their predictions, We use the following system. If more than 5 classifiers predict that a datapoint belong to a certain class, we assign an aggregate prediction to that class.
'''
prob_predictions = np.array(predictions).mean(0)
"""
The mean() function allows you to calculate the mean along a specific axis of the array. Axis 0 corresponds to the vertical axis (rows), while axis 1 corresponds to the horizontal axis (columns).

When you use mean(0), it calculates the mean along axis 0, which means it will compute the mean for each column separately
"""
agg_predictions = (prob_predictions>0.5)
agg_accuracy = accuracy_score(agg_predictions, y_test)

print('Accuracy score of aggregated 10 samples:')
print(agg_accuracy)
