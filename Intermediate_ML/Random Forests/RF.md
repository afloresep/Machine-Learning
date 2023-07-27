Bagging
Random forests create different trees using a process known as bagging, which is short for bootstrapped aggregating. As we already covered bootstrapping, the process starts with creating a single decision tree on a bootstrapped sample of data points in the training set. Then after many trees have been made, the results are “aggregated” together. In the case of a classification task, often the aggregation is taking the majority vote of the individual classifiers. For regression tasks, often the aggregation is the average of the individual regressors.

We will dive into this process for the cars dataset we used in the previous exercise. The dataset has six features:

buying: car price as a categorical variable: “vhigh”, “high”, “med”, or “low”
maint: cost of maintaining the car; can be “vhigh”, “high”, “med”, or “low”.
doors: number of doors; can be “2”, “3”, “4”, “5more”.
persons: number of people the car can hold; can be “2”, “4”, or “more”.
lugboot: size of the trunk; can be “small”, “med”, or “big”.
safety: safety rating of the car; can be “low”, “med”, or “high”

See `bagging.py` first:
Code is pretty straight forward but this part could be tricky:

"preds = []
random_state = 0

for i in range(10):
    ids = x_train.sample(x_train.shape[0], replace=True, random_state=i).index
    dt2.fit(x_train.loc[ids], y_train[ids])
    preds.append(dt2.predict(x_test))"


- preds is a list that stores the predictions made by each decision tree on the test data (x_test). It will hold 10 arrays, each containing the predictions from one decision tree.

#### Aggregating Predictions:
After the for loop, the code aggregates the predictions from all the decision trees using the mean function along the rows (axis 0). This process creates an array ba_pred that represents the aggregated predictions of the 10 decision trees on the test data.

"ba_pred = np.array(preds).mean(0)
print(ba_pred)"

- np.array(preds) converts the list of prediction arrays into a numpy array.
- mean(0) calculates the mean of each column (across the rows) in the numpy array. Since each column represents predictions from one decision tree, taking the mean along axis 0 effectively averages the predictions made by all 10 decision trees for each sample in the test data.

#### Calculating Bagging Accuracy:
The final step is to calculate the accuracy of the bagged (aggregated) predictions on the test data.

ba_accuracy = accuracy_score(ba_pred>=0.5, y_test)
print(f'Accuracy score of aggregated 10 bootstrapped samples: {ba_accuracy.round(4)}')

- `ba_pred>=0.5` creates a boolean array where each element indicates whether the corresponding aggregated prediction is greater than or equal to 0.5 (the threshold for the positive class). Everything equal or above 0.5 is now True (or 1) and below 0,5 is False (or 0). Since y_test is the true labels (all 1 or 0), now we can compare how did our model perfom. 
- `accuracy_score` then calculates the accuracy of these binary predictions compared to the true labels (y_test).
The final printed output will be the accuracy score of the bagged model on the test data. Bagging helps reduce overfitting and variance in the model by combining predictions from multiple models trained on different subsets of the data. This typically results in improved generalization and better overall performance on unseen data.


## 2. Random Feature Selection

When we use a decision tree, all the features are used and the split is chosen as the one that increases the information gain the most. While it may seem counter-intuitive, selecting a random subset of features can help in the performance of an ensemble model. In the following example, we will use a random selection of features prior to model building to add additional variance to the individual trees. While an individual tree may perform worse, sometimes the increases in variance can help model performance of the ensemble model as a whole.
See `random_feature_selection.py`
Our target variable for prediction is an acceptability rating, accep, that’s either True or False. For our final features sets, x_train and x_test, the categorical features have been dummy encoded, giving us 15 features in total.
