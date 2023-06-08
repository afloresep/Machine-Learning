# Predict Credit Card Fraud: Logistic Regression and create a predictive model to determine if a transaction is fraudulent or not. 
#  DataSet can be found at "https://www.kaggle.com/datasets/ealaxi/paysim1"
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv(r'C:\Users\USER\OneDrive\Codecademy - Copy\csv\PS_20174392719_1491204439457_log.csv')
print(df.head())
print(df.info())

# Number of fraudulent transactions
Number_fraudulent_transactions = df['isFraud'].sum()
print(Number_fraudulent_transactions) # 282 fraudulent transactions

# Summary statistics on amount column
summary_amount = Number_fraudulent_transactions = df['amount'].sum()
print(summary_amount) # 537 Million
print(df['amount'].describe()) 

# Create isPayment field using lambda function
df['isPayment'] = df['type'].apply(lambda x: 1 if x == 'PAYMENT' or x == 'DEBT' else 0)

# Create isMovement field
df['isMovement'] = df['type'].apply(lambda x: 1 if x == 'CASH_IN' or x == 'CASH_OUT' else 0)
print(df)

# Create accountDiff field
df['accountDiff'] = df['oldbalanceDest'] - df['newbalanceDest']

# Create features and label variables
features = df[['amount', 'isPayment', 'isMovement', 'accountDiff']]
label = df['isFraud']

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(features, label, train_size=0.7, test_size=0.3)

# Normalize the features variables
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
"""
Since sklearn‘s Logistic Regression implementation uses Regularization, we need to scale our feature data. Create a StandardScaler object, .fit_transform() it on the training features, and .transform() the test features
"""

# Fit the model to the training data
df_lr = LogisticRegression()
df_lr.fit(x_train, y_train)

# Score the model on the training data
print(df_lr.score(x_train, y_train))
"""
.score() method on the training data and print the training score.
Scoring the model on the training data will process the training data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy.
"""
# Score the model on the test data
print(df_lr.score(x_test, y_test))
"""
Scoring the model on the test data will process the test data through the trained model and will predict which transactions are fraudulent. The score returned is the percentage of correct classifications, or the accuracy, and will be an indicator for the sucess of your model.
"""

"""
Evaluating the model's performance on the test set provides an estimate of its ability to generalize to unseen data, while evaluating it on the training set assesses how well the model fits the training data. Both evaluations are useful for understanding the model's performance and generalization capability.
"""

# Print the model coefficients
print(df_lr.coef_, df_lr.intercept_)

# [[ 4.85837568e-07 -6.44343644e-12 -1.07882568e-11  7.06343566e-08]] [-1.62927807e-11]

# New transaction data
transaction1 = np.array([123456.78, 0.0, 1.0, 54670.1])
transaction2 = np.array([98765.43, 1.0, 0.0, 8524.75])
transaction3 = np.array([543678.31, 1.0, 0.0, 510025.5])

# Create a new transaction
your_transaction = np.array([234523.43, 1.0, 0.0, 14111])

# Combine new transactions into a single array
sample_transactions = np.array([transaction1, transaction2, transaction3, your_transaction])

# Normalize the new transactions
scaler.fit(sample_transactions)
sample_transactions = scaler.transform(sample_transactions)

# Predict fraud on the new transactions
fraud_s_transactions_predicted = df_lr.predict(sample_transactions)

# Show probabilities on the new transactions
print(df_lr.predict_proba(sample_transactions))