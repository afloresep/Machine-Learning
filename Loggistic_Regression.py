# Import pandas and the data
import pandas as pd
codecademyU = pd.read_csv('codecademyU_2.csv')

# Separate out X and y
X = codecademyU[['hours_studied', 'practice_test']]
y = codecademyU.passed_exam

# Transform X
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 27)

# Create and fit the logistic regression model here:
from sklearn.linear_model import LogisticRegression
cc_lr = LogisticRegression()
cc_lr.fit(X_train, y_train)

# Print the intercept and coefficients here:
print(cc_lr.coef_, cc_lr.intercept_)
# [[1.5100409  0.12002228]] [-0.13173123]
"""
Both coefficients are positive, which makes sense: we expect students who study more and earn higher grades on 
the practice test to be more likely to pass the final exam. The coefficient on hours_studied is larger than the coefficient on practice_test, 
suggesting that hours_studied is more strongly associated with students' probability of passing
"""
