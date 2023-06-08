import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# Load the Data (for Lenovo)
# df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Codecademy\csv\honeyproduction.csv")

# Load the Data (for PC)
df = pd.read_csv(r"C:\Users\USER\OneDrive\Codecademy\csv\honeyproduction.csv")

# Get the mean of totalprod per year
prod_per_year = df.groupby("year")["totalprod"].mean()

# Prod per year values or X and reshape
X = prod_per_year.values
X = X.reshape(-1,1)
"""
eshape your data either using array.reshape(-1, 1) 
if your data has a single feature or array.reshape(1, -1) if it contains545.45454545 a single sample.  
"""

#Years or y values
y = prod_per_year.index


#Plot Data
plt.scatter(X,y)

# Linear Regression Model
regr = linear_model.LinearRegression()
regr.fit(X,y)

# print out the slope of the line (stored in a list called regr.coef_) and the intercept of the line (regr.intercept_).
print(regr.coef_[0])


# List with predictions of regr model would make on X data
y_predicted = regr.predict(X)

# Plot regression line and show
plt.plot(X, y_predicted)


# Predict the Honey Decline

"""
Our known dataset stops at the year 2013, so lets
create a NumPy array called X_future that is the range from 2013 to 2050. 
The code below makes a NumPy array with the numbers 1 through 10
"""
X_future  = np.array(range(2014,2051))
X_future = X_future.reshape(-1,1)
print(X_future)

future_predict = regr.predict(X_future)

plt.plot(X_future, future_predict)
plt.show()
