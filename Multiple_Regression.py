import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split


streeteasy = pd.read_csv("https://raw.githubusercontent.com/sonnynomnom/Codecademy-Machine-Learning-Fundamentals/master/StreetEasy/manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

# Add the code here:

from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(x_train, y_train) 
# finds the coefficients and the intercept value

y_predict = mlr.predict(x_test)
# takes values calculated by `.fit()` and the `x` values, plugs them into the multiple linear regression equation, and calculates the predicted y values. 

"""
Features	Sonny’s Apartment
bedrooms	1
bathrooms	1
size_sqft	620 ft²
min_to_subway	16 min
floor	1
building_age_yrs	98 (built in 1920)
no_fee	1
has_roofdeck	0
has_washer_dryer	Yas
has_doorman	0
has_elevator	0
has_dishwasher	1
has_patio	1
has_gym	0
"""

# Sonny doesn't have an elevator so the 11th item in the list is a 0
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
 
predict = mlr.predict(sonny_apartment)
print("Predicted rent: $%.2f" % predict)


"""
The .fit() method gives the model two variables that are useful to us:
    .coef_, which contains the coefficients 
    .intercept_, which contains the intercept

After performing multiple linear regression, you can print the coefficients using .coef_.
Coefficients are most helpful in determining which independent variable carries more weight. 
For example, a coefficient of -1.345 will impact the rent more than a coefficient of 0.238, 
with the former impacting prices negatively and latter positively.
"""

# In our Manhattan model, we used 14 variables, so there are 14 coefficients:
print(mlr.coef_)

# .score() method from LinearRegression to find the mean squared error regression loss for the training set.
print(mlr.score(x_train, y_train))
