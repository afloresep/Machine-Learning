import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load and investigate the data here:
df = pd.read_csv(r"C:\Users\USER\OneDrive\Codecademy\csv\tennis_ace_starting\tennis_stats.csv")

# perform exploratory analysis here:
wins = df['Wins']
ranking = df['Ranking']

"""
# Plot data to see if there's a linear correlation
plt.scatter(wins, ranking)
plt.xlabel("Wins")
plt.ylabel("Ranking")
plt.show()
"""

# Select one feature and the outcome variable
Winnings = df['Winnings']
BreakPointsOpportunities = df['BreakPointsOpportunities']

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(Winnings, BreakPointsOpportunities, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train.values.reshape(-1, 1), y_train)

# Make predictions on the test set
y_pred = model.predict(X_test.values.reshape(-1, 1))

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# mse = 2953.3033570321727 

# Plot the model's predictions against the actual outcome variable
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel("Break Points Opportunities")
plt.ylabel("Winning")
plt.legend()
plt.show()























## perform two feature linear regressions here:






















## perform multiple feature linear regressions here:





















