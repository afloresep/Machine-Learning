"Create a few linear regression models that use two features to predict yearly earnings = MULTIPLE LINEAR REGRESSION MODEL"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# load and investigate the data here:
df = pd.read_csv(r"C:\Users\USER\OneDrive\Codecademy\csv\tennis_ace_starting\tennis_stats.csv")

# Select the two features and the target variable
features = df[["BreakPointsOpportunities", "ServiceGamesPlayed"]] # Features (X)
yearly_winnings = df["Winnings"] # Target (y)

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(features, yearly_winnings, test_size=0.2, random_state=42)

# Create and train the linear regression models. Initializing empty lists
models = []
mse_scores = []

# Model 1: features (Break Points Opp., First Serve Points Won)
model1 = LinearRegression()
model1.fit(X_train, y_train)
models.append(model1)
y_pred1 = model1.predict(X_test)
mse1 = mean_squared_error(y_test, y_pred1)
mse_scores.append(mse1)
print(mse1)


## multiple features linear regression

# select features and value to predict
features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon','SecondServePointsWon','SecondServeReturnPointsWon','Aces','BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities','BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon','ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon','TotalServicePointsWon']]
winnings = df[['Winnings']]

# train, test, split the data
features_train, features_test, winnings_train, winnings_test = train_test_split(features, winnings, train_size = 0.8)

# create and train model on training data
model = LinearRegression()
model.fit(features_train,winnings_train)

# score model on test data
print('Predicting Winnings with Multiple Features Test Score:', model.score(features_test,winnings_test))

# make predictions with model
winnings_prediction = model.predict(features_test)

# plot predictions against actual winnings
plt.scatter(winnings_test,winnings_prediction, alpha=0.4)
plt.title('Predicted Winnings vs. Actual Winnings - Multiple Features')
plt.xlabel('Actual Winnings')
plt.ylabel('Predicted Winnings')
plt.show()
plt.clf()