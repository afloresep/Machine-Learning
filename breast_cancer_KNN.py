

# Import dataset from sklearn 
from sklearn.datasets import load_breast_cancer
breast_cancer_data = load_breast_cancer()

# # See kind of data and names
# print(breast_cancer_data.data[0], breast_cancer_data.feature_names)
# print(breast_cancer_data.target, breast_cancer_data.target_names)

# Splitting the data into Training and Validation Sets 
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
# breast_cancer_data.data is the data we want to split. .target  is the labels associated with the data


# Create KKNN classifier 
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 3)

# Train classifier using the fit function.  
classifier.fit(x_train, y_train)

# Find how accurate it is on the validation set
print(classifier.score(x_test, y_test))

results = []
# Find best K value
for i in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors = i)
  classifier.fit(x_train, y_train)
# Find how accurate it is on the validation set
  results.append(classifier.score(x_test, y_test))

# Plot validation accuracy for 100 different K's
import matplotlib.pyplot as plt
K_values = range(1,101)
plt.plot(K_values, results)
plt.xlabel('K values')
plt.ylabel('Accuracy')
plt.title('Breast Cancer Classifier Accuracy')
plt.show()
