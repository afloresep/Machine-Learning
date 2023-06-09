# Importing the necessary variables from the 'movies' module
from movies import training_set, training_labels, validation_set, validation_labels


def distance(movie1, movie2):
    # Initialize the squared difference
    squared_difference = 0

    # Iterate over the elements of the movies
    for i in range(len(movie1)):
        # Calculate the squared difference of each element and add it to the total
        squared_difference += (movie1[i] - movie2[i]) ** 2

    # Calculate the final distance by taking the square root of the sum
    final_distance = squared_difference ** 0.5

    # Return the final distance
    return final_distance


def classify(unknown, dataset, labels, k):
    distances = []

    # Loop through all points in the dataset
    for title in dataset:
        movie = dataset[title]
        distance_to_point = distance(movie, unknown)

        # Adding the distance and point associated with that distance
        distances.append([distance_to_point, title])

    # Sort the distances in ascending order
    distances.sort()

    # Taking only the k closest points
    neighbors = distances[0:k]

    num_good = 0
    num_bad = 0

    # Count the number of good and bad labels among the neighbors
    for neighbor in neighbors:
        title = neighbor[1]

        if labels[title] == 0:
            num_bad += 1
        elif labels[title] == 1:
            num_good += 1

    # Make a prediction based on the majority label
    if num_good > num_bad:
        return 1
    else:
        return 0


def find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, k):
    num_correct = 0.0

    # Loop through the movies in the validation set
    for title in validation_set:
        guess = classify(validation_set[title], training_set, training_labels, k)

        # Check if the prediction matches the actual label
        if guess == validation_labels[title]:
            num_correct += 1

    # Calculate the validation accuracy as the ratio of correct predictions to the total number of movies
    accuracy = num_correct / len(validation_set)

    return accuracy


# Calculate the validation accuracy using the training and validation sets
validation_accuracy = find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 3)

# Print the validation accuracy
print(validation_accuracy)
# Output: 0.6639344262295082
