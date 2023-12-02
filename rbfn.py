import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy.random import uniform
import time

# Gaussian formula for weights calculation
def radial_basis_function(x, c, sigma):
    return np.exp(-np.square(np.linalg.norm(x - c)) / (2 * sigma**2))

# Activation function of the hidden layer
def activation_function(centers, data, sigma):
    hidden_layer_output_rows = []
    for x in data:
        row_values = []
        for c in centers:
            row_values.append(radial_basis_function(x, c, sigma))
        hidden_layer_output_rows.append(row_values)
    return np.array(hidden_layer_output_rows)

# Choosing centers of RBFN randomly
def random_centers(data,  num_centers):
    num_samples = data.shape[0]
    random_indices = np.random.choice(num_samples, num_centers, replace=False)
    return data[random_indices]

# Calculation of neural network losses
def get_loss(predicted, target):
    losses =[]
    for i in range(len(target)):
        loss = target[i] - predicted[i]
        if loss < 0:
            loss = loss * -1
        losses.append(loss)
    return np.mean(losses)

# Calculation of Euclidean distance between point & data
def euclidean_distance(point, data):
    return np.sqrt(np.sum((point - data)**2, axis=1))

# K-Means Clustering algorithm to find appropriate RBFN centers
def kmeans_centers(data, num_centers, max_iter=100):
    min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
    centers = np.array([uniform(min_, max_) for _ in range(num_centers)])
    iteration = 0
    labels = []
    prev_centers = None
    while np.not_equal(centers, prev_centers).any() and iteration < max_iter:
        labels = []
        sorted_points = [[] for _ in range(num_centers)]
        for x in data:
            dists = euclidean_distance(x, centers)
            label = np.argmin(dists)
            sorted_points[label].append(x)
            labels.append(label)
        prev_centers = centers
        centers = [np.mean(cluster, axis=0) for cluster in sorted_points]
        for i, center in enumerate(centers):
            if np.isnan(center).any():
                centers[i] = prev_centers[i]
        iteration += 1
    return np.array(centers), np.array(labels)
    
# Training of RBFN
def train_rbf_network(X_train, y_train, num_centers, sigma=1.0):
    labels = []
    centers, labels = kmeans_centers(X_train, num_centers)
    # centers = random_centers(X_train, num_centers)
    activation_function_output = activation_function(centers, X_train, sigma)
    rotated_matrix = np.linalg.pinv(activation_function_output)
    output_weights = rotated_matrix.dot(y_train)
    return labels, centers, output_weights

# Prediction with RBFN
def predict_rbf_network(data, centers, output_weights, sigma=1.0):
    hidden_layer_output = activation_function(centers, data, sigma)
    predicted_output = hidden_layer_output.dot(output_weights)
    return np.round(predicted_output)

# Gets the most appropriate number of centers for a given dataset
def get_optimal_centers_quantity(data, target, from_center, to_center, test_cycles=100):
    center_values = range(from_center, to_center+1)
    all_accuracies = []
    time_consumptions = []
    results = []
    for center_num in center_values:
        start_time = time.time()
        accuracies = []
        for _ in range(test_cycles):
            X_train, X_target, y_train, y_target = train_test_split(data, target, test_size=0.2, random_state=42)
            _, centers, weights = train_rbf_network(X_train, y_train, center_num)
            y_pred = predict_rbf_network(X_target, centers, weights)
            accuracy = accuracy_score(y_target, y_pred)
            accuracies.append(accuracy)
        time_spent = time.time()-start_time
        mean_accuracy = np.mean(accuracies)
        all_accuracies.append(mean_accuracy)
        time_consumptions.append(time_spent)
        results.append((1-mean_accuracy)*time_spent)
    return center_values[np.argmin(results)], results, time_consumptions, all_accuracies