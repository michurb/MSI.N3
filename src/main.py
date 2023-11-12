# Import necessary libraries
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10
import itertools
from multiprocessing import Pool, cpu_count


def load_and_preprocess_data():
    """Load CIFAR-10 data and preprocess it to extract car images, their edge images, and their labels."""
    (x_train, y_train), _ = cifar10.load_data()
    car_indices = np.where((y_train == 1) | (y_train == 9))[0]
    car_images = x_train[car_indices]
    labels = y_train[car_indices]
    edge_images = [cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200) for img in car_images]
    return car_images, np.array(edge_images), labels


def extract_features_from_edge_image(edge_image):
    """Extract features from an edge image."""
    mean_intensity = np.mean(edge_image)
    white_pixel_count = np.sum(edge_image > 0)
    std_dev = np.std(edge_image)
    white_pixel_ratio = white_pixel_count / (edge_image.shape[0] * edge_image.shape[1])
    return [mean_intensity, white_pixel_count, std_dev, white_pixel_ratio]


def extract_features_from_colored_image(image):
    """Extract features from a colored image."""
    # Average values of RGB channels
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    # Standard deviation of RGB channels
    std_r = np.std(image[:, :, 0])
    std_g = np.std(image[:, :, 1])
    std_b = np.std(image[:, :, 2])

    return [avg_r, avg_g, avg_b, std_r, std_g, std_b]


class SOM:
    def __init__(self, input_dim, map_size, data, learning_rate=0.5, radius=1.0):
        self.input_dim = input_dim
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.radius = radius
        self.weights = np.random.rand(map_size[0], map_size[1], self.input_dim)

    def _calculate_distance(self, x, y):
        return np.linalg.norm(x - y)

    def _find_winner(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def _update_weights(self, input_vector, winner_coords):
        # Calculate the distance from each neuron to the winner
        x = np.arange(self.map_size[0])
        y = np.arange(self.map_size[1])
        dist_x, dist_y = np.meshgrid(x, y)
        dist_to_winner = np.sqrt((dist_x - winner_coords[0]) ** 2 + (dist_y - winner_coords[1]) ** 2)

        # Only consider neurons within the given radius
        mask = dist_to_winner <= self.radius

        # Calculate the influence
        influence = np.exp(-dist_to_winner / (2 * (self.radius ** 2)))
        influence = influence * mask

        # Update weights for the influenced neurons
        delta = input_vector - self.weights
        for i in range(self.weights.shape[2]):
            self.weights[:, :, i] += self.learning_rate * influence * delta[:, :, i]

    def train(self, data, epochs):
        initial_lr = self.learning_rate
        initial_radius = self.radius
        for epoch in range(epochs):
            self.learning_rate = initial_lr * (1 - epoch / float(epochs))
            self.radius = initial_radius * (1 - epoch / float(epochs))
            for input_vector in data:
                winner_coords = self._find_winner(input_vector)
                self._update_weights(input_vector, winner_coords)


def visualize_som_clusters(cluster_centers, samples, ax):
    """
    Visualizes the SOM clusters and samples.

    Parameters:
    - cluster_centers: The cluster centers obtained from the SOM.
    - samples: The samples or data points.
    - ax: The matplotlib axis object to plot on.
    """
    # Plotting cluster centers
    for center in cluster_centers:
        ax.scatter(center[0], center[1], color='black', s=100)

        # Find the distance to the nearest other cluster center to set the influence radius
        other_centers = [c for c in cluster_centers if not np.array_equal(c, center)]
        distances = [np.linalg.norm(center - c) for c in other_centers]
        influence_radius = min(distances) / 2
        circle = plt.Circle((center[0], center[1]), radius=influence_radius, color='red', fill=False)
        ax.add_artist(circle)

    # Plotting samples
    ax.scatter(samples[:, 0], samples[:, 1], color='blue', s=30, label='Punkt danych')

    ax.set_xlim([-1, cluster_centers[:, 0].max() + 2])
    ax.set_ylim([-1, cluster_centers[:, 1].max() + 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.legend(loc='upper right')
    ax.grid(True)

def train_and_visualize(params):
    map_size, feature_vectors, learning_rate, radius, epochs, output_directory = params
    som = SOM(input_dim=6, map_size=map_size, data=feature_vectors, learning_rate=learning_rate, radius=radius)
    som.train(feature_vectors, epochs)

    # Extracting cluster centers from SOM weights
    cluster_centers = np.array([som.weights[i, j] for i in range(som.map_size[0]) for j in range(som.map_size[1])])
    samples_coords = np.array([som._find_winner(vec) for vec in feature_vectors])
    samples = np.array(
        [(coord[0] + np.random.normal(0, 0.2), coord[1] + np.random.normal(0, 0.2)) for coord in samples_coords])

    fig, ax = plt.subplots(figsize=(10, 10))
    visualize_som_clusters(cluster_centers, samples, ax)
    ax.set_title(f"Learning Rate: {learning_rate}, Radius: {radius}, Epochs: {epochs}", fontsize=12)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filename = f"LR_{learning_rate}_Radius_{radius}_Epochs_{epochs}.png"
    filepath = os.path.join(output_directory, filename)
    plt.savefig(filepath)

    plt.close(fig)  # Close the figure after saving

def visualize_som_results(map_size, feature_vectors, learning_rates, radii, epochs_list, output_directory="result"):
    """Visualizes the SOM results for varying learning rates, radii, and epochs. Saves plots to the specified directory."""
    for lr in learning_rates:
        for r in radii:
            for ep in epochs_list:
                # Training the SOM
                som = SOM(input_dim=4, map_size=map_size, data=feature_vectors, learning_rate=lr, radius=r)
                som.train(feature_vectors, epochs=ep)

                # Extracting cluster centers from SOM weights
                cluster_centers = np.array(
                    [som.weights[i, j] for i in range(som.map_size[0]) for j in range(som.map_size[1])])

                # Mapping feature vectors to the SOM to get their coordinates
                samples_coords = np.array([som._find_winner(vec) for vec in feature_vectors])

                # Convert samples_coords with a smaller jitter
                samples = np.array(
                    [(coord[0] + np.random.normal(0, 0.03), coord[1] + np.random.normal(0, 0.03)) for coord in
                     samples_coords])

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 10))
                # inny rodzaj graphow
                visualize_som_clusters(cluster_centers, samples, ax)
                ax.set_title(f"Wspolczynnik uczenia: {lr}, Promien: {r}, Epoki: {ep}", fontsize=12)

                # Adjusting visualization scale based on both samples and cluster centers
                all_points_x = np.concatenate([samples[:, 0], cluster_centers[:, 0]])
                all_points_y = np.concatenate([samples[:, 1], cluster_centers[:, 1]])
                ax.set_xlim([all_points_x.min() - 1, all_points_x.max() + 1])
                ax.set_ylim([all_points_y.min() - 1, all_points_y.max() + 1])

                # Save the plot to the directory
                filename = f"LR_{lr}_Promien_{r}_Epoki_{ep}.png"
                filepath = os.path.join(output_directory, filename)
                plt.savefig(filepath)

                plt.close(fig)

def main():
    car_images, _, _ = load_and_preprocess_data()
    feature_vectors = np.array([extract_features_from_colored_image(img) for img in car_images])
    learning_rates = [0.1, 0.5, 0.9]
    radii = [0.5, 1.0, 2.0]
    epochs_list = [100, 1000, 10000]
    map_size = (20, 20)
    output_directory = "result/run_parallelCOLOR"

    # Create all combinations of hyperparameters
    all_params = itertools.product(
        [map_size], [feature_vectors], learning_rates, radii, epochs_list, [output_directory]
    )

    for params in all_params:
        train_and_visualize(params)


def main_parallel():
    learning_rates = [0.3, 0.1, 0.05, 0.01]
    radii = [10, 5, 2.5, 1.25, 1]
    epochs_list = [1, 1000, 100000]
    map_size = (20, 20)
    output_directory_color = "result/run_parallelCOLOR"
    output_directory_edge = "result/run_parallelEDGE"

    # Load data
    car_images, edge_images, _ = load_and_preprocess_data()

    # Color images feature vectors
    color_feature_vectors = np.array([extract_features_from_colored_image(img) for img in car_images])
    color_feature_vectors = color_feature_vectors - np.mean(color_feature_vectors, axis=0)
    color_feature_vectors = color_feature_vectors / np.std(color_feature_vectors, axis=0)

    #edge images
    edge_feature_vectors = np.array([extract_features_from_edge_image(img) for img in edge_images])
    edge_feature_vectors = edge_feature_vectors - np.mean(edge_feature_vectors, axis=0)
    edge_feature_vectors = edge_feature_vectors / np.std(edge_feature_vectors, axis=0)


    # Get the number of available CPUs
    num_processes = cpu_count()

    # Initialize the Pool and map the function to the hyperparameters for color features
    with Pool(num_processes) as pool:
        color_params = itertools.product([map_size], [color_feature_vectors], learning_rates, radii, epochs_list,
                                         [output_directory_color])
        pool.map(train_and_visualize, color_params)

    # Initialize the Pool and map the function to the hyperparameters for edge features
    with Pool(num_processes) as pool:
        edge_params = itertools.product([map_size], [edge_feature_vectors], learning_rates, radii, epochs_list,
                                        [output_directory_edge])
        pool.map(train_and_visualize, edge_params)


if __name__ == "__main__":
    main_parallel()
