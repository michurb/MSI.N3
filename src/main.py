# Import necessary libraries
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10
from sklearn.metrics import silhouette_score

# Separate module for plotting functions
from plot import plot_images


def load_and_preprocess_data():
    """Load CIFAR-10 data and preprocess it to extract car images and their edge images."""
    (x_train, y_train), _ = cifar10.load_data()
    car_indices = np.where((y_train == 1) | (y_train == 9))[0]
    car_images = x_train[car_indices][::100]
    edge_images = [cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200) for img in car_images]
    return car_images, np.array(edge_images)


def extract_features_from_edge_image(edge_image):
    """Extract features from an edge image."""
    mean_intensity = np.mean(edge_image)
    white_pixel_count = np.sum(edge_image > 0)
    std_dev = np.std(edge_image)
    white_pixel_ratio = white_pixel_count / (edge_image.shape[0] * edge_image.shape[1])
    return [mean_intensity, white_pixel_count, std_dev, white_pixel_ratio]


class SOM:
    def __init__(self, input_dim, map_size, learning_rate=0.5, radius=1.0):
        self.input_dim = input_dim
        self.map_size = map_size
        self.learning_rate = learning_rate
        self.radius = radius
        self.weights = np.random.rand(map_size[0], map_size[1], input_dim)

    def _calculate_distance(self, x, y):
        return np.linalg.norm(x - y)

    def _find_winner(self, input_vector):
        distances = np.linalg.norm(self.weights - input_vector, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def _update_weights(self, input_vector, winner_coords):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist_to_winner = self._calculate_distance(np.array([i, j]), np.array(winner_coords))
                if dist_to_winner <= self.radius:
                    influence = np.exp(-dist_to_winner / (2 * (self.radius**2)))
                    self.weights[i, j] += self.learning_rate * influence * (input_vector - self.weights[i, j])

    def train(self, data, epochs):
        for _ in range(epochs):
            for input_vector in data:
                winner_coords = self._find_winner(input_vector)
                self._update_weights(input_vector, winner_coords)


def compute_silhouette_score(som, data):
    """Compute silhouette score for the given data using the trained SOM."""
    cluster_labels = [som._find_winner(vec) for vec in data]
    cluster_labels = [label[0] * som.map_size[1] + label[1] for label in cluster_labels]

    # Check for the number of unique clusters
    unique_clusters = len(set(cluster_labels))
    if unique_clusters < 2 or unique_clusters >= len(data):
        return -1  # Return a default value if silhouette score can't be computed

    return silhouette_score(data, cluster_labels)


def visualize_som_2d_multiple(som, data, map_size, ax, title=""):
    """Visualize the trained SOM with the given data."""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    cmap = plt.cm.viridis

    silhouette_val = compute_silhouette_score(som, data)
    title_with_score = f"{title} | Silhouette: {silhouette_val:.2f}"

    data_point_added = False
    som_node_added = False

    # Plotting the SOM nodes and circles around them
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            intensity = (som.weights[i, j, 0] - min_val[0]) / (max_val[0] - min_val[0])
            color = cmap(intensity)
            ax.scatter(i, j, marker='s', s=600, color=color, edgecolors='k', linewidths=2,
                       label='SOM Node' if not som_node_added else "")
            circle = plt.Circle((i, j), radius=som.radius, color='r', fill=False)
            ax.add_artist(circle)
            som_node_added = True

    # Plotting the input vectors (data points) with reduced size and increased transparency for better visualization
    for input_vector in data:
        x, y = som._find_winner(input_vector)
        intensity = (input_vector[0] - min_val[0]) / (max_val[0] - min_val[0])
        color = cmap(intensity)
        ax.scatter(x, y, marker='o', s=100, color=color, edgecolors='k', alpha=0.7,
                   label='Data Point' if not data_point_added else "")
        data_point_added = True

    ax.set_title(title_with_score)
    ax.set_xticks(range(map_size[0]))
    ax.set_yticks(range(map_size[1]))
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.2, 1.14))

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Intensity')


def main():
    car_images, edge_images = load_and_preprocess_data()
    feature_vectors = np.array([extract_features_from_edge_image(img) for img in edge_images])

    som = SOM(input_dim=4, map_size=(10, 10))
    som.train(feature_vectors, epochs=10)

    # Visualization generations based on different parameters
    learning_rates = [0.1, 0.5, 0.9]
    radii = [0.5, 1.0, 2.0]
    map_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]

    # Create a directory named by the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(os.getcwd(), "..", "results", current_time)
    os.makedirs(output_directory, exist_ok=True)

    for lr in learning_rates:
        for i, size in enumerate(map_sizes):
            for j, r in enumerate(radii):
                figure_size = (size[0] + 2, size[1] + 2)
                fig, ax = plt.subplots(figsize=figure_size)

                som = SOM(input_dim=4, map_size=size, learning_rate=lr, radius=r)
                som.train(feature_vectors, epochs=10)

                visualize_som_2d_multiple(som, feature_vectors, size, ax)
                ax.margins(0.15, 0.15)

                title = f"Learning Rate: {lr}, Size: {size}, Radius: {r}"
                ax.set_title(title, fontsize=12)

                # Save the plot to the directory
                filename = f"LR_{lr}_Size_{size[0]}x{size[1]}_Radius_{r}.png"
                filepath = os.path.join(output_directory, filename)
                plt.savefig(filepath)

                plt.close(fig)  # Close the figure after saving

if __name__ == "__main__":
    main()