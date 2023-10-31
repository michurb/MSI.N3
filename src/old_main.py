# Import necessary libraries
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10
from sklearn.metrics import silhouette_score
from matplotlib.lines import Line2D


# Separate module for plotting functions
from plot import plot_images


def load_and_preprocess_data():
    """Load CIFAR-10 data and preprocess it to extract car images, their edge images, and their labels."""
    (x_train, y_train), _ = cifar10.load_data()
    car_indices = np.where((y_train == 1) | (y_train == 9))[0]
    car_images = x_train[car_indices][::100]
    labels = y_train[car_indices][::100]
    edge_images = [cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200) for img in car_images]
    return car_images, np.array(edge_images), labels


def load_and_preprocess_data_rgb():
    """Load CIFAR-10 data and preprocess it to extract car images and their labels."""
    (x_train, y_train), _ = cifar10.load_data()

    # Extracting car images. In CIFAR-10, label 1 corresponds to 'automobile'
    car_indices = np.where((y_train == 1) | (y_train == 9))[0]
    car_images = x_train[car_indices][::100]
    labels = y_train[car_indices][::100]

    return car_images, labels


def extract_features_from_rgb_image(rgb_image):
    """Extracts 4 features from the RGB image: mean and standard deviation of the R, G, and B channels."""
    red_channel = rgb_image[:, :, 0]
    green_channel = rgb_image[:, :, 1]
    blue_channel = rgb_image[:, :, 2]

    features = [
        np.mean(red_channel),
        np.mean(green_channel),
        np.mean(blue_channel),
        np.std(red_channel) + np.std(green_channel) + np.std(blue_channel)  # Sum of std deviations
    ]

    return np.array(features)

# Usage
car_images, labels = load_and_preprocess_data_rgb()

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
                    influence = np.exp(-dist_to_winner / (2 * (self.radius ** 2)))
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


def visualize_som_2d_colored(som, data, labels, map_size, ax):
    """
    Visualize the trained SOM with the given data colored based on the given labels.
    """
    colors = {1: 'red',  # Car
              9: 'blue',  # Truck
              'unknown': 'yellow'}

    # Initialize matrix to count cars and trucks in each cell
    cars_count_matrix = np.zeros(map_size)
    trucks_count_matrix = np.zeros(map_size)

    # Count the number of cars and trucks for each SOM cell
    for idx, input_vector in enumerate(data):
        x, y = som._find_winner(input_vector)
        if labels[idx] == 1:
            cars_count_matrix[x, y] += 1
        else:
            trucks_count_matrix[x, y] += 1

    # Plotting the SOM nodes with colors based on the dominant vehicle type
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            cars_count = cars_count_matrix[i, j]
            trucks_count = trucks_count_matrix[i, j]
            if cars_count == 0 and trucks_count == 0:
                color = colors['unknown']
                label = 'Unknown'
            elif cars_count > trucks_count:
                color = colors[1]
                label = f'Car (n={int(cars_count)})'
            else:
                color = colors[9]
                label = f'Truck (n={int(trucks_count)})'

            ax.scatter(i, j, marker='s', s=600, color=color, edgecolors='k', linewidths=2, label=label)

            # Adding Data Points
            intensity = (som.weights[i, j, 0] - 0) / 1
            cmap = plt.cm.viridis
            color = cmap(intensity)
            ax.scatter(i, j, marker='o', s=50, color=color, edgecolors='k', alpha=0.7)

    # Custom legend
    custom_lines = [Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[9], lw=4),
                    Line2D([0], [0], color=colors['unknown'], lw=4)]
    ax.legend(custom_lines, ['Car', 'Truck', 'Unknown'])

    ax.set_xticks(range(map_size[0]))
    ax.set_yticks(range(map_size[1]))
    ax.grid(True)


def graphs1(learning_rates, map_sizes, radii, feature_vectors, output_directory):
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


def graphs2(learning_rates, map_sizes, radii, feature_vectors, labels, output_directory):
    for lr in learning_rates:
        for size in map_sizes:
            for r in radii:
                fig, ax = plt.subplots(figsize=(size[0] + 2, size[1] + 2))

                som = SOM(input_dim=4, map_size=size, learning_rate=lr, radius=r)
                som.train(feature_vectors, epochs=10)

                visualize_som_2d_colored(som, feature_vectors, labels, size, ax)
                ax.margins(0.15, 0.15)

                title = f"Learning Rate: {lr}, Size: {size}, Radius: {r}"
                ax.set_title(title, fontsize=12)

                # Save the plot to the directory
                # Save the plot to the directory
                filename = f"LR_{lr}_Size_{size[0]}x{size[1]}_Radius_{r}.png"
                filepath = os.path.join(output_directory, filename)
                plt.savefig(filepath)

                plt.close(fig)  # Close the figure after saving

def single_plot(learning_rates, map_sizes, radii, feature_vectors, labels):
    figure_size = (map_sizes[0] + 2, map_sizes[1] + 2)
    fig, ax = plt.subplots(figsize=figure_size)
    som = SOM(input_dim=4, map_size=map_sizes, learning_rate=learning_rates, radius=radii)
    som.train(feature_vectors, epochs=1000)

    visualize_som_2d_colored(som, feature_vectors, labels, map_sizes, ax)
    ax.margins(0.15, 0.15)

    title = f"Learning Rate: {learning_rates}, Size: {map_sizes}, Radius: {radii}"
    ax.set_title(title, fontsize=12)

    # Save the plot to the directory
    # Save the plot to the directory
    filename = f"LR_{learning_rates}_Size_{map_sizes[0]}x{map_sizes[1]}_Radius_{radii}.png"

    plt.show()

def main():
    car_images, edge_images, labels = load_and_preprocess_data()
    feature_vectors = np.array([extract_features_from_edge_image(img) for img in edge_images])

    car_images_rgb, labels_rgb = load_and_preprocess_data_rgb()
    feature_vectors_rgb = np.array([extract_features_from_rgb_image(img) for img in car_images_rgb])

    # Visualization generations based on different parameters
    learning_rates = [0.1, 0.5, 0.9]
    radii = [0.5, 1.0, 2.0]
    map_sizes = [(5, 5), (10, 10), (15, 15), (20, 20)]
    single_plot(0.95,(15,15),1.0,feature_vectors_rgb, labels)
    single_plot(0.95, (15, 15), 1.0, feature_vectors, labels)

    # Create a directory named by the current date and time
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_directory = os.path.join(os.getcwd(), "..", "results", "GRAPH1")
    # os.makedirs(output_directory, exist_ok=True)
    # graphs1(learning_rates,map_sizes,radii, feature_vectors, output_directory)
    #
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # output_directory = os.path.join(os.getcwd(), "..", "results", "GRAPH2")
    # os.makedirs(output_directory, exist_ok=True)
    # graphs2(learning_rates, map_sizes, radii, feature_vectors, labels, output_directory)
    plot_images(car_images, edge_images)

if __name__ == "__main__":
    main()
