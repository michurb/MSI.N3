import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.datasets import cifar10
# import to plotowania obarzkow
from plot import plot_images

# Wczytanie danych
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Wybieramy tylko obrazy samochodów (klasa 1 w CIFAR-10)
car_indices = np.where((y_train == 1) | (y_train == 9))[0]
car_images = x_train[car_indices]
print(np.sum(car_images))
# Konwersja obrazów do obrazów krawędziowych
edge_images = [cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200) for img in car_images]
edge_images = np.array(edge_images)

# Wyświetlenie kilku obrazów oryginalnych i ich odpowiedników krawędziowych

#plot_images(car_images, edge_images)

def extract_features_from_edge_image(edge_image):
    mean_intensity = np.mean(edge_image)
    white_pixel_count = np.sum(edge_image > 0)
    std_dev = np.std(edge_image)
    return [mean_intensity, white_pixel_count, std_dev]

feature_vectors = np.array([extract_features_from_edge_image(img) for img in edge_images])

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
        min_dist = float('inf')
        winner_coords = None
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist = self._calculate_distance(input_vector, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    winner_coords = (i, j)
        return winner_coords

    def _update_weights(self, input_vector, winner_coords):
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                dist_to_winner = self._calculate_distance(np.array([i, j]), np.array(winner_coords))
                if dist_to_winner <= self.radius:
                    influence = np.exp(-dist_to_winner / (2 * (self.radius**2)))
                    self.weights[i, j] += self.learning_rate * influence * (input_vector - self.weights[i, j])

    def train(self, data, epochs):
        for epoch in range(epochs):
            for input_vector in data:
                winner_coords = self._find_winner(input_vector)
                self._update_weights(input_vector, winner_coords)

# Przetwarzanie obrazów na wektory
car_vectors = car_images.reshape(car_images.shape[0], -1)

# Inicjalizacja i trening sieci SOM
som = SOM(input_dim=3, map_size=(10, 10), learning_rate=0.5, radius=1.0)
som.train(feature_vectors, epochs=10)


def visualize_som_2d(som, data, map_size):
    fig, ax = plt.subplots(figsize=(10, 10))

    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    cmap = plt.cm.viridis

    # Rysowanie próbek jako punktów
    for input_vector in data:
        x, y = som._find_winner(input_vector)
        intensity = (input_vector[0] - min_val[0]) / (max_val[0] - min_val[0])
        color = cmap(intensity)
        ax.scatter(x, y, marker='o', s=50, color=color, edgecolors='k')

    # Rysowanie reprezentantów klastrów jako większych okręgów
    for i in range(map_size[0]):
        for j in range(map_size[1]):
            intensity = (som.weights[i, j, 0] - min_val[0]) / (max_val[0] - min_val[0])
            color = cmap(intensity)
            ax.scatter(i, j, marker='o', s=200, color=color, edgecolors='k', linewidths=2)

    plt.xticks(range(map_size[0]))
    plt.yticks(range(map_size[1]))
    plt.grid(True)
    plt.show()



# Wizualizacja wyników w przestrzeni 2D

visualize_som_2d(som, feature_vectors, (10, 10))
