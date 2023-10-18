from matplotlib import pyplot as plt

def plot_images(car_images, edge_images):
    num_samples_to_display = 5

    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10, 10))

    for i in range(num_samples_to_display):
        # Oryginalny obraz
        axes[i, 0].imshow(car_images[i])
        axes[i, 0].set_title("Oryginalny obraz")
        axes[i, 0].axis('off')

        # Obraz krawędziowy
        axes[i, 1].imshow(edge_images[i], cmap='gray')
        axes[i, 1].set_title("Obraz krawędziowy")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()