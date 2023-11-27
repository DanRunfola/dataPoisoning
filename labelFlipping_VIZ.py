import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 data
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize pixel values
x_train = x_train / 255.0

# Class names in CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Label Flipping: Flip labels of 'bird' (class 2) to 'airplane' (class 0)
bird_indices = np.where(y_train == 2)[0]
np.random.shuffle(bird_indices)
num_flipped = 1000  # Number of images to flip
y_train[bird_indices[:num_flipped]] = 0

# Select a few images to visualize the label flipping
num_images = 10
flipped_indices = bird_indices[:num_images]
selected_images = x_train[flipped_indices]
selected_labels = y_train[flipped_indices]

# Create a grid of images
plt.figure(figsize=(15, 5))
for i in range(num_images):
    plt.subplot(2, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(selected_images[i], cmap=plt.cm.binary)
    plt.xlabel(f"Original: Bird\nFlipped: {class_names[selected_labels[i][0]]}")
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("cifar10_label_flipping.png")
