import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_data_subset(percentage=0.1):
    (x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()

    # Calculate the number of examples to include
    num_train = int(len(x_train_full) * percentage)
    num_test = int(len(x_test_full) * percentage)

    # Randomly select a subset of the data
    train_indices = np.random.choice(len(x_train_full), num_train, replace=False)
    test_indices = np.random.choice(len(x_test_full), num_test, replace=False)

    x_train = x_train_full[train_indices]
    y_train = y_train_full[train_indices]
    x_test = x_test_full[test_indices]
    y_test = y_test_full[test_indices]

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode labels
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def build_model():
    # Define a simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, x_test, y_test, class_names):
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Overall Test accuracy: {accuracy:.4f}")

    # Predictions
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Accuracy for specific classes
    for class_index in range(len(class_names)):
        class_accuracy = np.mean([
            predicted_classes[i] == true_classes[i] for i in range(len(predicted_classes)) 
            if true_classes[i] == class_index
        ])
        print(f"Class '{class_names[class_index]}' accuracy: {class_accuracy:.4f}")

# Load and preprocess data
print("Loading data.")
x_train, y_train, x_test, y_test = load_data_subset(0.1)

# Train the original model
print("Training original model...")
original_model = build_model()
original_model.fit(x_train, y_train, epochs=10, verbose=0)

# Apply label flipping (birds to airplanes)
bird_indices = np.where(np.argmax(y_train, axis=1) == 2)[0]
y_train[bird_indices] = to_categorical(0, 10)

# Train the model after label flipping
print("Training model after label flipping...")
flipped_model = build_model()
flipped_model.fit(x_train, y_train, epochs=10, verbose=0)

# Pick 5 random 'bird' images from the test set
num_examples = 5
bird_indices_test = np.where(np.argmax(y_test, axis=1) == 2)[0]
random_bird_indices = np.random.choice(bird_indices_test, num_examples, replace=False)

# Create a figure to visualize the images and predictions
plt.figure(figsize=(15, 6))

for i, idx in enumerate(random_bird_indices):
    bird_image = x_test[idx]
    bird_image_expanded = np.expand_dims(bird_image, axis=0)

    # Predict the class using both models
    original_pred_class = np.argmax(original_model.predict(bird_image_expanded), axis=1)[0]
    flipped_pred_class = np.argmax(flipped_model.predict(bird_image_expanded), axis=1)[0]

    # Plotting
    plt.subplot(2, num_examples, i + 1)
    plt.imshow(bird_image, cmap=plt.cm.binary)
    plt.title(f"{class_names[2]}")
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, num_examples, i + 1 + num_examples)
    plt.imshow(bird_image, cmap=plt.cm.binary)
    plt.title(f"Orig: {class_names[original_pred_class]}\nFlip: {class_names[flipped_pred_class]}")
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.savefig('poisoned_test.png')