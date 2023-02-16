# Import:
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras import layers, models

# Algoritmo:
# Il programma classifica gli oggetti del cifar10 dataset usando le cnn.

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

"""
# Plotting di un'immagine
plt.imshow(train_images[0], interpolation="nearest")
plt.show()
"""

# Inizializzo il modello
model = models.Sequential()

# Aggiungo un layer convolutional di dimensioni 32*32*3, con 32 filtri di dimensioni 3*3 e attivazione relu (rectified linear unit)
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(32, 32, 3), activation="relu"))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))

model.add(layers.Flatten())
model.add(layers.Dense(units=64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))  # Ci sono 10 classi finali

print(model.summary())

model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

history = model.fit(x=train_images, y=train_labels, epochs=7, verbose=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)


def predict(number):
    print(test_images[number])
    prediction = model.predict(test_images[number])
    print("The model predicted:", CLASSES[np.argmax(prediction)])
    plt.imshow(test_images[number], interpolation="nearest")
    plt.show()
    print("The actual answer is:", test_labels[number])


def take_input():
    ok = False
    number = 0
    while not ok:
        number = int(input("Insert a number between 0 and 10'000: "))
        if 0 <= number < 10000:
            ok = True
    predict(number)


take_input()
