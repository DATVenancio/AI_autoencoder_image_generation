import tensorflow as tf
from tensorflow import keras
import matplotlib.pylab as plt
import numpy as np

codings_size=30
model_decoder = keras.models.load_model("decoder.h5")
model_label_encoder = keras.models.load_model("label_encoder.h5")


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]


image_labels = np.array([[0],[0],[0],[0],[0],[0]])



codings = model_label_encoder.predict(image_labels)

print(codings[0]==codings[1])


images=model_decoder.predict(codings)

plt.imshow(images[0], cmap="binary")


plt.tight_layout()

# Exibir as figuras
plt.show()

