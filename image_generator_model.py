import tensorflow as tf
from tensorflow import keras
import matplotlib.pylab as plt

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]




encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])
decoder = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[30]),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
model_autoencoder = keras.models.Sequential([encoder, decoder])
model_autoencoder.compile(loss="binary_crossentropy",optimizer=keras.optimizers.SGD(learning_rate=1.5))

history = model_autoencoder.fit(X_train, X_train, epochs=10,validation_data=[X_valid, X_valid])



model_autoencoder.save("autoencoder.h5")
encoder.save("encoder.h5")
decoder.save("decoder.h5")










