import tensorflow as tf
from tensorflow import keras
import matplotlib.pylab as plt
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full,y_train_full),(X_test,y_test) = fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000]/255.0 , X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]



encoder = keras.models.load_model("encoder.h5")

input_train = y_train
codings_train = encoder.predict(X_train)

input_validation = y_valid
codings_validation = encoder.predict(X_valid)



model_label_encoder = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[1]),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(30, activation="selu"),
])



model_label_encoder.compile(loss="mean_squared_error", optimizer="sgd")
history = model_label_encoder.fit(
    input_train,codings_train,
    epochs=30,
    validation_data=(input_validation,codings_validation)
)



model_label_encoder.save("label_encoder.h5")


