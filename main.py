#from tutorial https://www.youtube.com/watch?v=Edhv7-4t0lc&t=1397s
#explains the fit function https://keras.io/api/models/model_training_apis/

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i], cmap='gray')
#plt.show()

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 64
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,shuffle=True, verbose=2)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,shuffle=True, verbose=2)

model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

probability_model = keras.Sequential([
    model,
    keras.layers.Softmax()
])
predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

