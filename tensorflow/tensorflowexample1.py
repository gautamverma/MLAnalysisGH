import numpy as np
import tensorflow as tf

from tensorflow import keras

# Successive layers are defined in Sequence
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

X = np.array([-1, 0, 1, 2, 3, 4], dtype=np.int32)
Y = np.array([-3,-1, 1, 3, 5, 7], dtype=np.int32)

# Goes thorough the training loop 500 times
model.fit(X, Y, epochs=500)

print("for 10 output it " , model.predict([10]))

