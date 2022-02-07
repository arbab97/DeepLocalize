import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.keras import activations
import keras
print('Loadind data......')
num_classes = 10
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Preparing data......')
#train_images -= int(np.mean(train_images))
#train_images = train_images /int(np.std(train_images))
#test_images -= int(np.mean(test_images))
#test_images = test_images / int(np.std(test_images))
train_images = train_images / 255
test_images  = test_images / 255
training_data = train_images.reshape(60000, 1, 28, 28)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]
#Data Slicing for check
size=5
training_data = training_data[:size]
training_labels =training_labels[:size]
testing_data =testing_data[:size]
testing_labels = testing_labels[:size]

# learn rate should be 0.01
# callback = keras.callbacks.DeepLocalize(inputs, outputs, layer_number, batch_size, startTime)
callback = keras.callbacks.DeepLocalize(np.arange(100).reshape(5, 20), np.zeros(5), 3, 1, 0)

model = keras.models.Sequential()
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation(activations.relu))
model.add(keras.layers.Dense(1))
model.compile(keras.optimizers.SGD(), loss='mse')
model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10, batch_size=1, 
                callbacks=[callback], verbose=0)
print("End")