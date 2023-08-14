# Implementation of InceptionV3 in Keras

import tensorflow as tf

import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
# demonstrate data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


x_train = np.load("trainx.npy")
y_train = np.load("trainy.npy")

# Formatting the data

x_train = x_train.reshape(-1, 512, 512, 3)
x_train = x_train.astype('float32')
x_train = x_train / 255.

input_shape = (512, 512, 3)
num_classes = 3
img_height = 512
img_width = 512
img_depth = 3

batch_size = 4  # was 128
epochs = 16
learning_rate = 0.02

# One-hot encoding for 3 classes.
y_train = keras.utils.to_categorical(y_train, num_classes)


'''
plt.figure(figsize=[5, 5])
plt.imshow(x_train[1500,:,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[1500]))
plt.show()
'''

# Data normalization
# create scaler

# Splitting input data (x_train, y_train) into 3 partitions: 80% for training, 20% for dev/validation
# and 20% for testing).

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05, random_state=17)

model = keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(512, 512, 3),
    pooling=max
)

x = model.output
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
predictions = keras.layers.Dense(num_classes, activation='softmax')(x)
model = keras.models.Model(inputs=model.input, outputs=predictions)

'''
'''
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()

# Comment out the below line if you want to have an image of your model's structure.

# tf.keras.utils.plot_model( model , show_shapes=True )

"""## 4) Training the Model

We'll train the model now.
"""

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid))

# Save model in file
model.save("trained_inception_model.h5")