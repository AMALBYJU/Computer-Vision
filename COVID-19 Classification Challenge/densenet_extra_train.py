import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from sklearn.model_selection import train_test_split

x_train = np.load("trainx.npy")
y_train = np.load("trainy.npy")

num_classes = 3
input_shape = (512, 512, 3)
img_height = 512
img_width = 512
img_depth = 3

batch_size = 4  # was 128
epochs = 10
learning_rate = 1e-5

x_train = x_train.reshape(-1, 512, 512, 3)
x_train = x_train.astype('float32')
x_train = x_train / 255.

# Converted to one-hot encoding
y_train = to_categorical(y_train, num_classes)

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05, random_state=13)

model = load_model("trained_densenet_model.h5")

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_valid, y_valid))

model.save("trained_densenet_model2.h5")

