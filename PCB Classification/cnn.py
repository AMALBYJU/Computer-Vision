import keras
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model

x_train = np.load("xtrain.npy")
y_train = np.load("ytrain.npy")

x_test = np.load("xtest.npy")
y_test = np.load("ytest.npy")

print('Training data shape : ', x_train.shape, y_train.shape)
print('Testing data shape : ', x_test.shape, y_test.shape)

# Display the first image in training data
'''
plt.figure(figsize=[5, 5])
plt.imshow(x_train[1,:,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(y_train[0]))
plt.show()
'''
x_train = x_train.reshape(-1, 300, 300, 3)
x_test = x_test.reshape(-1, 300, 300, 3)

x_train = x_train.astype('float32')
x_train = x_train / 255.
x_test = x_test.astype('float32')
x_test = x_test / 255.

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# x_train is training data
# x_valid is validation data

x_train, x_valid, train_label, valid_label = train_test_split(x_train, y_train_one_hot, test_size=0.2, random_state=13)

print(x_train.shape, x_valid.shape, train_label.shape, valid_label.shape)

batch_size = 64
epochs = 6
num_classes = 2

pcb_model = Sequential()
pcb_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(300, 300, 3)))
pcb_model.add(LeakyReLU(alpha=0.1))
pcb_model.add(MaxPooling2D((2, 2), padding='same'))
pcb_model.add(Dropout(0.25))
pcb_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
pcb_model.add(LeakyReLU(alpha=0.1))
pcb_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
pcb_model.add(Dropout(0.25))
pcb_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
pcb_model.add(LeakyReLU(alpha=0.1))
pcb_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
pcb_model.add(Dropout(0.4))
pcb_model.add(Flatten())
pcb_model.add(Dense(128, activation='linear'))
pcb_model.add(LeakyReLU(alpha=0.1))
pcb_model.add(Dropout(0.3))
pcb_model.add(Dense(num_classes, activation='softmax'))

pcb_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])

# TRAINING

trained_pcb_model = pcb_model.fit(x_train, train_label, batch_size=batch_size, epochs=epochs, verbose=1,
                                validation_data=(x_valid, valid_label))

pcb_model.save("trained_pcb_model.h5py")

# TESTING
test_eval = pcb_model.evaluate(x_test, y_test_one_hot, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = pcb_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)

correct = np.where(predicted_classes == y_test)[0]
print("Found " + str(len(correct)) + " correct labels")

incorrect = np.where(predicted_classes != y_test)[0]
print("Found " + str(len(incorrect)) + " incorrect labels")

target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names=target_names))
