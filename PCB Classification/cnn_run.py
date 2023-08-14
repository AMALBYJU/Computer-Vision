import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from keras.models import load_model

x_test = np.load("xtest.npy")
y_test = np.load("ytest.npy")
num_classes = 2

print('Testing data shape : ', x_test.shape, y_test.shape)

x_test = x_test.reshape(-1, 300, 300, 3)

x_test = x_test.astype('float32')
x_test = x_test / 255.

y_test_one_hot = to_categorical(y_test)

pcb_model = load_model("trained_pcb_model.h5py")

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
