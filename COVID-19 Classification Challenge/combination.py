import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from sklearn.model_selection import train_test_split

x_test = np.load("testx.npy")
y_test = np.load("testy.npy")

num_classes = 3

x_test = x_test.reshape(-1, 512, 512, 3)
x_test = x_test.astype('float32')
x_test = x_test / 255.

# Converted to one-hot encoding
y_test = to_categorical(y_test, num_classes)

members = [load_model("trained_densenet_model.h5"), load_model("trained_inception_model.h5")]


# TESTING
'''
test_eval = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
'''

predictions = [model.predict(x_test) for model in members]
predictions = np.array(predictions)
predictions[0] = predictions[0] * 0.3
predictions[1] = predictions[1] * 0.7
summed = np.sum(predictions, axis=0)
predicted_classes = np.argmax(np.round(summed), axis=1)
y_actual_classes = np.argmax(np.round(y_test), axis=1)  # Converted one-hot to integer

correct = np.where(predicted_classes == y_actual_classes)[0]
print("Found " + str(len(correct)) + " correct labels")

incorrect = np.where(predicted_classes != y_actual_classes)[0]
print("Found " + str(len(incorrect)) + " incorrect labels")


print("CONFUSION MATRIX")
print(confusion_matrix(y_true=y_actual_classes, y_pred=predicted_classes))

print("CLASSIFICATION REPORT")
target_names = ["COVID-19", "Normal", "Viral Pneumonia"]
print(classification_report(y_actual_classes, predicted_classes, target_names=target_names, digits = 5))

