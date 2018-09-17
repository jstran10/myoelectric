import keras
from keras.models import load_model

import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import confusion_matrix

model = load_model('second_EMG_NN.h5')

with open('waveletdata.pkl', 'rb') as f:
    nn_data = pickle.load(f)

# Set up training and test data
train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []

iterator = 0
for trial in nn_data:
    if iterator % 4 == 0:
        test_data_x.append(trial['wavelet'])
        test_data_y.append(
                np.where(trial['label'] == trial['label'].max())[0][0]
                )
    else:
        train_data_x.append(trial['wavelet'])
        train_data_y.append(
                np.where(trial['label'] == trial['label'].max())[0][0]
                )
    iterator += 1

train_data_y = keras.utils.to_categorical(train_data_y, num_classes=18)
test_data_y = keras.utils.to_categorical(test_data_y, num_classes=18)

test_data_x = np.reshape(test_data_x, (4677, 248, 16))
train_data_x = np.reshape(train_data_x, (14031, 248, 16))

prediction = model.predict(test_data_x)

predicted_data_y = []
for preds in prediction:
    loc = np.where(preds == preds.max())
    predicted_data_y.append(loc[0][0])

tested_data_y = []
for preds in test_data_y:
    loc = np.where(preds == preds.max())
    tested_data_y.append(loc[0][0])

cnf_matrix = confusion_matrix(tested_data_y, predicted_data_y)

print(cnf_matrix)

class_names = ['Rest',
               'Thumb Up',
               'Extension index, middle',
               'Extension index, middle, thumb',
               'Thumb opposing base of little finger',
               'Abduction of all fingers',
               'Fingers flexed into fist',
               'Pointing index',
               'Adduction of extended fingers',
               'Wrist supination (axis: MF)',
               'Wrist pronation (axis: MF)',
               'Wrist supination (axis: LF)',
               'Wrist pronation (axis: LF)',
               'Wrist flexion',
               'Wrist extension',
               'Wrist radial deviation',
               'Wrist ulnar deviation',
               'Wrist extension with closed hand']


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plt.figure(constrained_layout=True)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure(constrained_layout=True)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
