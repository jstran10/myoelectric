from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import pickle
from random import uniform


def import_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def pre_process_data(nn_data):
    # Set up training and test data
    rest_label = nn_data[0]['label']

    train_data_x_wavelet = []
    train_data_x_rms = []
    train_data_x_emg=[]
    train_data_y = []

    rest_label = nn_data[0]['label']

    iterator = -1

    for trial in nn_data:
        iterator += 1

        label_append = np.where(trial['label'] == trial['label'].max())[0][0]

        if all(trial['label'] == rest_label):
            if (uniform(1, 16) > 15):
                train_data_x_wavelet.append(trial['wavelet'])
                train_data_x_rms.append(trial['rms'])
                train_data_x_emg.append(trial['emg'])
                train_data_y.append(label_append)

        else:
            train_data_x_wavelet.append(trial['wavelet'])
            train_data_x_rms.append(trial['rms'])
            train_data_x_emg.append(trial['emg'])
            train_data_y.append(label_append)

    train_data_y = to_categorical(train_data_y, num_classes=18)

    train_data_x_wavelet = np.reshape(
            train_data_x_wavelet,
            (len(train_data_x_wavelet), 248, 16, 1)
            )
    train_data_x_rms = np.reshape(
            train_data_x_rms,
            (len(train_data_x_rms), 16)
            )

    train_data_x_emg = np.reshape(
            train_data_x_emg,
            (len(train_data_x_emg), 200, 16, 1)
            )
    rng_state = np.random.get_state()
    np.random.shuffle(train_data_y)
    np.random.set_state(rng_state)
    np.random.shuffle(train_data_x_wavelet)
    np.random.set_state(rng_state)
    np.random.shuffle(train_data_x_rms)
    np.random.set_state(rng_state)
    np.random.shuffle(train_data_x_emg)

    return train_data_x_wavelet, train_data_x_rms, train_data_x_emg, train_data_y


def find_confusion_matrix(model, x_in, y_actual, categorical=True):
    prediction = model.predict(x_in)

    if categorical:
        predicted_data_y = []
        for preds in prediction:
            loc = np.where(preds == preds.max())
            predicted_data_y.append(loc[0][0])

        tested_data_y = []
        for preds in y_actual:
            loc = np.where(preds == preds.max())
            tested_data_y.append(loc[0][0])
    else:
        tested_data_y = y_actual

    conf_matrix = confusion_matrix(tested_data_y, predicted_data_y)
    return conf_matrix


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

    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def show_confusion_matrix(mat):
    plt.figure(constrained_layout=True)
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
    plot_confusion_matrix(mat, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure(constrained_layout=True)
    plot_confusion_matrix(mat, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()


def show_learning(hist):
    plt.figure()
    plt.plot(hist.epoch, hist.history['loss'], linestyle='solid', marker='o',
             linewidth=1.2, label='Training')
    plt.plot(hist.epoch, hist.history['val_loss'], linestyle='solid',
             marker='o', linewidth=1.2, label='Validation')
    plt.ylabel('Validation Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.figure()
    plt.plot(hist.epoch, hist.history['acc'], linestyle='solid', marker='o',
             linewidth=1.2, label='Training')
    plt.plot(hist.epoch, hist.history['val_acc'], linestyle='solid',
             marker='o', linewidth=1.2, label='Validation')
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend(loc='lower right')

    plt.show()


def generator(x1, x2, x3, y1, batch_size, min_index, max_index):
    import numpy as np
    if max_index is None:
        max_index = len(x1) - 1

    i = min_index

    while True:
        if i + batch_size >= max_index:
            i = min_index

        rows = np.arange(i, min(i + batch_size, max_index))
        i += len(rows)

        samples = np.zeros((len(rows), 248, 16, 1))
        samples_rms = np.zeros((len(rows), 16))
        samples_emg = np.zeros((len(rows), 200, 16, 1))
        targets = np.zeros((len(rows), 18))

        for j, row in enumerate(rows):
            samples[j] = x1[row]
            samples_rms[j] = x2[row]
            samples_emg[j] = x3[row]
            targets[j] = y1[row]

        yield ([samples, samples_rms, samples_emg], targets)
       # yield ([samples], targets)

#data = {'emg':[1,2,3], 'rms':[1,2,3],'wavelet':[1,2,3],'label':5}
#data['emg'] will get all of that
#so now need to add        