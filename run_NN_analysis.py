from nn_analysis_funcs import (import_data, pre_process_data,
                               show_confusion_matrix, find_confusion_matrix,
                               show_learning, generator)
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint


def main(batch_size=16):
    from functional_NN_EMG import define_NN_architecture
    from math import floor
    nn_data = import_data('waveletdata.pkl')
    (train_data_x, train_data_x_rms, train_data_y) = pre_process_data(nn_data)

    print('Running NN...')
    model = define_NN_architecture()

    train_step_length = int(0.7*len(train_data_y))
    val_length = int(0.2*len(train_data_y))
    test_length = int(0.1*len(train_data_y))

    train_gen = generator(train_data_x, train_data_x_rms,
                          train_data_y, batch_size,
                          0, train_step_length)

    val_gen = generator(train_data_x, train_data_x_rms,
                        train_data_y, batch_size,
                        train_step_length + 1, train_step_length + val_length)

    adadelta_optim = Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adadelta_optim,
                  metrics=['accuracy'])

    checkpoint_func = ModelCheckpoint('best_two_branch_NN_EMG.hdf5',
                                      monitor='val_acc',
                                      save_best_only=True,
                                      mode='max')

    train_step_num = floor(train_step_length/batch_size)
    val_step_num = floor(val_length/batch_size)

    hist = model.fit_generator(train_gen,
                               steps_per_epoch=train_step_num,
                               epochs=35,
                               validation_data=val_gen,
                               validation_steps=val_step_num,
                               callbacks=[checkpoint_func])

    conf_matrix = find_confusion_matrix(model,
                                        [train_data_x[-test_length:],
                                         train_data_x_rms[-test_length:]],
                                         train_data_y[-test_length:])

    show_confusion_matrix(conf_matrix)
    show_learning(hist)

    return model, hist


if __name__ == '__main__':
    model, hist = main()
