import numpy

import keras
from keras.utils import plot_model
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, LSTM, Input, concatenate
from keras import backend as K


def build_model(num_features, num_classes, learning_rate):
    model = Sequential()

    model.add(Dense(32, activation='relu', input_shape=(num_features, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer
    # opt = keras.optimizers.Adadelta(lr=learning_rate)
    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.9, decay=0.0, nesterov=False)
    # opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt,
                  metrics=['accuracy'])

    return model


def load_model(time_path):
    path = './files/training_logs/' + time_path + '/model'

    # Load json and create the model
    with open(path + '/model_config.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights(path + '/model_weights.h5')
    print('Loaded model from disk')

    return model


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = numpy.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = numpy.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0 * batch_size * (shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = numpy.round(total_memory / (1024.0 ** 3), 3)

    print('Approximately memory usage : {} gb'.format(gbytes))


def save_model(model, classes, output_dir):
    print('Saving model details')
    # Save model config
    model_json = model.to_json()
    with open(output_dir + 'model_config.json', 'w') as f:
        f.write(model_json)

    # Save model summary
    plot_model(model, to_file=output_dir + 'model_summary.png', show_shapes=True, show_layer_names=True)

    with open(output_dir + 'model_summary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    # Save labels
    with open(output_dir + 'trained_labels.txt', 'w') as f:
        f.write('\n'.join(classes) + '\n')

    print('Saving trained model')
    model.save_weights(output_dir + 'model_weights.h5')
