from pathlib import Path
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from tensorflow.keras.layers import Input, Dropout, Activation, Dense, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def cnn_model(params):
    cnn = Sequential()
    cnn.add(Conv3D(filters=8, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                   padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                   bias_initializer='zeros', activation="relu", name="conv3d_1"))

    cnn.add(MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_1"))

    cnn.add(Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                   padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                   bias_initializer='zeros', activation="relu", name="conv3d_2"))

    cnn.add(MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_2"))

    cnn.add(Conv3D(filters=32, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                   padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                   bias_initializer='zeros', activation="relu", name="conv3d_3"))

    cnn.add(MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_3"))

    cnn.add(Conv3D(filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                   padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                   bias_initializer='zeros', activation="relu", name="conv3d_4"))

    cnn.add(MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_4"))

    cnn.add(Conv3D(filters=128, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                   padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                   bias_initializer='zeros', activation="relu", name="conv3d_5"))

    cnn.add(MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_5"))

    cnn.add(Flatten(name="flatten_1"))

    cnn.add(Dense(128, activation="relu",
                  kernel_initializer=tf.keras.initializers.RandomNormal(
                      mean=0., stddev=1.),
                  bias_initializer='zeros', name="dense_1"))

    cnn.add(Dense(128, activation="relu",
                  kernel_initializer=tf.keras.initializers.RandomNormal(
                      mean=0., stddev=1.),
                  bias_initializer='zeros', name="dense_2"))

    cnn.add(Dense(1, activation="linear",
                  kernel_initializer=tf.keras.initializers.VarianceScaling(
                      scale=1., mode="fan_avg", distribution="uniform", seed=None),
                  bias_initializer='zeros', name="dense_3"))

    cnn.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')

    return cnn


def context_aware_model(params, cnn_dir):
    cnn = load_model(cnn_dir)

    cnn_input = cnn.layers[0].input
    cnn_l10_output = cnn.layers[10].output

    # make first 10 layers non-trainable
    for layer in cnn.layers[:11]:
        layer.trainable = False

    # to-do transform the output here

    # input_1 = Input(shape=(15360))
    # branch_1 = Dropout(.2)(input_1)
    branch_1 = BatchNormalization()(cnn_l10_output)
    branch_1 = Dropout(.2)(branch_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Dropout(.2)(branch_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    input_2 = Input(shape=(132))
    branch_2 = Activation('relu')(input_2)

    x = concatenate([branch_1, branch_2])
    # x = BatchNormalization()(x)
    x = Dropout(.2)(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)

    model2 = tf.keras.Model(inputs=[cnn_input, input_2], outputs=x)
    model2.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')
    return model2

def enhanced_context_aware_model(params, cnn_dir):
    cnn = load_model(cnn_dir)

    cnn_input = cnn.layers[0].input
    cnn_l10_output = cnn.layers[10].output

    # make first 10 layers non-trainable
    for layer in cnn.layers[:11]:
        layer.trainable = False

    branch_1 = BatchNormalization()(cnn_l10_output)
    branch_1 = Dropout(.2)(branch_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Dropout(.2)(branch_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    input_2 = Input(shape=(132))
    branch_2 = Dense(128)(input_2)
    branch_2 = Dropout(.2)(branch_2)
    branch_2 = Activation('relu')(input_2)

    x = concatenate([branch_1, branch_2])
    x = Dropout(.2)(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)

    model2 = tf.keras.Model(inputs=[cnn_input, input_2], outputs=x)
    model2.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')
    return model2


def load_saved_model(args, paths):
    if args.model == 'cnn':
        model_dir = paths['cnn_model']
    elif args.model == 'context_aware':
        model_dir = paths['context_aware_model']

    saved_model_dir = Path(model_dir, 'model')
    assert saved_model_dir.exists, f'{args.model} model is not saved at: {saved_model_dir}'
    model = load_model(saved_model_dir)
    return model
