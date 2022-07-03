import logging
import pickle
import tensorflow as tf
from pathlib import Path
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


def create_transform_layer(paths):

    def transform(X):
        mm_scaler_path = Path(paths['cnn_model'], 'min_max_scaler')
        sc_scaler_path = Path(paths['cnn_model'], 'std_scaler')

        with open(mm_scaler_path, 'rb') as mm_scaler_file:
            MM = pickle.load(mm_scaler_file)

        with open(sc_scaler_path, 'rb') as sc_scaler_file:
            SC = pickle.load(sc_scaler_file)

        X = tf.math.multiply(X, MM.scale_)
        X = tf.math.add(X, MM.min_)
        if SC.with_mean:
            X = tf.math.subtract(X, SC.mean_)
        if SC.with_std:
            X = tf.math.div(X, SC.scale_)
        return X

    return transform


def context_aware_model(params):

    input_1 = Input(shape=(15360))
    branch_1 = Dropout(.2)(input_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)
    branch_1 = Dropout(.2)(branch_1)
    branch_1 = Dense(128)(branch_1)
    branch_1 = Activation('relu')(branch_1)

    input_2 = Input(shape=(132))
    branch_2 = Activation('relu')(input_2)

    x = concatenate([branch_1, branch_2])
    x = Dropout(.2)(x)
    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = Dense(10)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)

    model2 = tf.keras.Model(inputs=[input_1, input_2], outputs=x)
    model2.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')
    return model2


def unfreezed_context_aware_model(params):
    input_1 = Input(shape=(172, 220, 156, 1))
    branch1 = Conv3D(filters=8, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                     padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                     bias_initializer='zeros', activation="relu", name="conv3d_1")(input_1)

    branch1 = MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_1")(branch1)

    branch1 = Conv3D(filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                     padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                     bias_initializer='zeros', activation="relu", name="conv3d_2")(branch1)

    branch1 = MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_2")(branch1)

    branch1 = Conv3D(filters=32, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                     padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                     bias_initializer='zeros', activation="relu", name="conv3d_3")(branch1)

    branch1 = MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_3")(branch1)

    branch1 = Conv3D(filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                     padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                     bias_initializer='zeros', activation="relu", name="conv3d_4")(branch1)

    branch1 = MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_4")(branch1)

    branch1 = Conv3D(filters=128, kernel_size=[3, 3, 3], strides=[1, 1, 1],
                     padding="same", data_format="channels_last", dilation_rate=[1, 1, 1],
                     bias_initializer='zeros', activation="relu", name="conv3d_5")(branch1)

    branch1 = MaxPool3D(pool_size=(2, 2, 2), strides=(
        2, 2, 2), padding='valid', data_format="channels_last", name="maxpool3d_5")(branch1)

    branch1 = Flatten(name="flatten_1")(branch1)

    branch_1 = BatchNormalization()(branch1)
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

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=x)
    model.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')
    return model


def enhanced_context_aware_model(params):
    input_1 = Input(shape=(15360))
    branch_1 = BatchNormalization()(input_1)
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

    model = tf.keras.Model(inputs=[input_1, input_2], outputs=x)
    model.compile(
        Adam(lr=params.learning_rate), loss='mean_squared_error')
    return model


def get_model(args, params, paths):
    if args.model == 'cnn' or args.model == 'srgan_cnn':
        model = cnn_model(params)
    else:
        cnn_dir = Path(paths['cnn_model'], 'model')
        if args.model == 'srgan_context_aware':
            cnn_dir = Path(paths['srgan_cnn_model'], 'model') 
        assert cnn_dir.exists(), 'cnn model must be trained before training context aware model'
        if args.model == 'context_aware' or args.model == 'srgan_context_aware':
            model = context_aware_model(params)
        elif args.model == 'unfreezed_context_aware':
            model = unfreezed_context_aware_model(params)
        elif args.model == 'enhanced_context_aware':
            model = enhanced_context_aware_model(params)

    return model


def load_saved_model(args, paths):

    model_dir_name = f'{args.model}_model'
    model_dir = Path(args.experiment_dir, model_dir_name)

    saved_model_dir = Path(model_dir, 'model')
    assert saved_model_dir.exists, f'{args.model} model is not saved at: {saved_model_dir}'
    logging.info(f'loading {args.model} model from {saved_model_dir}')
    model = load_model(saved_model_dir)
    if args.model in ['context_aware', 'enhanced_context_aware']:
        cnn_dir = Path(paths['cnn_model'], 'model')
        assert cnn_dir.exists(), 'cnn model must be trained before training context aware model'
        cnn = load_model(cnn_dir)
        cnn_input = cnn.layers[0].input
        cnn_l10_output = cnn.layers[10].output
        transform = create_transform_layer(paths)
        output_after_transform = tf.keras.layers.Lambda(
            transform)(cnn_l10_output)
        model2_input = tf.keras.layers.Input((132))
        second_stage_output = model([output_after_transform, model2_input])
        model = tf.keras.Model(
            inputs=[cnn_input, model2_input], outputs=second_stage_output)
    return model
