import logging
import time
import pickle
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from .datasets import create_tf_dataset
from .models import load_saved_model, get_model
from .plots import plot_history

COLUMN_NAMES = ['dataset', 'site_name', 'subject_id', 'age', 'predictions']


def train_model(args, params, paths):

    training_dataset = create_tf_dataset(
        paths['training_data'], args, params, paths, training=True)
    validation_dataset = create_tf_dataset(
        paths['validation_data'], args, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(args, params, paths)
        model_dir_name = f'{args.model}_model'
        model_dir = Path(args.experiment_dir, model_dir_name)

        model_dir.mkdir(exist_ok=True)

        # create callback
        save_model_dir = Path(model_dir, 'model')
        callbacks = []
        if params.model_check_point == True:
            logging.info('adding ModelCheckpoint to callbacks')
            checkpoint = ModelCheckpoint(filepath=save_model_dir,
                                         monitor='val_loss',
                                         verbose=1,
                                         save_best_only=True,
                                         mode='min')
            callbacks.append(checkpoint)

        if params.variable_learning_rate == True:
            logging.info('adding ReduceLROnPlateau to callbacks')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                          patience=5, min_lr=params.min_learning_rate, verbose=1)
            callbacks.append(reduce_lr)

        logging.info(f'--------------------------------------------------')
        logging.info(f'starting training for {args.model} model')
        start = time.time()
        hist = model.fit(x=training_dataset, validation_data=validation_dataset,
                         epochs=params.num_epochs, verbose=2, callbacks=callbacks)
        end = time.time()-start
        logging.info(f'model training time: {end}')

        save_model_dir = Path(model_dir, 'model')
        save_model_dir.mkdir(exist_ok=True)
        model.save(save_model_dir)

    # print history
    logging.info(f'--------------------------------------------------')
    logging.info('training history:')
    loss_list = hist.history['loss']
    val_loss_list = hist.history['val_loss']
    for epoch, (loss, val_loss) in enumerate(zip(loss_list, val_loss_list), start=1):
        logging.info(f'epoch: {epoch}, loss: {loss}, val_loss: {val_loss}')

    logging.info(f'--------------------------------------------------')
    # save history
    history_file = Path(model_dir, 'training_history')
    with open(history_file, 'wb') as file_pi:
        pickle.dump(hist.history, file_pi)

    plot_history(hist.history, paths['plots_dir'])

def create_normer_and_scaler(args, paths, conv_features):
    min_max_scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_conv_feats = min_max_scaler.fit_transform(conv_features)
    
    mm_scaler_file = Path(paths[f'{args.model}_model'], 'min_max_scaler')
    with open(mm_scaler_file, 'wb') as file_pi:
        pickle.dump(min_max_scaler, file_pi)
    
    std_scaler = StandardScaler(with_mean=True, with_std=False)
    std_scaler.fit(scaled_conv_feats)
    
    std_scaler_file = Path(paths[f'{args.model}_model'], 'std_scaler') 
    with open(std_scaler_file, 'wb') as file_pi:
        pickle.dump(std_scaler, file_pi)
    
    
def evaluate_model(args, params, paths):
    '''
    make predictions on test dataset, save it in csv and calculate RMSE
    '''
    logging.info(f'--------------------------------------------------')
    logging.info(f'evaluating {args.model} model')
    test_dataset = create_tf_dataset(paths['test_data'], args, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = load_saved_model(args, paths)
        # evaluate model
        predictions = model.predict(x=test_dataset)
        
        # get output of flatten layer of cnn and find a normer and scaler for that output
        # that output is being used in context aware model
        if args.model in ['cnn', 'srgan_cnn']:
            model = load_saved_model(args, paths)
            cnn_input_layer = model.layers[0].input
            cnn_layer_10 = model.layers[10].output
            temp_model = tf.keras.Model(inputs=cnn_input_layer, outputs=cnn_layer_10)
            conv_features = temp_model.predict(x=test_dataset)
            create_normer_and_scaler(args, paths, conv_features)

    test_df = pd.read_csv(paths['test_data'])
    test_df['predictions'] = predictions
    rmse = ((test_df.predictions - test_df.age) ** 2).mean() ** .5

    logging.info(f'RMSE of the {args.model} model: {rmse}')
    model_dir = paths[f'{args.model}_model']
    predictions_csv_path = Path(model_dir, f'predictions.csv')
    test_df.to_csv(predictions_csv_path, index=False, columns=COLUMN_NAMES)

    results_csv_path = Path(model_dir, f'results.csv')
    if results_csv_path.exists():
        results_df = pd.read_csv(results_csv_path)
        results = {'metric':'rmse', 'result/deviation': rmse}
        results_df = results_df.append(results, ignore_index=True)
    else:
        results_df = pd.DataFrame([['rmse', rmse]], columns=['metric', 'result/deviation'],)
    results_df.to_csv(results_csv_path, index=False)

    logging.info(f'--------------------------------------------------')
    