import logging
import time
import pickle
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from .datasets import create_tf_dataset
from .models import cnn_model, context_aware_model
from .plots import plot_history


def train_model(args, params, paths):

    training_dataset = create_tf_dataset(
        paths['training_data'], args, params, training=True)
    validation_dataset = create_tf_dataset(
        paths['validation_data'], args, params)
    test_dataset = create_tf_dataset(paths['test_data'], args, params)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        if args.model == 'cnn':
            model = cnn_model(params)
            model_dir = paths['cnn_model']
        elif args.model == 'context_aware':
            cnn_dir = Path(paths['cnn_model'], 'model')
            assert cnn_dir.exists(), 'cnn model must be trained before training context aware model'
            model = context_aware_model(params, cnn_dir)
            model_dir = paths['context_aware_model']

        model_dir.mkdir(exist_ok=True)

        # create callback
        save_model_dir = Path(model_dir, 'model')
        checkpoint = ModelCheckpoint(filepath=save_model_dir,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='min')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=5, min_lr=params.min_learning_rate, verbose=1)
        callbacks = [checkpoint, reduce_lr]

        logging.info(f'--------------------------------------------------')
        logging.info(f'starting training for {args.model} model')
        start = time.time()
        hist = model.fit(x=training_dataset, validation_data=validation_dataset,
                         epochs=params.num_epochs, verbose=2, callbacks=callbacks)
        end = time.time()-start
        logging.info(f'model training time: {end}')

        # evaluate model
        predictions = model.predict(x=test_dataset)

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

    test_df = pd.read_csv(paths['test_data'])
    test_df['predictions'] = predictions
    rmse = ((test_df.predictions - test_df.age) ** 2).mean() ** .5
    logging.info(f'--------------------------------------------------')
    print(f'RMSE of the {args.model} model: {rmse}')
    logging.info(f'--------------------------------------------------')

    predictions_csv_path = Path(model_dir, f'{args.model}_evaluation.csv')
    test_df.to_csv(predictions_csv_path, index=False, columns=[
                   'dataset', 'site_name', 'subject_id', 'age', 'predictions'])
