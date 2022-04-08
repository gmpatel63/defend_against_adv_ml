import logging
from unittest import result
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from . import config
from .datasets import load_normalized_mri
from .plots import plot_categorical_deviation, plot_comparison

COLUMN_NAMES = ['dataset', 'site_name', 'subject_id', 'age']

def get_anatomical_features(subj_record):
    anat_features = []
    for anat_column in config.ANATOMICAL_COLUMNS:
        anat_features.append(subj_record[anat_column])
    anat_features = np.array(anat_features)
    anat_features = np.expand_dims(anat_features, axis=0)
    anat_features = tf.convert_to_tensor(anat_features, dtype=tf.float32)
    return anat_features


def gsm_attack(args, params, paths, model):
    logging.info(f'Starting gsm attack')
    
    test_df = pd.read_csv(paths['test_data'])
    EPS_VALUES = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    column_names = COLUMN_NAMES + ['predictions'] + EPS_VALUES
    logging.info(column_names)
    results = []
    for index, sample in test_df.iterrows():
        
        result = []
        for column_name in COLUMN_NAMES:
            result.append(sample[column_name])
        
        mri = load_normalized_mri(sample.mri_path)
        if args.with_anat_features:
            anat_features = get_anatomical_features(sample)
        max_value = mri.max()
        min_value = mri.min()
        
        # model only accepts array of examples so add mri into an array
        mri = np.expand_dims(mri, axis=0)
        mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
        if args.with_anat_features:
            input_tensor = [(mri_tensor, anat_features)]
        else:
            input_tensor = mri_tensor

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            pred = model(input_tensor)
            result.append(pred.numpy()[0][0])

        grad = tape.gradient(pred, input_tensor)
        if args.with_anat_features:
            grad = grad[0][0]

        for eps_rate in EPS_VALUES:
            eps = eps_rate * (max_value - min_value)
            perturbation = np.sign(grad) * eps
            # range check - verify it better.
            adv_mri = mri_tensor + perturbation
            adv_mri = tf.where(adv_mri > max_value, max_value, adv_mri)
            adv_mri = tf.where(adv_mri < min_value, min_value, adv_mri)
            if args.with_anat_features:
                adv_input_tensor = [(adv_mri, anat_features)]
            else:
                adv_input_tensor = adv_mri
            new_pred = model.predict(adv_input_tensor)
            result.append(new_pred[0][0])

        logging.info(result)
        results.append(result)

    results_df = pd.DataFrame(results, columns=column_names)
    model_dir = Path(args.experiment_dir, f'{args.model}_model')
    df_path = Path(model_dir, f'{args.attack}.csv')
    results_df.to_csv(df_path, index=False)

    plot_categorical_deviation(
        args, results_df, EPS_VALUES, paths['plots_dir'])
    plot_comparison(args, EPS_VALUES, paths['plots_dir'])


def l0_attack(args, params, paths, model, direction='max'):

    logging.info(f'Starting l0 attack')
    test_df = pd.read_csv(paths['test_data'])
    
    INTERVAL = 10  # 100
    NUMBERS_INTERVAL = 5  # 40

    attack_columns = [
        f'{i*NUMBERS_INTERVAL}' for i in range(1, NUMBERS_INTERVAL + 1)]
    
    column_names = COLUMN_NAMES + ['predictions'] + attack_columns
    logging.info(column_names)
    results = [] 
    for index, sample in test_df.iterrows():
        
        result = []
        for column_name in COLUMN_NAMES:
            result.append(sample[column_name])

        mri = load_normalized_mri(sample.mri_path)
        if args.with_anat_features:
            anat_features = get_anatomical_features(sample)
        max_value = mri.max()
        min_value = mri.min()
        mri = np.expand_dims(mri, axis=0)
        adv_mri = mri.copy()
        mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
        if args.with_anat_features:
            input_tensor = [(mri_tensor, anat_features)]
        else:
            input_tensor = mri_tensor

        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            pred = model(input_tensor)
            result.append(pred.numpy()[0][0])

        grad = tape.gradient(pred, input_tensor)
        if args.with_anat_features:
            grad = grad[0][0]
        abs_grad = np.abs(grad)
        best_prediction = pred
        predictions = []
        cnt = 0
        cnt_failure = 0

        while True:
            # it will flatten the array and then return index of nanargmax
            argmax_num = np.nanargmax(abs_grad)
            # get actual index from flattened index
            argmax_indices = np.unravel_index(argmax_num, abs_grad.shape)
            abs_grad[argmax_indices] = 0

            if direction == "max":
                best_value = max_value if grad[argmax_indices] > 0 else min_value
            else:
                best_value = min_value if grad[argmax_indices] > 0 else max_value
            saved_value = adv_mri[argmax_indices]
            flag_better = False
            for k in np.linspace(min_value, max_value, num=3):
                adv_mri[argmax_indices] = k
                adv_input_tensor = adv_mri
                if args.with_anat_features:
                    adv_input_tensor = [(adv_mri, anat_features)]
                new_pred = model(adv_input_tensor)
                if(new_pred[0][0] > best_prediction[0][0] and direction == "max") or (
                        new_pred[0][0] < best_prediction[0][0] and direction == "min"):
                    best_value = k
                    best_prediction = new_pred
                    flag_better = True
            adv_mri[argmax_indices] = best_value if flag_better else saved_value
            cnt = cnt + 1 if flag_better else cnt
            cnt_failure = cnt_failure + 1 if not flag_better else 0
            if cnt % INTERVAL == 0 and flag_better:
                predictions.append(float(best_prediction[0][0]))
            if cnt_failure >= 20:
                predictions = predictions + \
                    [float(best_prediction[0][0])] * \
                    (NUMBERS_INTERVAL-len(predictions))
                break
            if cnt // INTERVAL >= NUMBERS_INTERVAL:
                # save one example before exit
                # if self.args.save_data and i >= self.args.start + self.args.instances - 1:
                #     y_pred_adv = self.model.predict_image(x_adv)
                #     self.save_variables([x, x_adv, predictions[0], y_pred_adv[0][0]])
                break
        
        result = result + predictions
        logging.info(result)
        results.append(result)
        
    results_df = pd.DataFrame(results, columns=column_names)
    model_dir = Path(args.experiment_dir, f'{args.model}_model')
    df_path = Path(model_dir, f'{args.attack}.csv')
    results_df.to_csv(df_path, index=False)
    plot_categorical_deviation(
        args, results_df, attack_columns, paths['plots_dir'])
    plot_comparison(args, attack_columns, paths['plots_dir'])
