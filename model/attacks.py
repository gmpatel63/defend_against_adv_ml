import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import subprocess
import nibabel as nib

from . import config
from .datasets import load_normalized_mri, get_dataframe
from .plots import plot_categorical_deviation, plot_comparison

COLUMN_NAMES = ['dataset', 'site_name', 'subject_id', 'age']
# keep EPS_VALUES constant during an experiment
EPS_VALUES = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]

def get_anatomical_features(subj_record):
    anat_features = []
    for anat_column in config.ANATOMICAL_COLUMNS:
        anat_features.append(subj_record[anat_column])
    anat_features = np.array(anat_features)
    anat_features = np.expand_dims(anat_features, axis=0)
    anat_features = tf.convert_to_tensor(anat_features, dtype=tf.float32)
    return anat_features


def gsm_attack(args, params, paths, model):
    logging.info(f'Starting gsm attack for {args.model}')
    
    test_df = get_dataframe(paths['test_data'], args)
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
    test_df = get_dataframe(paths['test_data'], args)
    
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


def create_adv_inputs_gsm(args, params, paths, model):
    logging.info(f'Creating adversarial inputs for {args.model} using {args.attack} attack')

    test_df = get_dataframe(paths['test_data'], args)
    
    EPS_COLUMNS = [
        f'{args.attack}_{str(eps_value)}' for eps_value in EPS_VALUES]
    # column_names = COLUMN_NAMES + EPS_COLUMNS
    
    results = []
    experiment_dir = Path(args.experiment_dir)
    experiment_name = experiment_dir.name

    for index, sample in test_df.iterrows():

        # result = []
        # for column_name in COLUMN_NAMES:
        #     result.append(sample[column_name])

        mri = load_normalized_mri(sample.mri_path)
        
        # load srgan gradient
        srgan_grad_fname = sample.mri_path.replace('.nii.gz', '_grad.nii.gz')
        new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/legitimate_input'
        srgan_grad_fname = srgan_grad_fname.replace(config.DATA_DIR, new_path_str)
        srgan_grad = load_normalized_mri(srgan_grad_fname)
        srgan_grad = np.expand_dims(mri, axis=0)
        srgan_grad = tf.convert_to_tensor(srgan_grad, dtype=tf.float32)

        mri_filename = Path(sample.mri_path).name.partition('.')[0]
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

        grad = tape.gradient(pred, input_tensor)
        if args.with_anat_features:
            grad = grad[0][0]

        for eps_rate, eps_column_name in zip(EPS_VALUES, EPS_COLUMNS):
            eps = eps_rate * (max_value - min_value)
            perturbation = np.sign(tf.math.multiply(grad, srgan_grad)) * eps
            # perturbation = np.sign(grad) * eps 
            adv_mri = mri_tensor + perturbation
            adv_mri = tf.where(adv_mri > max_value, max_value, adv_mri)
            adv_mri = tf.where(adv_mri < min_value, min_value, adv_mri)
            
            
            subject_name = f'{sample.dataset}_{mri_filename}'
            adv_mri_dir = Path(args.work_dir, config.SRGAN_INPUT_DATA,
                               'adversarial_input', experiment_name, args.model, eps_column_name, subject_name)
            adv_mri_dir.mkdir(parents=True, exist_ok=True)
            mri_file = Path(
                adv_mri_dir, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
            
            adv_mri = adv_mri.numpy()
            adv_mri = np.squeeze(adv_mri)
            adv_image = nib.Nifti1Image(adv_mri, affine=np.eye(4))
            adv_image.to_filename(mri_file)
            mri_mask_path = Path(
                adv_mri_dir, 'T1_brain_extractedBrainExtractionMask.nii.gz')
            subprocess.run(['fslmaths', mri_file, '-bin', mri_mask_path])
            # result.append(subject_name)
            logging.info(f'created adversarial mri at {adv_mri_dir}')

        # results.append(result)

    # results_df = pd.DataFrame(results, columns=column_names)
    # model_dir = Path(args.experiment_dir, paths['csv_dir'])
    # df_path = Path(model_dir, f'{args.attack}_paths.csv')

    
def evaluate_adv_inputs_gsm(args, params, paths, model):
    logging.info(f'Evaluating adversarial inputs for {args.model} using {args.attack} attack')
    
    test_df = pd.read_csv(paths['test_data'])
    experiment_name = Path(args.experiment_dir).name
    new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/legitimate_input'
    test_df['mri_path'] = test_df['mri_path'].str.replace(config.DATA_DIR, new_path_str)
    results_df = test_df[COLUMN_NAMES].copy()
    
    # predict on legitimate input
    predictions = []
    for index, sample in test_df.iterrows():
        mri = load_normalized_mri(sample.mri_path)
        mri = np.expand_dims(mri, axis=0)
        if args.with_anat_features:
            anat_features = get_anatomical_features(sample)
        mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
        if args.with_anat_features:
            input_tensor = [(mri_tensor, anat_features)]
        else:
            input_tensor = mri_tensor
            
        prediction = model.predict(input_tensor)[0][0]
        predictions.append(prediction)
    results_df['predictions'] = predictions
     
    for eps_value in EPS_VALUES:
        df = get_dataframe(paths['test_data'], args, adversarial_input=True, eps=eps_value)
        current_predictions = []
        for index, sample in df.iterrows():
            mri = load_normalized_mri(sample.mri_path)
            mri = np.expand_dims(mri, axis=0)
            if args.with_anat_features:
                anat_features = get_anatomical_features(sample)
            mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
            if args.with_anat_features:
                input_tensor = [(mri_tensor, anat_features)]
            else:
                input_tensor = mri_tensor
            
            prediction = model.predict(input_tensor)[0][0]
            current_predictions.append(prediction)
        
        results_df[eps_value] = current_predictions 
        
    # results_df = pd.DataFrame(results, columns=column_names)
    model_dir = Path(args.experiment_dir, f'{args.model}_model')
    df_path = Path(model_dir, f'{args.attack}.csv')
    results_df.to_csv(df_path, index=False)
    plot_categorical_deviation(
        args, results_df, EPS_VALUES, paths['plots_dir'])
    plot_comparison(args, EPS_VALUES, paths['plots_dir'])

def create_adv_inputs_l0(args, params, paths, model, direction='max'):
    
    logging.info(f'Starting l0 attack')
    test_df = get_dataframe(paths['test_data'], args)
    
    INTERVAL = 10  # 100
    NUMBERS_INTERVAL = 5  # 40
    
    experiment_dir = Path(args.experiment_dir)
    experiment_name = experiment_dir.name

    attack_columns = [
        f'{i*NUMBERS_INTERVAL}' for i in range(1, NUMBERS_INTERVAL + 1)]
    
    # column_names = COLUMN_NAMES + ['predictions'] + attack_columns
    # logging.info(column_names)
    # results = [] 

    for index, sample in test_df.iterrows():
        
        result = []
        for column_name in COLUMN_NAMES:
            result.append(sample[column_name])

        mri = load_normalized_mri(sample.mri_path)
        
        # load srgan gradient
        srgan_grad_fname = sample.mri_path.replace('.nii.gz', '_grad.nii.gz')
        new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/legitimate_input'
        srgan_grad_fname = srgan_grad_fname.replace(config.DATA_DIR, new_path_str)
        srgan_grad = load_normalized_mri(srgan_grad_fname)
        srgan_grad = np.expand_dims(mri, axis=0)
        srgan_grad = tf.convert_to_tensor(srgan_grad, dtype=tf.float32)
        
        mri_filename = Path(sample.mri_path).name.partition('.')[0]
        
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
        grad = srgan_grad * grad
        abs_grad = np.abs(grad)
        best_prediction = pred
        predictions = []
        cnt = 0
        i = 0
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
            
            subject_name = f'{sample.dataset}_{mri_filename}'
            print(f'cnt: {cnt}, flag_better: {flag_better}')
            
            if cnt % INTERVAL == 0 and flag_better:

                attack_dir_name = f'{args.attack}_{attack_columns[i]}'
                i += 1
                adv_mri_dir = Path(args.work_dir, config.SRGAN_INPUT_DATA,
                                'adversarial_input', experiment_name, args.model, attack_dir_name, subject_name)
                adv_mri_dir.mkdir(parents=True, exist_ok=True)
                mri_file = Path(
                    adv_mri_dir, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
                # adv_mri = adv_mri.numpy()
                adv_mri_save = np.squeeze(adv_mri)
                adv_image = nib.Nifti1Image(adv_mri_save, affine=np.eye(4))
                adv_image.to_filename(mri_file)
                mri_mask_path = Path(
                    adv_mri_dir, 'T1_brain_extractedBrainExtractionMask.nii.gz')
                subprocess.run(['fslmaths', mri_file, '-bin', mri_mask_path])
                
            if cnt_failure >= 20:
                
                while(i < len(attack_columns)):
                    attack_dir_name = f'{args.attack}_{attack_columns[i]}'
                    i += 1
                    adv_mri_dir = Path(args.work_dir, config.SRGAN_INPUT_DATA,
                                    'adversarial_input', experiment_name, args.model, attack_dir_name, subject_name)
                    adv_mri_dir.mkdir(parents=True, exist_ok=True)
                    mri_file = Path(
                        adv_mri_dir, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
                    # adv_mri = adv_mri.numpy()
                    adv_mri_save = np.squeeze(adv_mri)
                    adv_image = nib.Nifti1Image(adv_mri_save, affine=np.eye(4))
                    adv_image.to_filename(mri_file)
                    mri_mask_path = Path(
                        adv_mri_dir, 'T1_brain_extractedBrainExtractionMask.nii.gz')
                    subprocess.run(['fslmaths', mri_file, '-bin', mri_mask_path])
                
                break
            if cnt // INTERVAL >= NUMBERS_INTERVAL:
                # save one example before exit
                # if self.args.save_data and i >= self.args.start + self.args.instances - 1:
                #     y_pred_adv = self.model.predict_image(x_adv)
                #     self.save_variables([x, x_adv, predictions[0], y_pred_adv[0][0]])
                break 
    


def evaluate_adv_inputs_l0(args, params, paths, model):
    logging.info(f'Evaluating adversarial inputs for {args.model} using {args.attack} attack')
    test_df = pd.read_csv(paths['test_data'])
    experiment_name = Path(args.experiment_dir).name
    new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/legitimate_input'
    test_df['mri_path'] = test_df['mri_path'].str.replace(config.DATA_DIR, new_path_str)
    results_df = test_df[COLUMN_NAMES].copy()
    
    # predict on legitimate input
    predictions = []
    for index, sample in test_df.iterrows():
        mri = load_normalized_mri(sample.mri_path)
        mri = np.expand_dims(mri, axis=0)
        if args.with_anat_features:
            anat_features = get_anatomical_features(sample)
        mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
        if args.with_anat_features:
            input_tensor = [(mri_tensor, anat_features)]
        else:
            input_tensor = mri_tensor
            
        prediction = model.predict(input_tensor)[0][0]
        predictions.append(prediction)
    results_df['predictions'] = predictions
    
    INTERVAL = 10  # 100
    NUMBERS_INTERVAL = 5  # 40 
    attack_columns = [
        f'{i*NUMBERS_INTERVAL}' for i in range(1, NUMBERS_INTERVAL + 1)]
    for eps_value in attack_columns:
        df = get_dataframe(paths['test_data'], args, adversarial_input=True, eps=eps_value)
        current_predictions = []
        for index, sample in df.iterrows():
            mri = load_normalized_mri(sample.mri_path)
            mri = np.expand_dims(mri, axis=0)
            if args.with_anat_features:
                anat_features = get_anatomical_features(sample)
            mri_tensor = tf.convert_to_tensor(mri, dtype=tf.float32)
            if args.with_anat_features:
                input_tensor = [(mri_tensor, anat_features)]
            else:
                input_tensor = mri_tensor
            
            prediction = model.predict(input_tensor)[0][0]
            current_predictions.append(prediction)
        
        results_df[eps_value] = current_predictions 
        
    # results_df = pd.DataFrame(results, columns=column_names)
    model_dir = Path(args.experiment_dir, f'{args.model}_model')
    df_path = Path(model_dir, f'{args.attack}.csv')
    results_df.to_csv(df_path, index=False)
    plot_categorical_deviation(
        args, results_df, attack_columns, paths['plots_dir'])
    plot_comparison(args, attack_columns, paths['plots_dir'])