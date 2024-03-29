import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import namedtuple
from imblearn.over_sampling import RandomOverSampler

from . import config
from .utils import get_paths
from .models import create_transform_layer

ROUNDED_AGE_CLMN = 'rounded_age'


def plot_dataset(df, output_dir, plot_title):
    'create a bar plot of age distribution of all subject ids'

    sorted_df = df.sort_values(ROUNDED_AGE_CLMN)
    count = sorted_df[ROUNDED_AGE_CLMN].value_counts()
    plt.rc('font', size=10)
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation=45)
    plot = sns.barplot(x=count.index, y=count.values)
    # training_data_plot.set(xticklabels=[])
    # training_data_plot.set(xlabel=None)
    plot.set(xlabel='rounded age', ylabel='sample count')
    plot.set(title=plot_title)
    plot_file_path = Path(output_dir, plot_title + '.png')
    plt.savefig(plot_file_path)
    logging.info(f'{plot_title} plot saved at: {plot_file_path}')
    plt.clf()


def create_datasets(input_df, params):
    '''
    Divide input data into training, validation and
    testing datasets based on minimun and maximum age
    '''
    input_df[ROUNDED_AGE_CLMN] = input_df['age'].round()
    age_list = input_df[ROUNDED_AGE_CLMN].unique()
    age_list = [age for age in age_list if age >=
                params.age_min and age <= params.age_max]
    column_list = input_df.columns.values.tolist()
    train_df = pd.DataFrame(columns=column_list)
    valid_df = pd.DataFrame(columns=column_list)
    test_df = pd.DataFrame(columns=column_list)

    for age in age_list:
        age_df = input_df[input_df[ROUNDED_AGE_CLMN] == age]
        df_len = len(age_df)

        if(df_len > params.age_max_sample_cnt):
            age_df = age_df.head(params.age_max_sample_cnt)
            df_len = len(age_df)

        # divide age group into training, validation and testing
        TRAIN_CUT = int(config.TRAIN_PCT * df_len)
        VALID_CUT = int(config.VALID_PCT * df_len)

        age_train_df = age_df[0:TRAIN_CUT]
        age_valid_df = age_df[TRAIN_CUT:(TRAIN_CUT+VALID_CUT)]
        age_test_df = age_df[(TRAIN_CUT+VALID_CUT):]

        train_df = train_df.append(age_train_df, ignore_index=True)
        valid_df = valid_df.append(age_valid_df, ignore_index=True)
        test_df = test_df.append(age_test_df, ignore_index=True)

    return train_df, valid_df, test_df


def preprocess_anatomical_features(train_df, valid_df, test_df):
    training_anat_features = train_df[config.ANATOMICAL_COLUMNS].to_numpy()
    mm_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_train_feats = mm_scaler.fit_transform(training_anat_features)

    std_scaler = StandardScaler(with_mean=True, with_std=False)
    scaled_train_feats = std_scaler.fit_transform(scaled_train_feats)

    train_anat_df = pd.DataFrame(
        scaled_train_feats, columns=config.ANATOMICAL_COLUMNS)

    other_train_feats = train_df.drop(config.ANATOMICAL_COLUMNS, axis=1)
    scaled_train_df = pd.concat([other_train_feats, train_anat_df], axis=1)

    validation_anat_features = valid_df[config.ANATOMICAL_COLUMNS].to_numpy()
    test_anat_features = test_df[config.ANATOMICAL_COLUMNS].to_numpy()

    scaled_valid_feats = mm_scaler.transform(validation_anat_features)
    scaled_test_feats = mm_scaler.transform(test_anat_features)

    scaled_valid_feats = std_scaler.transform(scaled_valid_feats)
    scaled_test_feats = std_scaler.transform(scaled_test_feats)

    valid_anat_df = pd.DataFrame(
        scaled_valid_feats, columns=config.ANATOMICAL_COLUMNS)
    test_anat_df = pd.DataFrame(
        scaled_test_feats, columns=config.ANATOMICAL_COLUMNS)

    other_valid_feats = valid_df.drop(config.ANATOMICAL_COLUMNS, axis=1)
    other_test_feats = test_df.drop(config.ANATOMICAL_COLUMNS, axis=1)

    scaled_valid_df = pd.concat([other_valid_feats, valid_anat_df], axis=1)
    scaled_test_df = pd.concat([other_test_feats, test_anat_df], axis=1)

    return scaled_train_df, scaled_valid_df, scaled_test_df


def create_and_plot_data_csvs(args, params, paths):
    '''
    - Read all input data from data_dir
    - create datasets: training, validation and testing
    - plot datasets
    - save csv files: training_data.csv, validation_data.csv, test_data.csv
    '''
    logging.info(f'--------------------------------------------------')
    logging.info(f'creating and plotting datasets')
    data_dir = paths['data_dir']
    csv_output_dir = paths['csv_dir']
    plots_dir = paths['plots_dir']
    # use existing datasets if already exists
    if csv_output_dir.exists():
        logging.info(f'Output dir already exists: {csv_output_dir}')
        logging.info(
            f'Not creating new datasets, existing datasets will be used')
        logging.info(f'--------------------------------------------------')
        return

    csv_output_dir.mkdir()
    plots_dir.mkdir()
    # ignore hidden files and get input_dataframe path from all datasets folder
    input_dataframes_path = [Path(path, config.INPUT_FNAME)
                             for path in data_dir.iterdir() if path.is_dir()]

    # read all dataframes and concatenate
    input_df = pd.concat([pd.read_csv(df)
                          for df in input_dataframes_path], ignore_index=True)

    input_csv_path = Path(csv_output_dir, 'all_input_data.csv')
    input_df.to_csv(input_csv_path, index=False)

    # shuffle input_dataframe
    input_df = input_df.sample(frac=1).reset_index(drop=True)
    input_df_len = len(input_df)
    logging.info(
        f'Found total {input_df_len} examples/records from all datasets')

    train_df, valid_df, test_df = create_datasets(input_df, params)

    train_df, valid_df, test_df = preprocess_anatomical_features(
        train_df, valid_df, test_df)

    # oversample training dataset
    if params.oversample == True:
        logging.info('oversampling training dataset using random over-sampler')
        oversample = RandomOverSampler(sampling_strategy='not majority')
        train_df, train_labels = oversample.fit_resample(
            train_df, train_df['rounded_age'])
        train_df = train_df.sample(frac=1).reset_index(drop=True)

    plot_dataset(train_df, plots_dir, 'Training Data Distribution')
    plot_dataset(valid_df, plots_dir,
                 'Validation Data Distribution')
    plot_dataset(test_df, plots_dir, 'Test Data Distribution')
    logging.info(f'Created data distribution plots at : {plots_dir}')

    logging.info(f'training dataset length: {len(train_df)}')
    logging.info(f'validation dataset length: {len(valid_df)}')
    logging.info(f'testing dataset length: {len(test_df)}\n')

    train_df.to_csv(paths['training_data'], index=False)
    valid_df.to_csv(paths['validation_data'], index=False)
    test_df.to_csv(paths['test_data'], index=False)
    logging.info(f'Created datasets at : {csv_output_dir}')
    logging.info(f'--------------------------------------------------')


def load_normalized_mri(image_path):
    'read nifti image using nibabel and convert it into numpu array'
    mri = nib.load(image_path)
    mri_array = mri.get_fdata()
    if mri_array.shape != config.MRI_SHAPE:
        mri_array = np.expand_dims(mri_array, axis=3)
    return mri_array


def create_generator(dataset_df, args):

    def generator():
        for index, row in dataset_df.iterrows():
            image_path = row['mri_path']
            image_data = load_normalized_mri(image_path)
            label = row['age']
            features = image_data
            if args.with_anat_features:
                anat_features = []

                for anat_column in config.ANATOMICAL_COLUMNS:
                    anat_features.append(row[anat_column])

                anat_features = np.array(anat_features)
                features = (image_data, anat_features)

            yield features, [label]

    return generator


def create_context_aware_input(df, args, paths):
    logging.info('creating dataset for context aware model')
    paths = get_paths(args)
    cnn_dir = Path(paths['cnn_model'], 'model')
    if args.model == 'srgan_context_aware':
        cnn_dir = Path(paths['srgan_cnn_model'], 'model')
    assert cnn_dir.exists(), 'cnn model must be trained before training context aware model'
    cnn = load_model(cnn_dir)
    cnn_input = cnn.layers[0].input
    cnn_l10_output = cnn.layers[10].output
    transform = create_transform_layer(paths)
    output_after_transform = tf.keras.layers.Lambda(
        transform)(cnn_l10_output)
    transform_model = tf.keras.Model(
        inputs=cnn_input, outputs=output_after_transform)

    Record = namedtuple('Record', 'conv_features anat_features age')
    context_aware_input = []
    conv_features = []
    for index, row in df.iterrows():
        mri_path = row['mri_path']
        image_data = load_normalized_mri(mri_path)
        image_data = np.expand_dims(image_data, axis=0)
        prediction = transform_model(image_data)
        anat_features = []
        for anat_column in config.ANATOMICAL_COLUMNS:
            anat_features.append(row[anat_column])
        anat_features = np.array(anat_features)
        conv_features.append(prediction.numpy()[0])
        new_record = Record(
            conv_features=prediction[0], anat_features=anat_features, age=row['age'])
        context_aware_input.append(new_record)

    df = pd.DataFrame(conv_features)
    if paths is not None:
        df.to_csv(Path(paths['csv_dir'], 'ca_branch1_input.csv'), index=False)

    return context_aware_input


def create_generator_for_context_aware_model(input_data, args):

    def generator():
        for record in input_data:
            features = (record.conv_features, record.anat_features)
            label = [record.age]
            yield features, label

    return generator


def get_dataframe(df_path, args, adversarial_input=False, eps=None):

    df = pd.read_csv(df_path)
    if args.model in ['srgan_cnn', 'srgan_context_aware'] and (args.train or args.evaluate or adversarial_input or args.attack=='red_attack'):
        experiment_name = Path(args.experiment_dir).name
        if args.evaluate or args.train or args.attack=='red_attack':
            new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/legitimate_input'
        if adversarial_input:
            assert eps is not None
            new_path_str = f'{config.SRGAN_OUTPUT_DATA}/{experiment_name}/evaluate/adversarial_input/{args.model}/{args.attack}_{eps}'
        logging.info(
            f'changing input path, replace {config.DATA_DIR} with {new_path_str}')
        df['mri_path'] = df['mri_path'].str.replace(
            config.DATA_DIR, new_path_str)

    return df


def create_tf_dataset(df_path, args, params, paths=None, training=False):
    '''
    create tf.data.DataSet object using from_generator
    '''
    logging.info(f'creating tf.data.Dataset from: {df_path.name}')
    df = get_dataframe(df_path, args)

    generator = create_generator(df, args)
    if args.model in ['context_aware', 'enhanced_context_aware'] and args.train:
        input_data = create_context_aware_input(df, args, paths)
        generator = create_generator_for_context_aware_model(input_data, args)
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    mri_spec = tf.TensorSpec(shape=config.MRI_SHAPE, dtype=tf.float32)
    cnn_transform_spec = tf.TensorSpec(shape=(15360), dtype=tf.float32)
    anat_features_spec = tf.TensorSpec(
        shape=config.ANAT_FEAT_SHAPE, dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=config.OUTPUT_SHAPE, dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=(mri_spec, label_spec))
    if args.with_anat_features:
        if args.model in ['context_aware', 'enhanced_context_aware'] and args.train:
            dataset = tf.data.Dataset.from_generator(
                generator, output_signature=((cnn_transform_spec, anat_features_spec), label_spec))
        else:
            dataset = tf.data.Dataset.from_generator(
                generator, output_signature=((mri_spec, anat_features_spec), label_spec))

    if training:
        dataset = dataset.shuffle(config.TF_DATASET_BUFFER)

    dataset = dataset.batch(params.batch_size).prefetch(
        2).with_options(options)

    return dataset
