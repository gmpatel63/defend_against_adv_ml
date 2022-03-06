import logging
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import tensorflow as tf

from . import config

ROUNDED_AGE_CLMN = 'rounded_age'


def plot_dataset(df, output_dir, plot_title):
    'create a bar plot of age distribution of all subject ids'

    sorted_df = df.sort_values(ROUNDED_AGE_CLMN)
    count = sorted_df[ROUNDED_AGE_CLMN].value_counts()
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
    mri_array = np.expand_dims(mri_array, axis=3)
    return mri_array


def create_generator(dataset_df, with_anat_features=False):
    def generator():
        for index, row in dataset_df.iterrows():
            image_path = row['mri_path']
            image_data = load_normalized_mri(image_path)
            label = row['age']
            features = image_data
            if with_anat_features:
                anat_features = []
                for anat_column in config.ANATOMICAL_COLUMNS:
                    anat_features.append(row[anat_column])
                anat_features = np.array(anat_features)
                features = (image_data, anat_features)
            yield features, [label]
    return generator


def create_tf_dataset(df_path, args, params, training=False):
    '''
    create tf.data.DataSet object using from_generator
    '''
    logging.info(f'creating tf.data.Dataset from: {df_path.name}')
    df = pd.read_csv(df_path)
    generator = create_generator(df, args.with_anat_features)
    # Disable AutoShard.
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    mri_spec = tf.TensorSpec(shape=config.MRI_SHAPE, dtype=tf.float32)
    anat_features_spec = tf.TensorSpec(
        shape=config.ANAT_FEAT_SHAPE, dtype=tf.float32)
    label_spec = tf.TensorSpec(shape=config.OUTPUT_SHAPE, dtype=tf.float32)

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=(mri_spec, label_spec))
    if args.with_anat_features:
        dataset = tf.data.Dataset.from_generator(
            generator, output_signature=((mri_spec, anat_features_spec), label_spec))

    if training:
        dataset = dataset.shuffle(config.TF_DATASET_BUFFER)

    dataset = dataset.batch(params.batch_size).prefetch(
        2).with_options(options)

    return dataset
