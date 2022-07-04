import argparse
from pathlib import Path
import shutil
import subprocess
import pandas as pd

from model import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_dir', default='experiments/base_model',
                        help='Experiment directory containing params.json')

    parser.add_argument('--work_dir', required=True, type=str,
                        help='main work dir containing \'model_input_data\' folder')

    args = parser.parse_args()
    return args


def create_srgan_dataset(df, srgan_data_dir, df_path):

    srgan_subject_names = []

    for index, row in df.iterrows():
        mri_path = row['mri_path']
        mri_filename = Path(mri_path).name.partition('.')[0]
        mri_dir_name = f'{row["dataset"]}_{mri_filename}'
        new_mri_dir = Path(srgan_data_dir, mri_dir_name)
        new_mri_dir.mkdir(exist_ok=True)
        new_mri_path = Path(
            new_mri_dir, 'T1_brain_extractedBrainExtractionBrain.nii.gz')
        mri_mask_path = Path(
            new_mri_dir, 'T1_brain_extractedBrainExtractionMask.nii.gz')

        # training and validation dataframes will have repeated samples because of oversampling
        # if new mri path already exists, skip
        if new_mri_path.exists() and mri_mask_path.exists():
            srgan_subject_names.append(mri_dir_name)
            continue
        shutil.copy2(mri_path, new_mri_path)
        subprocess.run(['fslmaths', new_mri_path, '-bin', mri_mask_path])
        srgan_subject_names.append(mri_dir_name)

    df['srgan_subject_names'] = srgan_subject_names
    df.to_csv(df_path, index=False)


def main():
    
    args = parse_args()
    data_dir = Path(args.work_dir, config.DATA_DIR)

    csv_dir = Path(args.experiment_dir, config.CSV_DIR)
    training_data_path = Path(csv_dir, 'training_data.csv')
    validation_data_path = Path(csv_dir, 'validation_data.csv')
    test_data_path = Path(csv_dir, 'test_data.csv')

    training_df = pd.read_csv(training_data_path)
    validation_df = pd.read_csv(validation_data_path)
    test_df = pd.read_csv(test_data_path)

    # srgan data dirs
    srgan_data_dir = Path(args.work_dir, config.SRGAN_INPUT_DATA, 'legitimate_input')
    srgan_data_dir.mkdir(parents=True, exist_ok=True)

    create_srgan_dataset(training_df, srgan_data_dir, training_data_path)
    create_srgan_dataset(validation_df, srgan_data_dir, validation_data_path)
    create_srgan_dataset(test_df, srgan_data_dir, test_data_path)


if __name__ == '__main__':
    main()
