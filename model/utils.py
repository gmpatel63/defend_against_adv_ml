"""General utility functions"""

import argparse
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

from . import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_dir', default='experiments/base_model',
                        help='Experiment directory containing params.json')

    parser.add_argument('--work_dir', required=True, type=str,
                        help='main work dir containing \'model_input_data\' folder')

    parser.add_argument('--model', required=True, choices=config.MODEL_NAMES)

    parser.add_argument('--op', choices=['train', 'evaluate'])
    parser.add_argument('--attack', help='gsm or l0')

    # parser.add_argument(
    #     '--restore_from', help='Optional, directory or file containing saved models')

    args = parser.parse_args()
    assert Path(args.work_dir).is_dir(), f'invalid path for work_dir'
    
    args.train = False
    args.evaluate = False
        
    if args.op == 'train':
        args.train = True
        args.evaluate = False
    elif args.op == 'evaluate':
        args.train = False
        args.evaluate = True

    if args.model == 'cnn':
        args.with_anat_features = False
    else:
        args.with_anat_features = True

    return args


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(args):
    """Sets the logger to log info in terminal and file `log_path`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logs_dir = Path(args.experiment_dir, 'logs')
    logs_dir.mkdir(exist_ok=True)

    now = datetime.now()
    dt_string = now.strftime('%Y_%m_%d_%H_%M')
    exec_string = 'train' if args.train else 'evaluate'
    if args.attack is not None:
        exec_string = 'attack' + args.attack
    log_file_name = f'{dt_string}_{exec_string}_{args.model}.log'
    log_path = Path(logs_dir, log_file_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)


def get_paths(args):
    data_dir = Path(args.work_dir, config.DATA_DIR)
    csv_dir = Path(args.experiment_dir, config.CSV_DIR)
    paths = {
        'data_dir': data_dir,
        'csv_dir': csv_dir,
        'training_data': Path(csv_dir, 'training_data.csv'),
        'validation_data': Path(csv_dir, 'validation_data.csv'),
        'test_data': Path(csv_dir, 'test_data.csv'),
        'plots_dir': Path(args.experiment_dir, 'plots'),
        'cnn_model': Path(args.experiment_dir, config.CNN_DIR),
        'context_aware_model': Path(args.experiment_dir, config.CONTEXT_AWARE_MODEL_DIR),
        'enhanced_context_aware_model': Path(args.experiment_dir, config.ENHANCED_CONTEXT_AWARE_MODEL_DIR),
        'unfreezed_context_aware_model': Path(args.experiment_dir, config.UNFREEZED_CONTEXT_AWARE_MODEL_DIR)
    }
    return paths


# def change_work_dir_in_datasets(csv_dir_path, old_work_dir_path, new_work_dir_path):
#     '''
#     change path of 'mri_path' in dataset csvs
#
#     Example:
#     change '/path/to/work_dir/model_input_data/ABIDE_I/Leuven1_0050691.nii.gz'
#     to '/new_path/new_work_dir/model_input_data/ABIDE_I/Leuven1_0050691.nii.gz'
#     '''
#     csv_dir = Path(csv_dir_path)
#     assert csv_dir.exists(), f'invalid path: {csv_dir}'
#
#     for path in csv_dir.iterdir():
#         if path.suffix != '.csv' or path.is_dir():
#             continue
#
#         df = pd.read_csv(path)
#         df['mri_path'] = df['mri_path'].str.replace(old_work_dir_path, new_work_dir_path)
#         df.to_csv(path, index=False)
