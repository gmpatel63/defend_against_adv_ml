"""General utility functions"""

from datetime import datetime
import argparse
import json
import logging
from pathlib import Path

from . import config


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', default='experiments/base_model',
                        help='Experiment directory containing params.json')

    # parser.add_argument(
    #     '--data_dir', help='Directory containing all the dataset folders and dataframes')

    # parser.add_argument(
    #     '--restore_from', help='Optional, directory or file containing saved models')

    parser.add_argument('--work_dir', required=True, type=str,
                        help='main work dir containing \'model_input_data\' folder')

    # parser.add_argument('--with_anat_feats', dest='with_anat_feats', action='store_true',
    #                     help='use context aware model if true')

    parser.add_argument('--model', required=True,
                        help='cnn or context_aware')

    parser.add_argument('--attack', help='gsm or l0')

    args = parser.parse_args()
    assert Path(args.work_dir).is_dir(), f'invalid path for work_dir'

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
    model_name = args.model
    exec_string = 'training'
    if args.attack is not None:
        exec_string = 'attack' + args.attack
    log_file_name = f'{dt_string}_{model_name}_{exec_string}.log'
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
        'context_aware_model': Path(args.experiment_dir, config.CONTEXT_AWARE_MODEL_DIR)
    }
    return paths
