import os
from pathlib import Path

from model.utils import Params, parse_args, get_paths, set_logger
from model.datasets import create_and_plot_data_csvs
from model.training import train_model, evaluate_model

os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'


def main():
    args = parse_args()
    # set logger
    set_logger(args)

    # get params
    json_path = Path(args.experiment_dir, 'params.json')
    assert json_path.is_file(), f'No config file found at {json_path}'
    params = Params(json_path)

    # get paths
    paths = get_paths(args)

    # Check that we are not overwriting some previous experiment
    # model_dir_has_best_weights = Path(args.model_dir, "best_weights")
    # overwritting = model_dir_has_best_weights and args.restore_from is None
    # assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    if args.train:
        create_and_plot_data_csvs(args, params, paths)
        train_model(args, params, paths)

    if args.evaluate:
        evaluate_model(args, params, paths)

    if args.attack == 'gsm':
        # gsm attack
        pass
    elif args.attack == 'l0':
        # l0 attack
        pass


if __name__ == '__main__':
    main()
