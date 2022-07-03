import os
from pathlib import Path

from model.utils import Params, parse_args, get_paths, set_logger
from model.datasets import create_and_plot_data_csvs
from model.training import train_model, evaluate_model
from model.attacks import create_adv_inputs_gsm, gsm_attack, l0_attack
from model.models import load_saved_model

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

    model = load_saved_model(args, paths)
    if args.model in ['srgan_cnn', 'srgan_context_aware']:
        if args.create_adv_input is True:
            if args.attack == 'gsm':
                create_adv_inputs_gsm(args, params, paths, model)
            elif args.attack == 'l0':
                # create adv inputs l0
                pass
        elif args.evaluate_adv_input is True:
            if args.attack == 'gsm':
                # evaluate_adv_inputs_gsm(args, params, paths, model)
                pass
            elif args.attack == 'l0':
                # evaluate adv inputs l0
                pass   
    else:
        if args.attack == 'gsm':
            gsm_attack(args, params, paths, model)
        elif args.attack == 'l0':
            l0_attack(args, params, paths, model)


if __name__ == '__main__':
    main()
