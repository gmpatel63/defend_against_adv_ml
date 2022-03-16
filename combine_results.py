import pandas as pd
from pathlib import Path

from model import config


def main():
    experiments_dir = Path('./experiments')
    results_data = []
    for experiment in experiments_dir.iterdir():
        if(experiment.name.startswith('.') or experiment.is_file()):
            continue

        for model_name in config.MODEL_NAMES:
            # [ experiment_name, model_name, rmse, gsm, l0]
            model_dir = Path(experiment, f'{model_name}_model')
            results_csv = Path(model_dir, 'results.csv')
            if(not model_dir.exists() or not results_csv.exists):
                continue
            results_df = pd.read_csv(results_csv)

            model_results = {}
            model_results['experiment'] = experiment.name
            model_results['model'] = model_name

            for index, row in results_df.iterrows():
                print('----------- new row ------------')
                print(row)
                model_results[row['metric']] = row['result/deviation']

            results_data.append(model_results)

    all_results = pd.DataFrame(results_data)
    all_results.to_csv(Path(experiments_dir, 'comparison.csv'))


if __name__ == '__main__':
    main()
