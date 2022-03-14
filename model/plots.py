import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# def plot_evaluation(pred_df, output_dir):
#     fig, ax1 = plt.subplots(figsize=(10, 10))
#     melted_df = pred_df.melt(id_vars='subject_id', value_vars=[
#                              'age', 'predictions'])
#     prediction_plt = sns.barplot(
#         x='subject_id', y='value', hue='variable', data=melted_df, ax=ax1)
#     prediction_plt.set(xticklabels=[])
#     prediction_plt.set(xlabel=None)
#     plot_file_path = Path(output_dir, 'predictions.png')
#     plt.savefig(plot_file_path)
#     logging.info(f'prediction plot saved at: {plot_file_path}\n')
#     plt.clf()


def plot_history(history, output_dir):
    # from: https://www.tensorflow.org/tutorials/keras/regression#linear_regression_with_one_variable
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Age]')
    plt.legend()
    plt.grid(True)
    plot_file_path = Path(output_dir, 'training_history.png')
    plt.savefig(plot_file_path)
    logging.info(f'training history plot saved at: {plot_file_path}\n')
    plt.clf()

def findDeviation(df, col1, col2):
    return (df[str(col1)] - df[str(col2)]).abs().mean()

def plot_categorical_deviation(args, df, attack_columns, output_dir):
    df.columns = df.columns.map(str)
    df0_14 = df.query('age < 15')
    df15_25 = df.query('age >= 15 and age <= 25')
    df26_50 = df.query('age > 25 and age <= 50')
    df51_65 = df.query('age > 50 and age <= 65')
    df_gt65 = df.query('age > 65')
    
    prediction_column = 'predictions'
    deviation_df = pd.DataFrame([], columns=['perturbation', '<15', '15-25', '26-50', '51-65', '>65'])
    
    for attack_column in attack_columns:
        c1 = (findDeviation(df0_14, prediction_column, attack_column))
        c2 = (findDeviation(df15_25, prediction_column, attack_column))
        c3 = (findDeviation(df26_50, prediction_column, attack_column))
        c4 = (findDeviation(df51_65, prediction_column, attack_column))
        c5 = (findDeviation(df_gt65, prediction_column, attack_column))
        deviation_df = deviation_df.append({'perturbation': attack_column, '<15': c1, '15-25': c2, '26-50': c3, '51-65': c4, '>65': c5}, ignore_index=True)
        
    fig, ax1 = plt.subplots(figsize=(10, 10))
    melted_df = deviation_df.melt(id_vars='perturbation', value_vars=['<15', '15-25', '26-50', '51-65', '>65'])
    deviation_plt = sns.barplot(x='perturbation', y='value', hue='variable', data=melted_df, ax=ax1)
    deviation_plt.legend()
    plot_file_path = Path(output_dir, f'{args.model}_{args.attack}_categorical_deviation.png')
    plt.savefig(plot_file_path)
    
    
def plot_comparison(args, perturbation_columns, output_dir):
    
    logging.info(f'plotting comparision of {args.attack} against all existing models')
    experiment_dir = Path(args.experiment_dir)
    model_dirs = [path for path in experiment_dir.iterdir() if path.is_dir() and path.name.endswith('_model')]
    prediction_column = 'predictions'
    model_names = [dir.name for dir in model_dirs]
    column_names = ['perturbation'] + model_names
    
    deviation_df = pd.DataFrame([], columns=column_names)
    deviation_df['perturbation'] = perturbation_columns

    for model_dir in model_dirs:
        logging.info(f'finding deviation for model: {model_dir.name}')
        
        model_deviation = []
        attack_results_file = Path(model_dir, f'{args.attack}.csv')
        if not attack_results_file.exists():
            logging.info(f'gsm.csv FileNotFound: Skipping comparison of {args.attack} attack for {model_dir.name}.')
            continue
        attack_df = pd.read_csv(attack_results_file)
        
        results_path = Path(model_dir, 'results.csv')
        results_df = pd.read_csv(results_path)
        
        for column_name in perturbation_columns:
            deviation = findDeviation(attack_df, prediction_column, column_name)
            logging.info(f'perturbation: {column_name}, deviation: {deviation}')
            model_deviation.append(deviation)
            results_column_name = f'{args.attack}_{column_name}' 
            if results_column_name in results_df.values:
                results_df.loc[results_df['metric'] == results_column_name, 'result/deviation'] = deviation
            else:
                results = {'metric':f'{args.attack}_{column_name}', 'result/deviation': deviation}
                results_df = results_df.append(results, ignore_index=True)
        
        model_name = model_dir.name
        deviation_df[model_name] = model_deviation
        results_df.to_csv(results_path, index=False)
    
    fig, ax1 = plt.subplots(figsize=(10, 10))
    melted_df = deviation_df.melt(id_vars='perturbation', value_vars=model_names)
    # if args.attack == 'gsm':
    #     x_column_name = 'eps (noise)'
    # elif args.attack == 'l0':
    #    x_column_name = '|.|'
    experiment_name = experiment_dir.name
    plt.title(f'{experiment_name} - {args.attack}')
    deviation_plt = sns.barplot(x='perturbation', y='value', hue='variable', data=melted_df, ax=ax1)
    deviation_plt.legend()
    plot_file_path = Path(output_dir, f'{args.attack}_comparison.png')
    plt.savefig(plot_file_path)
    