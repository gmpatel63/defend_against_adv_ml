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
    return (df[col1] - df[col2]).abs().mean()

def plot_categorical_deviation(args, df, attack_columns, output_dir):
  
    df0_14 = df.query('age < 15')
    df15_25 = df.query('age >= 15 and age <= 25')
    df26_50 = df.query('age > 25 and age <= 50')
    df51_65 = df.query('age > 50 and age <= 65')
    df_gt65 = df.query('age > 65')
    
    prediction_column = 'predictions'
    deviation_df = pd.DataFrame([], columns=['attack', '<15', '15-25', '26-50', '51-65', '>65'])
    
    for attack_column in attack_columns:
        c1 = (findDeviation(df0_14, prediction_column, attack_column))
        c2 = (findDeviation(df15_25, prediction_column, attack_column))
        c3 = (findDeviation(df26_50, prediction_column, attack_column))
        c4 = (findDeviation(df51_65, prediction_column, attack_column))
        c5 = (findDeviation(df_gt65, prediction_column, attack_column))
        deviation_df = deviation_df.append({'attack': attack_column, '<15': c1, '15-25': c2, '26-50': c3, '51-65': c4, '>65': c5}, ignore_index=True)
        
    fig, ax1 = plt.subplots(figsize=(10, 10))
    melted_df = deviation_df.melt(id_vars='attack', value_vars=['<15', '15-25', '26-50', '51-65', '>65'])
    deviation_plt = sns.barplot(x='attack', y='value', hue='variable', data=melted_df, ax=ax1)
    deviation_plt.legend()
    plot_file_path = Path(output_dir, f'{args.attack}_deviation.png')
    plt.savefig(plot_file_path)
    
    
def plot_overall_deviation(args, df, attack_columns, output_dir):
    pass