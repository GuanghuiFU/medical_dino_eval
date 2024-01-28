import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import os
import glob
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


def bootstrap_ci(data, statistic=np.mean, alpha=0.05, num_samples=5000):
    n = len(data)
    rng = np.random.RandomState(47)
    samples = rng.choice(data, size=(num_samples, n), replace=True)
    stat = np.sort(statistic(samples, axis=1))
    lower = stat[int(alpha / 2 * num_samples)]
    upper = stat[int((1 - alpha / 2) * num_samples)]
    return lower, upper


def cal_avg_bootstrap_confidence_interval(x):
    x_avg = np.average(x)
    bootstrap_ci_result = bootstrap_ci(x)
    return np.round(x_avg, 4), np.round(bootstrap_ci_result[0], 4), np.round(bootstrap_ci_result[1], 4)


def calculate_metrics(csv_path, average='weighted'):
    df = pd.read_csv(csv_path, skipfooter=1, engine='python')
    y_true = df['true_label']
    y_pred = df['prediction']

    bootstrapped_accuracies = []
    bootstrapped_precisions = []
    bootstrapped_recalls = []
    bootstrapped_f1_scores = []
    for _ in range(5000):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        sampled_y_true = y_true.iloc[indices]
        sampled_y_pred = y_pred.iloc[indices]

        accuracy = accuracy_score(sampled_y_true, sampled_y_pred)
        if average == 'macro': precision, recall, f1, _ = precision_recall_fscore_support(sampled_y_true, sampled_y_pred, average='macro')
        elif average == 'micro': precision, recall, f1, _ = precision_recall_fscore_support(sampled_y_true, sampled_y_pred, average='micro')
        elif average == 'weighted': precision, recall, f1, _ = precision_recall_fscore_support(sampled_y_true, sampled_y_pred, average='weighted')
        else: precision, recall, f1, _ = precision_recall_fscore_support(sampled_y_true, sampled_y_pred, average=None)
        bootstrapped_accuracies.append(accuracy)
        bootstrapped_precisions.append(precision)
        bootstrapped_recalls.append(recall)
        bootstrapped_f1_scores.append(f1)

    accuracy_avg_ci = cal_avg_bootstrap_confidence_interval(np.array(bootstrapped_accuracies))
    precision_avg_ci = cal_avg_bootstrap_confidence_interval(np.array(bootstrapped_precisions))
    recall_avg_ci = cal_avg_bootstrap_confidence_interval(np.array(bootstrapped_recalls))
    f1_avg_ci = cal_avg_bootstrap_confidence_interval(np.array(bootstrapped_f1_scores))

    return accuracy_avg_ci, precision_avg_ci, recall_avg_ci, f1_avg_ci


def cal_csv_bci(csv_path, average):
    # Prepare the output strings
    task_model_str = 'Task: ' + csv_path.split('/')[-3] + '; Model: ' + csv_path.split('/')[-2]
    accuracy_ci, precision_ci, recall_ci, f1_ci = calculate_metrics(csv_path, average)
    accuracy_str = f"Accuracy: {accuracy_ci[0]} ({accuracy_ci[1]}, {accuracy_ci[2]})"
    precision_str = f"Precision: {precision_ci[0]} ({precision_ci[1]}, {precision_ci[2]})"
    recall_str = f"Recall: {recall_ci[0]} ({recall_ci[1]}, {recall_ci[2]})"
    f1_str = f"F1 Score: {f1_ci[0]} ({f1_ci[1]}, {f1_ci[2]})"
    print(task_model_str)
    print(accuracy_str)
    print(precision_str)
    print(recall_str)
    print(f1_str)
    return task_model_str, accuracy_str, precision_str, recall_str, f1_str


def main():
    eval_dir = 'prediction/result-public/'
    csv_path_list = sorted(glob.glob(f'{eval_dir}/*/*/*_best_a*_prediction.csv'))
    print(csv_path_list)
    average = 'weighted'
    base_name = os.path.dirname(eval_dir)
    print(f'Saving performance to: {base_name}_{average}.txt')
    with open(f'{base_name}_{average}.txt', 'w') as file:
        for csv_path in csv_path_list:
            task_model_str, accuracy_str, precision_str, recall_str, f1_str = cal_csv_bci(csv_path, average=average)
            file.write(f'{task_model_str}\n')
            file.write(f'{accuracy_str}\n')
            file.write(f'{precision_str}\n')
            file.write(f'{recall_str}\n')
            file.write(f'{f1_str}\n')

if __name__ == '__main__':
    main()
