import re
import os


def read_text_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def convert_metric_to_percentage(metric):
    # Extract the main value and the confidence interval
    main_value, interval = re.match(r"([\d.]+) \(([\d., ]+)\)", metric).groups()
    # Convert to percentage
    main_value_percentage = f"{float(main_value) * 100:.2f}"
    interval_values = interval.split(', ')
    interval_percentage = f"[{float(interval_values[0]) * 100:.2f}, {float(interval_values[1]) * 100:.2f}]"
    return f"{main_value_percentage} {interval_percentage}"


def convert_to_latex_table(text):
    tasks = text.strip().split("Task:")
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|c|c|}\n"
    latex_table += "\\hline\n"
    latex_table += "Task & Model & Freeze & Accuracy & Precision & Recall & F1 Score \\\\ \n"
    latex_table += "\\hline\n"

    for task in tasks:
        if task:
            task_name = re.search(r"(\d+_.*?);", task).group(1).strip()
            model_name = re.search(r"Model: (\w+)", task).group(1).strip()
            freeze = "Yes" if "f" in model_name else "No"
            if freeze == "Yes":
                model_name = model_name.replace("_f", "")

            accuracy = convert_metric_to_percentage(re.search(r"Accuracy: ([\d.]+ \([\d., ]+\))", task).group(1))
            precision = convert_metric_to_percentage(re.search(r"Precision: ([\d.]+ \([\d., ]+\))", task).group(1))
            recall = convert_metric_to_percentage(re.search(r"Recall: ([\d.]+ \([\d., ]+\))", task).group(1))
            f1_score = convert_metric_to_percentage(re.search(r"F1 Score: ([\d.]+ \([\d., ]+\))", task).group(1))

            latex_table += f"{task_name} & {model_name} & {freeze} & {accuracy} & {precision} & {recall} & {f1_score} \\\\ \n"
            latex_table += "\\hline\n"

    latex_table += "\\end{tabular}\n\\caption{Performance Metrics}\n\\end{table}"
    return latex_table


average_type = 'weighted'
file_path = f'prediction/result-private_{average_type}.txt'
basename = os.path.basename(file_path).replace('.txt', '')
text = read_text_from_file(file_path)
latex_table = convert_to_latex_table(text)
print(latex_table)
with open(f'prediction/{basename}_latex.txt', 'w') as file:
    file.write(f'{latex_table}\n')
