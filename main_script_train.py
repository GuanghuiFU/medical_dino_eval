import os
from tqdm import tqdm


def process_sentence_list(sentence_list, run=False):
    for sentence in tqdm(sentence_list):
        print(sentence)
        if run:
            os.system(sentence)


# Basic setting
dataset_list = ['chest_xray', 'eye_fund', 'skin_dermoscopy']
model_list = ['VGG16', 'ResNet', 'DenseNet', 'Dino_s', 'Dino_b', 'Dino_l']
freeze_list = [True, False]

script_list = []
# Select the dataset and the model you plan to train:
dataset_name = dataset_list[0]
model_name = model_list[0]
# Freeze or un-freeze the backbone for model training
for freeze in freeze_list:
    # Construct the script for each combination
    script = f'python train_model.py --dataset_name={dataset_name} --num_epoch=100 --batch_size=32 --optimizer_name=Adam --model_name={model_name} --pretrain'
    if freeze:
        script += ' --freeze'
    script_list.append(script)

process_sentence_list(script_list[1:], run=True)
