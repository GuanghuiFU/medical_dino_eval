import os
from tqdm import tqdm
def process_sentence_list(sentence_list, run=False, turn_off=False):
    for sentence in tqdm(sentence_list):
        print(sentence)
        if run:
            os.system(sentence)
    if turn_off:
        os.system("shutdown -s -t  60")
model_list = ['Dino_s', 'Dino_b', 'VGG16', 'ResNet', 'DenseNet']
freeze_list = [True, False]
script_list = []

for model_name in model_list[3:]:
    for freeze in freeze_list:
        # Construct the script for each combination
        script = f'python model_train.py --num_epoch=100 --batch_size=32 --optimizer_name=Adam --model_name={model_name} --pretrain --train_ratio=1.0'
        if freeze:
            script += ' --freeze'
        script_list.append(script)

process_sentence_list(script_list[1:], run=True, turn_off=False)
