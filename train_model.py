import argparse
from utils import *
import time
torch.manual_seed(1)
random.seed(1)

def main():
    args = get_arguments()
    local_time = time.strftime('%Y%m%d-%H%M', time.localtime())
    args.local_time = local_time
    '''
    Set up the train set. The code will automatically split train and validation set by 8:2. So you only need to give the folder with different categories.
    For example, the ChestX-ray dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
    already split the train and test set, so we just need to put the two categories (normal, pneumonia) as below, and give them the label tag (0, 1).
    '''
    paths_labels = {
        'your/dataset/path/chest_xray/train/NORMAL': 0,
        'your/dataset/path/chest_xray/train/PNEUMONIA': 1,
    }
    categories, img_channel, trainset, valset = create_dataset(paths_labels)
    args.categories = categories
    subset_ratio = args.subset_ratio
    if subset_ratio != 1.0: # if you want to try different percentage of train/val dataset, you can use this setting. Note that this is not for train, test split, this is for use part of training set for experiment.
        train_dataset, val_dataset = trainset, valset
        print('Train set original:', len(train_dataset))
        print('Val set original:', len(val_dataset))
        train_subset = create_subset(train_dataset, subset_ratio)
        val_subset = create_subset(val_dataset, subset_ratio)
        print('train_subset set original:', len(train_subset))
        print('val_subset set original:', len(val_subset))
        # val_subset = val_dataset # For EyeFundus, we load the full validation set
        real_trainset = train_subset
        real_valset = val_subset
        print('real_trainset:', len(real_trainset))
        print('real_valset:', len(real_valset))
    else:
        real_trainset = trainset
        real_valset = valset
    train_total, train_category_count = count_images_in_subset(real_trainset)
    val_total, val_category_count = count_images_in_subset(real_valset)
    data_info = f"Train set size: {train_total}, Train category statistics: {train_category_count} \n Val set size: {val_total}, Val category statistics: {val_category_count} \n "
    print(data_info)
    write_args(args, data_info)
    trainloader, valloader = create_loader(args, real_trainset, real_valset)
    model, optimizer, criterion = choose_model(args, categories, img_channel)
    train_model(args, model, optimizer, criterion, categories, trainloader, valloader)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="chest_xray", choices=('chest_xray', 'eye_fund', 'skin_dermoscopy'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--machine', type=str, default="Cluster", choices=('Cluster', 'Mac'))
    parser.add_argument('--model_name', type=str, default="VGG16", choices=('VGG16', 'ResNet', 'DenseNet', 'Dino_s', 'Dino_b', 'Dino_l'))
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--freeze', action="store_true")
    parser.add_argument('--optimizer_name', type=str, default="Adam", choices=('SGD', 'Adam'))
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--subset_ratio', default=1.0, type=float) # if you want to try different percentage of train/val dataset, you can use this setting. Note that this is not for train, test split, this is for use part of training set for experiment.
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
