import argparse
from biu_dino.utils import *
import time
torch.manual_seed(1)
random.seed(1)

def main():
    args = get_arguments()
    local_time = time.strftime('%Y%m%d-%H%M', time.localtime())
    args.local_time = local_time
    categories, img_channel, trainset, valset = create_dataset()
    args.categories = categories
    percentage = args.train_ratio
    if percentage != 1.0:
        train_dataset, val_dataset = trainset, valset
        print('Train set original:', len(train_dataset))
        print('Val set original:', len(val_dataset))
        train_subset = create_subset(train_dataset, percentage=percentage)
        val_subset = create_subset(val_dataset, percentage=percentage)
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
    parser.add_argument('--dataset_name', type=str, default="BIUDataset", choices=('BIUDataset'))
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--machine', type=str, default="PC", choices=('PC', 'Mac'))
    parser.add_argument('--model_name', type=str, default="VGG16", choices=('VGG16', 'ResNet', 'DenseNet', 'Dino_s', 'Dino_b', 'Dino_l'))
    parser.add_argument('--pretrain', action="store_true")
    parser.add_argument('--freeze', action="store_true")
    parser.add_argument('--optimizer_name', type=str, default="Adam", choices=('SGD', 'Adam'))
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_epochs', default=300, type=int)
    parser.add_argument('--train_ratio', default=1.0, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
