import os
import errno
import random
import torch
import csv
torch.manual_seed(1)
random.seed(1)
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler, Subset, Dataset, random_split
from torchvision import models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

basic_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_medical = transforms.Compose([
    transforms.Resize(256),  # Resize the input image to 256x256 pixels
    transforms.CenterCrop(224),  # Crop the central part of the image (224x224 pixels)
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(  # Normalize the image with mean and standard deviation
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Basic Function
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


# Training related function
def choose_device(args):
    if args.machine == 'Mac':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def load_image(img: str) -> torch.Tensor:
    img = Image.open(img)
    transformed_img = transform_medical(img)[:3].unsqueeze(0)
    return transformed_img


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def choose_model(args, categories):
    model_name = args.model_name
    optimizer_name = args.optimizer_name
    learning_rate = args.learning_rate
    momentum = args.momentum
    device = choose_device(args)
    if model_name == 'VGG16':
        model = models.vgg16(pretrained=False)
        if args.freeze: freeze_model(model.features)
        model.classifier = torch.nn.Linear(model.classifier.in_features, categories)
        model = model.to(device)
    elif model_name == 'ResNet':
        model = models.resnet50(pretrained=False)
        if args.freeze:
            freeze_model(model)
            model.fc.requires_grad = True
        model.classifier = torch.nn.Linear(model.classifier.in_features, categories)
        model = model.to(device)
    elif model_name == "DenseNet":
        model = models.densenet121(pretrained=False)
        if args.freeze:
            freeze_model(model.features)
        model.classifier = torch.nn.Linear(model.classifier.in_features, categories)
        model = model.to(device)
    elif model_name == "Dino_s":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        num_features = dino_model.linear_head.out_features
        model = DinoModel(dino_model, num_features, categories, freeze_dino_model=args.freeze)
    elif model_name == "Dino_b":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        num_features = dino_model.linear_head.out_features
        model = DinoModel(dino_model, num_features, categories, freeze_dino_model=args.freeze)
    elif model_name == "Dino_l":
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        num_features = dino_model.linear_head.out_features
        model = DinoModel(dino_model, num_features, categories, freeze_dino_model=args.freeze)
    else:
        raise ValueError('Wrong model name.')
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion


def calculate_metrics(y_true, y_pred, categories):
    accuracy = accuracy_score(y_true, y_pred)
    if categories != 2:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    else:
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return accuracy, precision, recall, f1


def write_args(args, data_info):
    model_name_save = args.model_name
    if args.pretrain: model_name_save += '_p'
    if args.freeze: model_name_save += '_f'
    argsDict = args.__dict__
    for k in args.__dict__:
        print(k + ":" + str(args.__dict__[k]))
    with open(f"./log/{args.dataset_name}_{model_name_save}_{args.local_time}_setting.txt", 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines(data_info)
        f.writelines('------------------- end -------------------')


def train_model(args, model, optimizer, criterion, categories, trainloader, valloader):
    num_epochs = args.num_epochs
    device = choose_device(args)
    best_val_accuracy = 0
    best_val_loss = 20
    model_name_save = args.model_name
    if args.pretrain: model_name_save += '_p'
    if args.freeze: model_name_save += '_f'
    with open(f"./log/{args.dataset_name}_{model_name_save}_{args.local_time}_losses.csv", "w",
              newline="") as loss_file:
        fieldnames = ["Epoch", "Train Loss", "Train Accuracy", "Train Precision", "Train Recall", "Train F1",
                      "Val Loss", "Val Accuracy", "Val Precision", "Val Recall", "Val F1", ]
        csv_writer = csv.DictWriter(loss_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs + 1}:")
            running_loss = []
            running_accuracy = []
            running_precision = []
            running_recall = []
            running_f1 = []
            # Train
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                # print(outputs.size(), labels.size())
                # print('outputs:', outputs, '. labels:', labels)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                outputs = torch.argmax(outputs, dim=1)
                train_accuracy, train_precision, train_recall, train_f1 = calculate_metrics(
                    labels.cpu().detach().numpy(), outputs.cpu().detach().numpy(), categories)
                # print('Training....')
                # print('Label:', labels)
                # print('Outputs:', outputs)
                # print('train_accuracy, train_precision, train_recall, train_f1:',train_accuracy, train_precision, train_recall, train_f1)
                running_accuracy.append(train_accuracy)
                running_precision.append(train_precision)
                running_recall.append(train_recall)
                running_f1.append(train_f1)
            epoch_loss = np.average(running_loss)
            epoch_accuracy = np.average(running_accuracy)
            epoch_precision = np.average(running_precision)
            epoch_recall = np.average(running_recall)
            epoch_f1 = np.average(running_f1)
            print(
                f"\t Train - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy * 100:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}, F1: {epoch_f1:.4f}")

            # Validation
            val_running_loss = []
            val_running_accuracy = []
            val_running_precision = []
            val_running_recall = []
            val_running_f1 = []
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valloader, 0):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss = criterion(outputs, labels)
                    val_running_loss.append(val_loss.item())
                    outputs = torch.argmax(outputs, dim=1)
                    val_accuracy, val_precision, val_recall, val_f1 = calculate_metrics(labels.cpu().detach().numpy(),
                                                                                        outputs.cpu().detach().numpy(),
                                                                                        categories)
                    # print('Validation....')
                    # print('Label:', labels)
                    # print('Outputs:', outputs)
                    # print('val_accuracy, val_precision, val_recall, val_f1:', val_accuracy, val_precision, val_recall, val_f1)
                    val_running_accuracy.append(val_accuracy)
                    val_running_accuracy.append(val_accuracy)
                    val_running_precision.append(val_precision)
                    val_running_recall.append(val_recall)
                    val_running_f1.append(val_f1)
            val_epoch_loss = np.average(val_running_loss)
            val_epoch_accuracy = np.average(val_running_accuracy)
            val_epoch_precision = np.average(val_running_precision)
            val_epoch_recall = np.average(val_running_recall)
            val_epoch_f1 = np.average(val_running_f1)
            print(
                f"\t Val - Loss: {val_epoch_loss:.4f}, Accuracy: {val_epoch_accuracy * 100:.4f}, Precision: {val_epoch_precision:.4f}, Recall: {val_epoch_recall:.4f}, F1: {val_epoch_f1:.4f}")

            csv_writer.writerow({
                "Epoch": epoch + 1,
                "Train Loss": epoch_loss,
                "Train Accuracy": epoch_accuracy,
                "Train Precision": epoch_precision,
                "Train Recall": epoch_recall,
                "Train F1": epoch_f1,
                "Val Loss": val_epoch_loss,
                "Val Accuracy": val_epoch_accuracy,
                "Val Precision": val_epoch_precision,
                "Val Recall": val_epoch_recall,
                "Val F1": val_epoch_f1
            })

            # loss_file.write(f"{epoch} {epoch_loss:.4f} {epoch_accuracy:.4f} {epoch_precision:.4f} {epoch_recall:.4f} {epoch_f1:.4f} {val_epoch_loss:.4f} {val_epoch_accuracy:.4f} {val_epoch_precision:.4f} {val_epoch_recall:.4f} {val_epoch_f1:.4f}\n")
            # model_save_path = f"./model/{args.dataset_name}_{args.loader_name}_{args.local_time}_model.pt"

            if val_epoch_accuracy > best_val_accuracy:
                best_val_accuracy = val_epoch_accuracy
                model_save_path = f"./model/{args.dataset_name}_{model_name_save}_{args.local_time}_best_accuracy_model.pt"
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"\t New best model saved with Val Accuracy: {best_val_accuracy * 100:.2f}%, Loss: {val_epoch_loss}")
            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                model_save_path = f"./model/{args.dataset_name}_{model_name_save}_{args.local_time}_best_loss_model.pt"
                torch.save(model.state_dict(), model_save_path)
                print(
                    f"\t New best model saved with Val Loss: {val_epoch_loss}, Accuracy: {val_epoch_accuracy * 100:.2f}%")


# Dataset
def create_dataset(paths_labels):
    biu_dataset = BIUDataset(paths_labels, transform=transform_medical)
    # Split the training set into train and validation in 8:2
    train_size = int(0.8 * len(biu_dataset))
    val_size = len(biu_dataset) - train_size
    trainset, valset = random_split(biu_dataset, [train_size, val_size])
    categories = 3
    img_channel = 3
    return categories, img_channel, trainset, valset


def create_loader(args, trainset, valset):
    batch_size = args.batch_size
    num_workers = args.num_workers
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return trainloader, valloader


class BIUDataset(Dataset):
    def __init__(self, paths_labels, transform=None):
        self.transform = transform
        self.images = []
        for path, label in paths_labels.items():
            self.images.extend([(os.path.join(path, img), label) for img in os.listdir(path)])
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, label = self.images[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# This is almost the same as the BIUDataset class, except that it returns one more item, the image name, for model inference.
class BIUDataset_name(Dataset):
    def __init__(self, paths_labels, transform=None):
        self.transform = transform
        self.images = []
        for path, label in paths_labels.items():
            self.images.extend([(os.path.join(path, img), label) for img in os.listdir(path)])
        random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, label = self.images[index]
        image_name = os.path.basename(img_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, image_name

def create_subset(dataset, subset_ratio):
    category_indices = {0: [], 1: [], 2: []}

    for idx, (_, label) in enumerate(dataset):
        category_indices[label].append(idx)

    subset_indices = []
    for cat_indices in category_indices.values():
        subset_size = int(len(cat_indices) * subset_ratio)
        subset_indices.extend(cat_indices[:subset_size])

    random.shuffle(subset_indices)
    return Subset(dataset, subset_indices)


def count_images_in_subset(subset):
    category_count = {}
    for idx in subset.indices:
        _, label = subset.dataset[idx]
        if label not in category_count:
            category_count[label] = 0
        category_count[label] += 1
    return len(subset.indices), category_count


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]
        with open(image_path, 'rb') as f:
            image = Image.open(f)
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CustomSampler(Sampler):
    def __init__(self, data_source, order):
        super().__init__(data_source)
        self.data_source = data_source
        self.order = order

    def __iter__(self):
        return iter(self.order)

    def __len__(self):
        return len(self.data_source)


# Model
class DinoModel(torch.nn.Module):
    def __init__(self, dino_model, num_features, num_classes, freeze_dino_model=False):
        super(DinoModel, self).__init__()
        self.dino_model = dino_model
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        if freeze_dino_model:
            for param in self.dino_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.dino_model(x)
        x = self.classifier(x)
        return x
