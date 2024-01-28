from sklearn.metrics import precision_score, recall_score, f1_score
from biu_dino.utils import *
import glob

num_classes = 3  # Update this to match the number of classes in your dataset
device = torch.device("cuda")
model_name_list = ['Dino_s', 'Dino_b', 'VGG16', 'ResNet', 'DenseNet']
modality_name = 'T1'
for model_name in model_name_list:
    model_base_path = f"your/model/path/dir"
    # Two kind of way to save the model, save base on best accuracy or best loss. In the result report part, I evaluate based on the best accuracy model
    best_of_list = ['accuracy', 'loss']
    for best_of in best_of_list[0]:
        model_path_list = glob.glob(f'{model_base_path}/*_{model_name}_p_2024*_best_{best_of}*.pt')
        category1_train_path = f"/your/train_dataset/category1"
        category2_train_path = f"/your/train_dataset/category2"
        category3_train_path = f"/your/train_dataset/category3"
        # Load the images from the specified folder
        chest_xray_test_dataset = BIUDataset_name(category1_train_path, category2_train_path, category3_train_path, transform=transform_medical)
        dataloader = DataLoader(chest_xray_test_dataset, batch_size=1, shuffle=False)

        for model_path in model_path_list:
            print('model_path:',model_path)
            data_name = os.path.basename(model_path).split('_')[0]
            print('Data name:', data_name)
            mkdir(f"./prediction/{model_name}/")
            csv_file_path = f"./prediction/{model_name}/{data_name}_best_{best_of}_prediction.csv"
            if model_name == 'VGG16':
                model = models.vgg16(pretrained=True)
                model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
            elif model_name == 'ResNet':
                model = models.resnet50(pretrained=True)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif model_name == "DenseNet":
                model = models.densenet121(pretrained=True)
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            elif model_name == "Dino_s":
                dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
                num_features = dino_model.linear_head.out_features
                model = DinoModel(dino_model, num_features, num_classes)
            elif model_name == "Dino_b":
                dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
                num_features = dino_model.linear_head.out_features
                model = DinoModel(dino_model, num_features, num_classes)
            elif model_name == "Dino_l":
                dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
                num_features = dino_model.linear_head.out_features
                model = DinoModel(dino_model, num_features, num_classes)
            else:
                raise ValueError('Wrong model name.')
            model.load_state_dict(torch.load(model_path))
            model.to(device)
            model.eval()

            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    ["image_name", "prediction", "true_label", "accuracy", "precision", "recall", "f1_score"])

                # Iterate over the images in the dataloader
                y_true = []
                y_pred = []
                for batch_idx, (inputs, labels, image_name) in enumerate(dataloader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Make predictions
                    with torch.no_grad():
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)

                    # Save the predictions and true labels for evaluation
                    y_true.append(labels.item())
                    y_pred.append(predicted.item())

                    # Compute evaluation metrics for the current image
                    accuracy = accuracy_score([labels.item()], [predicted.item()])
                    precision = precision_score([labels.item()], [predicted.item()], average='weighted', zero_division=0)
                    recall = recall_score([labels.item()], [predicted.item()], average='weighted', zero_division=0)
                    f1 = f1_score([labels.item()], [predicted.item()], average='weighted', zero_division=0)

                    # Write the results to the CSV file
                    csv_writer.writerow([image_name, predicted.item(), labels.item(), accuracy, precision, recall, f1])

            # Compute overall evaluation metrics
            overall_accuracy = accuracy_score(y_true, y_pred)
            overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            print(f"Overall Accuracy:{overall_accuracy:.4f}")
            print(f"Overall Precision: {overall_precision:.4f}")
            print(f"Overall Recall: {overall_recall:.4f}")
            print(f"Overall F1-score: {overall_f1:.4f}")
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Overall", "", "", overall_accuracy, overall_precision, overall_recall, overall_f1])
