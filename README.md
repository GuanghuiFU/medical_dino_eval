# Comparative Analysis of ImageNet Pre-Trained Deep Learning Models and DINOv2 in Medical Imaging Classification

This repository contains material associated to this [paper](#Citation).
If you use this material, we would appreciate if you could cite the following reference.

## Citation
* Anonymous authors. Comparative Analysis of ImageNet Pre-Trained Deep Learning Models and DINOv2 in Medical Imaging Classification. Preprint.

## Code explanations
* [`main_script_train.py`](<main_script_train.py>): build model training scripts, you can easily set up and let the code run.
* [`train_model.py`](<train_model.py>): model training function, the dataset setting is here
* [`utils.py`](<utils.py>): basic files for various modules, including definition of data sets, model definition, data partitioning, model training, etc.
* [`inference_biu.py`](<inference_biu.py>): inference the trained model on the test set and save the results for evaluation
* [`eval_biu.py`](<eval_biu.py>): model evaluation module, uses 95% bootstrap confidence interval, and calculates Precision, recall, and F1-score in a weighted average way. Save the final performance into a txt file for generating LaTex files
* [`perf2latex.py`](<perf2latex.py>): read performance txt files and generate LaTex files

## Related codes

1. **[DINOv2](<https://github.com/facebookresearch/dinov2>)** [1]: https://github.com/facebookresearch/dinov2

We use the pretrained DINOv2 classifiers that loaded via PyTorch Hub:
```python
import torch
# DINOv2
dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
dinov2_vitl14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
```

2. **[TorchVision](<https://pytorch.org/vision/stable/models.html>)**: https://pytorch.org/vision/stable/models.html

We use the ImageNet pretrained weighted for different models:
```python
from torchvision import models
# Pretrained models
vgg_model = models.vgg16(pretrained=True)
resnet_model = models.resnet50(pretrained=True)
densenet_model = models.densenet121(pretrained=True)
```

## Related public datasets

1. **Chest X-ray [2]:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **iChallenge-AMD (Eye funds) [3]:** iChallenge-AMD-Training 400: 
   1. Introduction: https://refuge.grand-challenge.org/iChallenge-AMD/
   2. Download: https://ai.baidu.com/broad/download
3. **HAM10000 (Skin lesions) [4]:** HAM1000, ISIC2018: https://challenge.isic-archive.com/data/#2018


## References

1. Oquab, Maxime, et al. "DINOv2: Learning robust visual features without supervision." *arXiv preprint arXiv:2304.07193* (2023).
2. Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." *cell* 172.5 (2018): 1122-1131.
3. Orlando, José Ignacio, et al. "Refuge challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs." *Medical image analysis* 59 (2020): 101570.
4. Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific data* 5.1 (2018): 1-9.
