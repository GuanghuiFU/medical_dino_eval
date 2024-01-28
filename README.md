# Comparative Analysis of ImageNet Pre-Trained Deep Learning Models and DINOv2 in Medical Imaging Classification

* Conference version:

## Introduction

This repository hosts the code for our research, which concentrates on applying transfer learning to medical image analysis. Specifically, it includes glioma grading using three modalities of brain MRI data and additional classification tasks on three public datasets: chest X-ray, eye fundus, and skin dermoscopy. Our study investigates the effectiveness of various pre-trained deep learning models, such as DINOv2 and those based on ImageNet, and assesses their performance across diverse medical tasks. The code is publicly available to facilitate reproducibility, though the clinical data used remains confidential to ensure patient privacy is maintained.


## Code explanation

* `utils.py`: Basic files for various modules, including definition of data sets, model definition, data partitioning, model training, etc.
* `model_train.py`: model training file
* `train_script.py`: Batch training model
* `inference_biu.py`: Test the trained model on the test set and save the results to
* `eval_biu.py`: Model verification module, uses 95% bootstrap confidence interval as verification, and calculates Precision, recall, and F1-score in a weighted average way. Save the final performance into a txt file for generating LaTex files
* `perf2latex.py`: Read performance txt files and generate LaTex files

## Related codes

1. DINOv2 [1]: https://github.com/facebookresearch/dinov2

## Related public dataset

1. **Chest X-ray [2]:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. **iChallenge-AMD (Eye funds) [3]:** iChallenge-AMD-Training 400: 
   1. Introduction: https://refuge.grand-challenge.org/iChallenge-AMD/
   2. Download: https://ai.baidu.com/broad/download
3. **HAM10000 (Skin lesions) [4]:** HAM1000, ISIC2018: https://challenge.isic-archive.com/data/#2018


## References

1. Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." *arXiv preprint arXiv:2304.07193* (2023).
2. Kermany, Daniel S., et al. "Identifying medical diagnoses and treatable diseases by image-based deep learning." *cell* 172.5 (2018): 1122-1131.
3. Orlando, José Ignacio, et al. "Refuge challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs." *Medical image analysis* 59 (2020): 101570.
4. Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." *Scientific data* 5.1 (2018): 1-9.

## Citing us

* Yuning Huang, Jingchen Zou, Lanxi Meng, Xin Yue, Qing Zhao, Jianqiang Li, Changwei Song, Gabriel Jimenez, Shaowu Li, **Guanghui Fu**."Comparative Analysis of ImageNet Pre-Trained Deep Learning Models and DINOv2 in Medical Imaging Classification. In *IEEE COMPSAC*. 2024.

```
@inproceedings{huang2024comparative,
  title={Comparative Analysis of ImageNet Pre-Trained Deep Learning Models and DINOv2 in Medical Imaging Classification},
  author={Yuning Huang, Jingchen Zou, Lanxi Meng, Xin Yue, Qing Zhao, Jianqiang Li, Changwei Song, Gabriel Jimenez, Shaowu Li, Guanghui Fu},
  booktitle={Proc.IEEE COMPSAC 2024},
  year={2024}
}
```
