# PFENet++
This is the implementation of our paper [**PFENet++: Boosting Few-Shot Semantic Segmentation With the Noise-Filtered Context-Aware Prior Mask**](https://ieeexplore.ieee.org/document/10305430) that has been accepted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). 


# Get Started

### Environment
+ torch==1.7.1 
+ numpy==1.20.3
+ tensorboardX==2.2
+ opencv-python==4.5.2.52


### Datasets and Data Preparation

Please download the following datasets:

+ PASCAL-5i is based on the [**PASCAL VOC 2012**](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [**SBD**](http://home.bharathh.info/pubs/codes/SBD/download.html) where the val images should be excluded from the list of training samples.

Images are available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

annotations: https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing

Note: If you wish to reproduce the results presented in the paper, please follow the provided [**datalist**](https://github.com/dvlab-research/PFENet/tree/master/lists) to use the specified data. As we followed Shaban's OSLSM work, we only utilized a subset of the complete dataset, which is not the entire 12,000 entries. However, different studies may have different usage requirements. To ensure fair comparison, we kindly request you to select the data according to your specific needs.

+ [**COCO 2014**](https://cocodataset.org/#download).

This code reads data from .txt files where each line contains the paths for image and the correcponding label respectively. Image and label paths are seperated by a space. Example is as follows:

    image_path_1 label_path_1
    image_path_2 label_path_2
    image_path_3 label_path_3
    ...
    image_path_n label_path_n

Then update the train/val/test list paths in the config files.

#### [Update] We have uploaded the lists we use in our paper.
+ The train/val lists for COCO contain 82081 and 40137 images respectively. They are the default train/val splits of COCO. 
+ The train/val lists for PASCAL5i contain 5953 and 1449 images respectively. The train list should be **voc_sbd_merge_noduplicate.txt** and the val list is the original val list of pascal voc (**val.txt**).

##### To get voc_sbd_merge_noduplicate.txt:
+ We first merge the original VOC (voc_original_train.txt) and SBD ([**sbd_data.txt**](http://home.bharathh.info/pubs/codes/SBD/train_noval.txt)) training data. 
+ [**Important**] sbd_data.txt does not overlap with the PASCALVOC 2012 validation data.
+ The merged list (voc_sbd_merge.txt) is then processed by the script (duplicate_removal.py) to remove the duplicate images and labels.

### Run Demo / Test with Pretrained Models
+ Please download the pretrained models.
+ We provide **8 pre-trained models**: 8 ResNet-50 based [**models**](https://drive.google.com/drive/folders/1jeOJBasHjuX2Teyh1GIXI4bE4h9Vr4ig?usp=sharing.
+ Update the config file by speficifying the target **split** and **path** (`weights`) for loading the checkpoint.
+ Execute `mkdir initmodel` at the root directory.
+ Download the ImageNet pretrained [**backbones**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155122171_link_cuhk_edu_hk/EQEY0JxITwVHisdVzusEqNUBNsf1CT8MsALdahUhaHrhlw?e=4%3a2o3XTL&at=9) and put them into the `initmodel` directory.
+ Then execute the command: 

    `sh test.sh {*dataset*} {*model_config*}`

Example: Test PFENet with ResNet50 on the split 0 of PASCAL-5i: 

    sh test.sh pascal split0_resnet50


### Train

Execute this command at the root directory: 

    sh train.sh {*dataset*} {*model_config*}


# Related Repositories

This project is built upon **PFENet**: https://github.com/dvlab-research/PFENet. 

Other projects in few-shot segmentation:
+ OSLSM: https://github.com/lzzcd001/OSLSM
+ CANet: https://github.com/icoz69/CaNet
+ PANet: https://github.com/kaixin96/PANet
+ FSS-1000: https://github.com/HKUSTCV/FSS-1000
+ AMP: https://github.com/MSiam/AdaptiveMaskedProxies
+ On the Texture Bias for FS Seg: https://github.com/rezazad68/fewshot-segmentation
+ SG-One: https://github.com/xiaomengyc/SG-One
+ FS Seg Propogation with Guided Networks: https://github.com/shelhamer/revolver
+ PFENet: https://github.com/dvlab-research/PFENet


Many thanks to their greak work!

# Citation

If you find this project useful, please consider citing:
```
@ARTICLE{luo2023pfenet++,
  author={Luo, Xiaoliu and Tian, Zhuotao and Zhang, Taiping and Yu, Bei and Tang, Yuan Yan and Jia, Jiaya},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={PFENet++: Boosting Few-Shot Semantic Segmentation With the Noise-Filtered Context-Aware Prior Mask}, 
  year={2024},
  volume={46},
  number={2},
  pages={1273-1289},
  doi={10.1109/TPAMI.2023.3329725}}
  
@article{tian2020pfenet,
  title={Prior Guided Feature Enrichment Network for Few-Shot Segmentation},
  author={Tian, Zhuotao and Zhao, Hengshuang and Shu, Michelle and Yang, Zhicheng and Li, Ruiyu and Jia, Jiaya},
  journal={TPAMI},
  year={2020}
}
@InProceedings{peng2023hierarchical,
  title={Hierarchical Dense Correlation Distillation for Few-Shot Segmentation},
  author={Peng, Bohao and Tian, Zhuotao and Wu, Xiaoyang and Wang, Chenyao and Liu, Shu and Su, Jingyong and Jia, Jiaya},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}

@InProceedings{tian2022gfsseg,
    title={Generalized Few-shot Semantic Segmentation},
    author={Zhuotao Tian and Xin Lai and Li Jiang and Shu Liu and Michelle Shu and Hengshuang Zhao and Jiaya Jia},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}

```
