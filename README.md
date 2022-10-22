# DADS7202 Assignment 2 (Group Moo-ka-ta)
![0_Head](https://i.imgur.com/weolMpc.png)

## 🌟 Highlight
- The accuracy of all fine-tuned CNN models are better than the imageNet dataset. Our datasets do not exist in the original pretrained model.
- The best model that gives the best accuracy for this project is the fine-tuning Densenet pre-tained model (Densenet-Model 1).
- Adding more layers to the model will increase accuracy and reduce loss.
- However, it depends on the complexity of the problem. Too many layers can cause overfitting of the network. It performs best on the training data, but it won't be able to generalize to new unseen data.

## 📍 Table of Contents
1. [Introduction](#1-introduction) <br>
2. [Data Preparation](#2-data-preparation) <br>
3. [Model](#3-model)<br>
   - [3.1 VGG16](#31-vgg16)<br>
   - [3.2 NASNetMobile](#32-nasnetmobile) <br>
   - [3.3 DenseNet121](#33-densenet121) <br>
4. [Result](#4-result) <br>
5. [Discussion](#5-discussion) <br>
6. [Conclusion](#6-conclusion) <br>
7. [Reference](#7-reference) <br>
8. [Citing](#8-citing) <br>
9. [Members, Contribution and Responsibility](#9-members,-contribution-and-responsibility) <br>
10. [End Credit](#10-end-credit) <br>

## 1. Introduction
This project aims to build the best deep learning model that gives the highest accuracy for image classification task. We trained 3 different pre-trained models including VGG16, NASNetMobile, and DenseNet121 on a digitized painting dataset to compare their art classification performances between the original pre-trained models and fine-tuned models. 

Hierarchy of Painting Genre, the concept of categorizing Western paintings into 5 “genre” or “category” was proposed in 1669 by Andre Felibien the secretary of the French Academie des beaux-des Beaux-Arts (Academy of Fine Arts). It is a system that ranks paintings in terms of its cultural value. The five genres listed in order of their official ranking or importance, are as follows:  
- History Painting 
- Portrait Art 
- Genre Painting 
- Landscape Painting 
- Still Life Painting <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 2. Data Preparation

**Data source:** Our dataset consists of paintings downloaded from publicly available sources such as [WikiArt](https://www.wikiart.org/), [The Met](https://www.metmuseum.org/), [My Art Magazine](https://myartmagazine.com/), [Colossal](https://www.thisiscolossal.com/). For more information about the dataset, please refer to [the excel](https://github.com/dads7202/assignment2/blob/main/fileReference/Reference.xlsx).
**Column in this excel** <br>
- Type: type of paintings such as genre, stillife, portrait, landscape and history <br>
- image_name: type of painting_index.type file such as genre_001.jpg <br>
- url: the reference in each image. <br>

The dataset in this assignment are `fine-art painting images` which are classified into `5 categories`. <br>
Total images are `1,106 images`, including,
1. Genre Painting 223 images
2. History Painting 210 images
3. Landscape Painting 210 images
4. Portrait Painting 235 images
5. Still Life Painting 228 images

**Data preparation and pre-processing:** <br>
To get data ready for model: <br>
- We apply `augmentations`, which are techniques used to increase the amount of data by adding slightly modified copies of already existing data.  
- The following data augmentations are applicable on train data: 
   - 1. `rescale`: rescaling factor (1./255) 
   - 2. `shear_range`: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees). (0.2) 
   - 3. `zoom_range`: Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]. (0.2) 
   - 4. `horizontal_flip`: Boolean. Randomly flip inputs horizontally. (True) 
   - 5. `vertical_flip`: Boolean. Randomly flip inputs vertically. (True) 

**Data splitting (train/val/test):** <br>
`validation_split: 0.1`
<br>
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 3. Model
We use `ImageNet` as the pre-trained weights on model, `Adam` as optimizer, categorical `crossentropy` as loss function for multi-class classification task, and `224 x 224` as input size. 
## 3.1 VGG16
Before fine-tuning the layers of pre-trained model, we attempt to optimize variables such as batch size, and epoch for original pre-trained vgg16 model. 
### 3.1.1 Batch Size
we experiment with batch sizes of `16`, `32`, and `64` to find out which batch size results in the `highest accuracy` and `lowest loss` for test set at epoch 25, original pre-trained model. <br>
![Imgur](https://i.imgur.com/vhmiuNt.png)
![Imgur](https://i.imgur.com/RtNbLsn.png)
### 3.1.2 Epoch
![Imgur](https://i.imgur.com/hgeZIDr.png)
![Imgur](https://i.imgur.com/CIP04V1.png)
### 3.1.3 Fine-tuning pre-trained VGG16 model
![Imgur](https://i.imgur.com/f5CClNA.png)

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 3.2 NASNetMobile
### 3.2.1 Batch size
![Imgur](https://i.imgur.com/tAFO8Vo.png)
### 3.2.2 Fine-tuning pre-trained NASNetMobile model
![Imgur](https://i.imgur.com/O0TSWyY.png)


[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 3.3 DenseNet121
We utilized ImageNet as the pre-trained weights on model, Adam as optimizer, categorical crossentropy as loss function for multi-class classification task, and 224 x 224 as input size.
### 3.3.1 Epoch
![Imgur](https://i.imgur.com/xGab4E9.png) <br>
We trained data on 200 epochs to find an appropriate number of epoch. We found that train loss does not overfit and gradually decreases. Otherwise, in the accuracy graph around the 150th epoch, the accuracy of the validation set is less than the training data set. As a result, we decided to train the model for 150 epochs with a batch size of 32.
### 3.3.2 Fine-tuning pre-trained DenseNet121 model
![Imgur](https://i.imgur.com/ynlWDPk.png)
We fine-tuned the DenseNet121 pre-trained model by adding dense layers and dropout layers, as shown in figure, with a batch size of 32 and 150 epochs. We found that
- Model 1 gave the best accuracy at 83.54%, meanwhile Model 3 given the best loss at 0.6226 on testing set
- The dropout 0.2 in Model 1 is too little and causes overfit, but it gives the best accuracy among the 4 models. Meanwhile, the dropout rate in model 2 is 0.5. It does not overfit, but the accuracy is not better than the original base model.

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 4. Result
### 4.1 Batch size 
![Imgur](https://i.imgur.com/QBhOWz5.png)
### 4.2 Model1: VGG16
![Imgur](https://i.imgur.com/nWTW8mn.png)
### 4.3 Model2: NASNetMobile
![Imgur](https://i.imgur.com/KfXiwp1.png)
### 4.4 Model3: DenseNet121
![Imgur](https://i.imgur.com/KdDIBpd.png)
### 4.5 Compare the best performance of each model
![Imgur](https://i.imgur.com/Xe6Jap7.png)
### 4.6 Inference on the best accuracy of fine-tuning the layers of DenseNet121(model1)

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 5. Discussion

![Imgur](https://i.imgur.com/5XGwrZA.png)

![Imgur](https://i.imgur.com/yU79ef6.png)

![Imgur](https://i.imgur.com/YjSRso1.png)


[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 6. Conclusion
In this project aims to build the best CNN model with pre-trained models that gives the highest accuracy for multi-class image classification of 5 types of painting. We experiment with original (ImageNet) and pre-trained models including VGG16, NASNetMobile, and DenseNet121 and compare the results between the original pre-trained and models after fine-tuning. 

Experimental results show that  
- The performance models of all fine-tuning CNN are better than the imageNet on our dataset. Our datasets do not exist in the original pretrained model. 
- The performance models of fine-tuning CNN might improve the accuracy or might not, it really depends on the complexity of the problem. 
- The model with the most accuracy on test dataset is DenseNet121 (model 1 the fine-tuning CNN). <br>
 
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 7. Reference
Google Drive for weight file: https://drive.google.com/drive/folders/1tMoJg7qz9SUWL8Vyt67vVnleLwDnme6V?usp=sharing <br>
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 8. Citing
```  
    @inproceedings{plummerCITE2018, 
	Author = {Nidchapan N., Prapatsorn T., Chotika B., Juthamas P., Naliya M.}, 
	Title = {CNN classification: The Academic Hierarchy of the Genres}, 
	Year = {2022}, 
  	howpublished = "\url{https://github.com/dads7202/assignment2}" 
    } 
```  
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 9. Member, Contribution and Responsibility

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 10. End Credit
This project is a part of subject DADS7202. Data Analytics and Data Science. NIDA


