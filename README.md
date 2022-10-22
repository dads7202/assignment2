# DADS7202 Assignment 2 (Group Moo-ka-ta)
![0_Head](https://i.imgur.com/weolMpc.png)

## 🌟 Highlight

## Table of Contents
1. [Introduction](#1-introduction) <br>
2. [Data Preparation](#2-data-preparation) <br>
3. [Model](#3-model)<br>
   - [3.1 VGG16](#31-vgg16)<br>
   - [3.2 NASNetMobile](#32-nasnetmobile) <br>
   - [3.3 DenseNet121](#33-densenet121) <br>
4. [Prediction](#4-prediction) <br>
5. [Result](#5-result) <br>
6. [Discussion](#6-discussion) <br>
7. [Conclusion](#7-conclusion) <br>
8. [Reference](#8-reference) <br>
9. [Citing](#9-citing) <br>
10. [Member, Contribution and Responsibility](#10-member,-contribution-and-responsibility) <br>
11. [End Credit](#11-end-credit) <br>

## 1. Introduction
This project aims to build the best model that gives the highest accuracy for this task by training models that can predict the 5 "genres" or "categories" of paintings. We experiment with pre-trained models including VGG16, NASNetMobile, and DenseNet121 and compare the results between the original pre-trained and the model after fine-tuning. The five categories of fine art painting are as follows:  
- History Painting 
- Portrait Art 
- Genre Painting 
- Landscape Painting 
- Still Life Painting 

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


## 3.2 NASNetMobile
### 3.2.1 Batch size
![Imgur](https://i.imgur.com/tAFO8Vo.png)
### 3.2.2 Fine-tuning pre-trained NASNetMobile model
![Imgur](https://i.imgur.com/O0TSWyY.png)

## 3.3 DenseNet16
###

## 4. Prediction

## 5. Result

## 6. Discussion

## 7. Conclusion
In this project aims to build the best CNN model with pre-trained models that gives the highest accuracy for multi-class image classification of 5 types of painting. We experiment with original (ImageNet) and pre-trained models including VGG16, NASNetMobile, and DenseNet121 and compare the results between the original pre-trained and models after fine-tuning. 

Experimental results show that  
- The performance models of all fine-tuning CNN are better than the imageNet on our dataset. Our datasets do not exist in the original pretrained model. 
- The performance models of fine-tuning CNN might improve the accuracy or might not, it really depends on the complexity of the problem. 
- The model with the most accuracy on test dataset is DenseNet121 (model 1 the fine-tuning CNN). 

## 8. Reference
Google Drive for weight file: https://drive.google.com/drive/folders/1tMoJg7qz9SUWL8Vyt67vVnleLwDnme6V?usp=sharing 

## 9. Citing
```  
    @inproceedings{plummerCITE2018, 
	Author = {Nidchapan N., Prapatsorn T., Chotika B., Juthamas P., Naliya M.}, 
	Title = {CNN classification: The Academic Hierarchy of the Genres}, 
	Year = {2022}, 
  	howpublished = "\url{https://github.com/dads7202/assignment2}" 
    } 
```  

## 10. Member, Contribution and Responsibility




## 11. End Credit
This project is a part of subject DADS7202. Data Analytics and Data Science. NIDA


