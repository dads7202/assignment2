# DADS7202 Assignment 2 (Group Moo-ka-ta)

The objective of this project is `multi-class image classification`of `Fine-Art painting` (5 categories) with `CNN models`. <br>
To build the best model that gives the highest accuracy for this task, we experiment with pre-trained models including VGG16, NASNetMobile, and DenseNet121 
and compare the results between the original pre-trained and the model after fine-tuning.

## 🌟 Highlight

## Table of Contents
1. [Introduction](#1.-introduction) <br>
2. [Data Preparation](#2.-data-preparation) <br>
3. Model<br>
   - [3.1 VGG16](#3.1-vgg16)<br>
   - [3.2 NASNetMobile](#3.2-nasnetmobile) <br>
   - [3.3 DenseNet121](#3.3-densenet121) <br>
4. [Prediction](#4.-prediction) <br>
5. [Result](#5.-result) <br>
6. [Discussion](#6.-discussion) <br>
7. [Conclusion](#7.-conclusion) <br>
8. [Reference](#8.-reference) <br>
9. [Citing](#9.-citing) <br>
10. [Member, Contribution and Responsibility](#10.-member,-contribution-and-responsibility) <br>
11. [End Credit](#11.-end-credit) <br>

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

Dataset in this assignment are `fine-art painting images` which classified into `5 categories`. <br>
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

## 3.2 NASNetMobile

## 3.3 DenseNet16

## 4. Prediction

## 5. Result

## 6. Discussion

## 7. Conclusion

## 8. Reference

## 9. Citing

## 10. Member, Contribution and Responsibility

## 11. End Credit
