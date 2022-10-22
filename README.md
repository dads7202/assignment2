# DADS7202 Assignment 2 (Group Moo-ka-ta)
![0_Head](https://i.imgur.com/weolMpc.png)

## 🌟 Highlight
- The accuracy of all fine-tuned CNN models are better than the imageNet dataset. Our datasets do not exist in the original pretrained model.
- The best model that gives the best accuracy for this project is the fine-tuning Densenet pre-tained model (Densenet-Model 1).
- Adding more layers to the model will increase accuracy and reduce loss.
- However, it depends on the complexity of the problem. Too many layers can cause overfitting of the network. It performs best on the training data, but it won't be able to generalize to new unseen data.

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

## 💬 1. Introduction
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

**Data splitting (train/val/test):** `validation_split: 0.1`
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
- the results of batch sizes of 16, 32, and 64 trends are nearly equal in terms of validation accuracy and loss. At final epoch, the greatest batch size is 32, which gives the highest validation accuracy and lowest validation loss
- we utilized batch size of 32 as default value for this task.

### 3.1.2 Epoch
Overfitting occurs when a model performs well on training data but poorly on validation or unknown data. To avoid overfitting, we experimented with 200 epochs to observe the trend of curve loss at batch sizes of 32, original pre-trained model.
![Imgur](https://i.imgur.com/hgeZIDr.png)
![Imgur](https://i.imgur.com/CIP04V1.png) <br>
- From the graphs presented, validation loss begins to exceed training loss indicating that overfitting occurs when the epoch exceeds 100. The ultimate epoch for getting the highest accuracy and the lowest validation loss for the original pre-trained model is 90.
- We chose the epoch of 90 to compare the results of the original pre-trained model and the fine-tuned models

### 3.1.3 Fine-tuning pre-trained VGG16 model
![Imgur](https://i.imgur.com/f5CClNA.png) <br>
We attempted to fine-tune the VGG16 pre-trained model by adding dense layers and droupout layers, as shown in the figure, with a batch size of 32 and 90 epochs. We concluded that, <br>
- Adding layers improved performance which increased accuracy and reduced loss on test set, in conclusion, fine-tuning model performed better than the original pre-trained model.
- The best model is Model 3, accuracy from the test set is 80.73 ± 0.09% and loss from test set is 0.57 ± 0.02% <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 3.2 NASNetMobile
We used ImageNet as the pre-trained weights on model Hyperparameter that use base mode by the Imagenet
- input_shape: [None, 224, 224, 3]
- weights: [imagenet]
- Activation function in Output layer: [softmax]
- Loss function: [binary_crossentropy]
- Optimizer: [Adam]
- Batch: [32]
- Epoch: [200]

### 3.2.1 Batch size
We experimented with original model (ImageNet) by validate epoch between 0 and 200. <br>
![Imgur](https://i.imgur.com/tAFO8Vo.png)
Experimental results showed the NASNetMobile did not performed well and ended up overfitting after only approximately 7 epochs. We supposed NASNetMobile probably isn't suitable for this dataset. First, we selected 90 epochs before attempting to fine-tune the model because it's a steady state of accuracy as shown in the graphs. The accuracy was not different between 90 and 200 epochs. Next, we attempted to find the best conditions for this model.

### 3.2.2 Fine-tuning pre-trained NASNetMobile model
![Imgur](https://i.imgur.com/O0TSWyY.png)
We attempted to fine-tune the NASNetMobile pre-trained model by adding dense layers and dropout layers, as shown in figure, with a batch size of 32 and 90 epochs. We found that
- As we find ways to improve model accuracy by increasing the number of hidden layers, we found that the accuracy depends on the complexity of the problem. Thus, the accuracy increased from Model 1 (accuracy from test sets 78.47 ± 1.30%) to Model 2 (accuracy from test sets 81.25 ± 2.25%) but decreased from Model 2 to Model 3 (accuracy from test sets 79.51 ± 1.96%). 
- From Model 2 to Model 3, we increased 2 dense layers and a dropout layer but doing so lowered the accuracy on the test set. We suspected that the number of layers in model 2 is already sufficient. Too many layers can cause overfitting to the network. It performs best on the training data, but it won't be able to generalize to new unseen data.
- The best model is Model 2, accuracy from test set is 81.25 ± 2.25% and loss from test set is 29.10 ± 3.80% <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 3.3 DenseNet16
###


[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 4. Prediction


[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 5. Result


[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 6. Discussion
<p> We chose VGG16 as our first pre-trained model since, from previous published works, it tends to perform well on general image classifications problems. We experimented with 16, 32 and 64 batch size and found that batch size 32 gave the highest validation accuracy with lowest validation loss. We then utilized this number of batch size as default throughout the experimentation. </p>

![Imgur](https://i.imgur.com/vhmiuNt.png)
![Imgur](https://i.imgur.com/RtNbLsn.png)

<p> This finding aligned with what has been observed in practice that large-batch methods tend to converge to sharp minimizers of the training and testing functions and leads to poorer generalization. <br>
We hand-picked 3 pre-trained models that performed best in the preliminary testing: VGG16, NASNetMobile, and DenseNet121 with 4 other pre-trained models being Xception, ResNET50, NASNetLarge, and EfficientNet. Then we compared performances of the original 3 models with 3 other fine-tuned models each. Fine-tuning was done by (1) adjusting epochs and (2) adding layers to the models. </p>

#### The effect of epoch on the model performance.
<p> We observed that out of the 3 models at batch size 32, NASNetMobile showed overfitting tendencies at only after 7th epoch. There were no significant differences observed in the model performances at later epoch. We picked epoch 90 for NASNetMobile before adding layers to the model. <br>
VGG16 and DenseNet121 showed signs of overfitting at 100th and 150th epoch respectively. </p>

![Imgur](https://i.imgur.com/tS0mtBw.png) <br>

<p> We found that for this specific dataset, the optimal number of epochs ranges widely among the selected pre-trained model, but the desirable loss and accuracy in both the training set and test set is observed at around 90 epochs. </p>

#### The effect of hidden layers on the model performance.
<p> In all 3 models we used Rectified Linear Unit (ReLU) activation function in the dense layers because it is well-studied that ReLU outperformed other activation functions, such as Sigmoid and Hyperbolic tangent. </p>

![Imgur](https://i.imgur.com/QqCVB0z.png) <br>

<p> The best performers of each model have 3-5 dense layers. We observed that the more dense layer added to the model, the less ability of the model to generalized. <br>
Adding dropout layers and increasing dropout rates to the model help with overfitting issue as best observed in NASNetMobile Model 2. <br>
The results after fine-tuning compared to the original pretrained models, correspondingly, demonstrated that fine-tuned models outperformed the original models in the prediction of painting genre. </p>



[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 7. Conclusion
In this project aims to build the best CNN model with pre-trained models that gives the highest accuracy for multi-class image classification of 5 types of painting. We experiment with original (ImageNet) and pre-trained models including VGG16, NASNetMobile, and DenseNet121 and compare the results between the original pre-trained and models after fine-tuning. 

Experimental results show that  
- The performance models of all fine-tuning CNN are better than the imageNet on our dataset. Our datasets do not exist in the original pretrained model. 
- The performance models of fine-tuning CNN might improve the accuracy or might not, it really depends on the complexity of the problem. 
- The model with the most accuracy on test dataset is DenseNet121 (model 1 the fine-tuning CNN). <br>
 
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 8. Reference
Google Drive for weight file: https://drive.google.com/drive/folders/1tMoJg7qz9SUWL8Vyt67vVnleLwDnme6V?usp=sharing <br>
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 9. Citing
```  
    @inproceedings{plummerCITE2018, 
	Author = {Nidchapan N., Prapatsorn T., Chotika B., Juthamas P., Naliya M.}, 
	Title = {CNN classification: The Academic Hierarchy of the Genres}, 
	Year = {2022}, 
  	howpublished = "\url{https://github.com/dads7202/assignment2}" 
    } 
```  
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 10. Member, Contribution and Responsibility

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 11. End Credit
This project is a part of subject DADS7202. Data Analytics and Data Science. NIDA


