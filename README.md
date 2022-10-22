# DADS7202 Assignment 2 (Group Moo-ka-ta)
![0_Head](https://i.imgur.com/weolMpc.png)

## 🌟 Highlight
-   All fine-tuning pre-trained CNN models are better than the original pre-trained models which trained on the ImageNet dataset.
-   The best model which gives the highest accuracy of test set for this project is the fine-tuning DenseNet121 pre-tained model (DenseNet121-Model 1).
-   Adding layers improves performance which increases accuracy and reduces loss of test set.
-   However, adding too many layers leads to overfitting, which performs well on training data but not on unseen data.

## Table of Contents
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
9. [Member, Contribution and Responsibility](#9-member,-contribution-and-responsibility) <br>
10. [End Credit](#10-end-credit) <br>

## 1. Introduction
This project aims to build the best deep learning model that gives the highest accuracy for image classification task. We trained 3 different pre-trained models including VGG16, NASNetMobile, and DenseNet121 on a digitized painting dataset to compare their art classification performances between the original pre-trained models and fine-tuned models. 
`Our datasets do not exist on ImageNet dataset`

Hierarchy of Painting Genre, the concept of categorizing Western paintings into 5 “genre” or “category” was proposed in 1669 by Andre Felibien the secretary of the French Academie des beaux-des Beaux-Arts (Academy of Fine Arts). It is a system that ranks paintings in terms of its cultural value. The five genres listed in order of their official ranking or importance, are as follows:  
- History Painting 
- Portrait Art 
- Genre Painting 
- Landscape Painting 
- Still Life Painting <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 2. Data Preparation
GPU Name: We use NVIDIA TESLA P100 as GPU for this project.
**Data source:** Our dataset consists of paintings downloaded from publicly available sources such as [WikiArt](https://www.wikiart.org/), [The Met](https://www.metmuseum.org/), [My Art Magazine](https://myartmagazine.com/), [Colossal](https://www.thisiscolossal.com/). For more information about the dataset, please refer to [the excel](https://github.com/dads7202/assignment2/blob/main/fileReference/Reference.xlsx).
**Columns description** <br>
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
We use pre-trained models which trained on `ImageNet` dataset, Adam as optimizer, `categorical crossentropy` as loss function for multi-class classification task, and `224 x 224` as input size. 

### 3.1 VGG16
Before fine-tuning the layers of pre-trained model, we attempt to optimize variables such as batch size, and epoch for original pre-trained vgg16 model. 
### 3.1.1 Batch Size
we experiment with batch sizes of `16`, `32`, and `64` to find out which batch size results in the `highest accuracy` and `lowest loss` for test set at epoch 25, original pre-trained model. <br>
![Imgur](https://i.imgur.com/YEtt4Fj.png) <br>
- the results of batch sizes of 16, 32, and 64 trends are nearly equal in terms of validation accuracy and loss. At final epoch, the greatest batch size is 32, which gives the highest validation accuracy and lowest validation loss.
- we utilized batch size of 32 as default value for this task.

### 3.1.2 Epoch
Overfitting occurs when a model performs well on training data but poorly on validation or unknown data. To avoid overfitting, we experimented with 200 epochs to observe the trend of curve loss at batch sizes of 32, original pre-trained model.
![Imgur](https://i.imgur.com/fYARtwO.png) <br>
- From the graphs presented, validation loss begins to exceed training loss indicating that overfitting occurs when the epoch exceeds 100. The ultimate epoch for getting the highest accuracy and the lowest validation loss for the original pre-trained model is 90.
- We chose the epoch of 90 to compare the results of the original pre-trained model and the fine-tuned models

### 3.1.3 Fine-tuning pre-trained VGG16 model
![Imgur](https://i.imgur.com/d43Rcus.png) <br>
We attempted to fine-tune the VGG16 pre-trained model by adding dense layers and droupout layers, as shown in the figure, with a batch size of 32 and 90 epochs. We concluded that, <br>
- Adding layers improved performance which increased accuracy and reduced loss on test set, in conclusion, fine-tuning model performed better than the original pre-trained model.
- The best model is Model 3, accuracy from the test set is 80.73 ± 0.09% and loss from test set is 0.57 ± 0.02%. <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 3.2 NASNetMobile
We used ImageNet as the pre-trained weights on model and there are the hyperparameters listed below.
- input_shape: [None, 224, 224, 3]
- weights: [imagenet]
- Activation function in Output layer: [softmax]
- Loss function: [categorical_crossentropy]
- Optimizer: [Adam]
- Batch: [32]
- Epoch: [200]

### 3.2.1 Batch size
We experimented with original model (ImageNet) by validate epoch between 0 and 200. <br>
![Imgur](https://i.imgur.com/tAFO8Vo.png)
Experimental results showed the NASNetMobile did not performed well and ended up overfitting after only approximately 7 epochs. We supposed that which this epoch NASNetMobile probably isn't suitable for unseen data (perform well on training set but not on unseen data) that lead to overfits. First, we selected 90 epochs before attempting to fine-tune the model because it's a steady state of accuracy as shown in the graphs. The accuracy was not different between 90 and 200 epochs. Next, we attempted to find the best conditions for this model. 

### 3.2.2 Fine-tuning pre-trained NASNetMobile model
![Imgur](https://i.imgur.com/O0TSWyY.png) <br> 
We attempted to fine-tune the NASNetMobile pre-trained model by adding dense layers and dropout layers, as shown in figure, with a batch size of 32 and 90 epochs. We found that
- As we find ways to improve model accuracy by increasing the number of hidden layers, we found that the accuracy depends on the complexity of the problem. Thus, the accuracy increased from Model 1 (accuracy from test sets 78.47 ± 1.30%) to Model 2 (accuracy from test sets 81.25 ± 2.25%) but decreased from Model 2 to Model 3 (accuracy from test sets 79.51 ± 1.96%). 
- From Model 2 to Model 3, we increased 2 dense layers and a dropout layer but doing so lowered the accuracy on the test set. We suspected that the number of layers in model 2 is already sufficient. Too many layers can cause overfitting to the network. It performs best on the training data, but it won't be able to generalize to new unseen data.
- The best model is Model 2, accuracy from test set is 81.25 ± 2.25% and loss from test set is 29.10 ± 3.80%. <br>

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)

## 3.3 DenseNet121
We utilized ImageNet as the pre-trained weights on model, Adam as optimizer, categorical crossentropy as loss function for multi-class classification task, and 224 x 224 as input size. <br>

### 3.3.1 Epoch
![Imgur](https://i.imgur.com/NXvokFy.png) <br>
We trained data on 200 epochs to find an appropriate number of epoch. We found that train loss does not overfit and gradually decreases. Otherwise, in the accuracy graph around the 150th epoch, the accuracy of the validation set is less than the training data set. As a result, we decided to train the model for 150 epochs with a batch size of 32.

### 3.3.2 Fine-tuning pre-trained DenseNet121 model
![Imgur](https://i.imgur.com/YiDmIT5.png) <br>
We fine-tuned the DenseNet121 pre-trained model by adding dense layers and dropout layers, as shown in figure, with a batch size of 32 and 150 epochs. We found that
- Model 1 gave the best accuracy at 83.54%, meanwhile Model 3 given the best loss at 0.6226 on testing set.
- The dropout 0.2 in Model 1 is too little and causes overfit, but it gives the best accuracy among the 4 models on test set. Meanwhile, the dropout rate in model 2 is 0.5. It does not overfit, but the accuracy is not better than the original base model. 

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 4. Result

#### 4.1 Batch size
Firstly, we set the hypothesis that on this dataset if we increase batch size, the accuracy should be improved. Then we examined batch sizes of 16, 32, and 64 respectively to establish which batch size produces the highest accuracy and lowest loss for the test set at epoch 25 and the original pre-trained model.

![Imgur](https://i.imgur.com/mUk3k3Q.png) <br>

We found that batch size 32 produced the most accurate and the least loss on test dataset. So that we decided to set the batch size to 32 as the default value for this task.

#### 4.2 Model 1: VGG16

![Imgur](https://i.imgur.com/PReFdld.png) <br>
Table 4.1 comparing performances of original pre-trained VGG16 and three fine-tuned VGG16 models

As you can see from table 4.1, all three fine-tuned VGG16 models have more accuracy on the test dataset than the original pre-trained VGG16 model. `Model 3` is the most accurate of the VGG16 models, with an accuracy of 80.73 ± 0.01% on the test dataset, with batch size of 32 and 90 epochs, and the shortest mean time to train (8.49 ± 0.12 seconds on GPU). Moreover, Model 3 has the least loss from the test dataset (57.90 ± 0.02%).

#### 4.3 Model 2: NASNetMobile

![Imgur](https://i.imgur.com/GxBkEbW.png) <br>
Table 4.2 comparing performances of original pre-trained NASNetMobile and three fine-tuned NASNetMobile models

As you can see from table 4.2, all three fine-tuned NASNetMobile models produced more accuracy on the test dataset than the original pre-trained NASNetMobile model. `Model 2` is the most accurate model compared to all the NASNetMobile models, with an accuracy of 81.25% on the test dataset, with batch size of 32 and 90 epochs, and it takes the least time to train (12.79 ± 1.93 seconds on GPU). However, Model 1 has the least loss on the test dataset (26.98%).

#### 4.4 Model 3: DenseNet121

![Imgur](https://i.imgur.com/TvFKAg4.png) <br> 
Table 4.3 comparing performances of original pre-trained DenseNet121 and three fine-tuned DenseNet121 models

As you can see from table 4.3, of the three fine-tuned DenseNet121 models, Model 2, has less accuracy on the test dataset than the original pre-trained DenseNet121 model (with accuracy of 73.13 ± 0.02%). **Model 1** is the most accurate model compared with all DenseNet121 models, with an accuracy of 83.54 ± 0.02% on the test dataset, with batch size of 32 and 150 epochs. However, model 3 has the least loss on the test dataset with 62.26 ± 0.06%.

#### 4.5 Compare the best performance of each model

![Imgur](https://i.imgur.com/qr5UtbF.png) <br>
Table 4.4 comparing performance of each best fine-tuned model

Comparing all the results from the best-performed model of each backbone, we found that,
1. VGG16 took the least mean time for each epoch in the training model, with a mean time of 8.49 ± 0.11 seconds per epoch.
2. NASNetMobile has the lowest loss accuracy on the test dataset, with a loss accuracy of 0.29 ± 0.03% on the test dataset.
3. DenseNet121 is the most accurate model, with an accuracy of 83.54 ± 0.02% on the test dataset.

#### 4.6 Inference on the best accuracy of fine-tuning the layers of DenseNet121(model1).
![Imgur](https://i.imgur.com/5HggmT2.png) <br>
Figure shows the prediction on testing set using DenseNet121 ( model 3 ). <br>
[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 5. Discussion
<p> We chose VGG16 as our first pre-trained model since, from previous published works, it tends to perform well on general image classifications problems. We experimented with 16, 32 and 64 batch size and found that batch size 32 gave the highest validation accuracy with lowest validation loss. We then utilized this number of batch size as default throughout the experimentation. </p>

![Imgur](https://i.imgur.com/vhmiuNt.png)
![Imgur](https://i.imgur.com/RtNbLsn.png)

<p> This finding aligned with what has been observed in practice that large-batch methods tend to converge to sharp minimizers of the training and testing functions and leads to poorer generalization. <br>
We hand-picked 3 pre-trained models that performed best in the preliminary testing: VGG16, NASNetMobile, and DenseNet121. Then we compared performances of the original 3 models with 3 other fine-tuned models each. Fine-tuning were done by (1) adjusting epochs and (2) adding layers to the models. </p>

#### The effect of epoch on the model performance.
<p> We observed that out of the 3 models at batch size 32, NASNetMobile showed overfitting tendencies at only after 7th epoch. There were no significant differences observed in the model performances at later epoch. We picked epoch 90 for NASNetMobile before adding layers to the model. <br>
VGG16 and DenseNet121 showed signs of overfitting at 100th and 150th epoch respectively. </p>

![Imgur](https://i.imgur.com/tS0mtBw.png) <br>

<p> We found that for this specific dataset, the optimal number of epochs ranges widely among the selected pre-trained model, but the desirable loss and accuracy in both the training set and test set is observed at around 90 epochs. </p>

#### The effect of hidden layers on the model performance.
<p> In all 3 models we used Rectified Linear Unit (ReLU) activation function in the dense layers because it is well-studied that ReLU outperformed other activation functions, such as Sigmoid and Hyperbolic tangent. </p>

![Imgur](https://i.imgur.com/fgGe0Ow.png)

<p> The best performers of each model have 3-5 dense layers. We observed that the more dense layer added to the model, the less ability of the model to generalized. <br>
Adding dropout layers and increasing dropout rates to the model help with overfitting issue as best observed in NASNetMobile Model 2. <br>
The results after fine-tuning compared to the original pretrained models, correspondingly, demonstrated that fine-tuned models outperformed the original models in the prediction of painting genre. </p>



[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 6. Conclusion
In this project aims to build the best CNN model with pre-trained models that gives the highest accuracy for multi-class image classification of 5 types of painting. We experiment with original (ImageNet) and pre-trained models including VGG16, NASNetMobile, and DenseNet121 and compare the results between the original pre-trained and models after fine-tuning. 

Experimental results show that  
- The performance models of almost fine-tuning CNN are better than original pre-trained models which trained on the imageNet dataset (Our datasets doesn't exist on ImageNet dataset).
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
1. 6310412022 (20%) Prepare dataset, review & VGG16 and result 
2. 6410422007 (20%) Prepare dataset, review & data collection and result 
3. 6410422016 (20%) Prepare dataset, review & NASNetMobile, conclusion 
4. 6410422022 (20%) Prepare dataset, review & DenseNet121 and result 
5. 6410422030 (20%) Prepare dataset, review & introduction and discussion 

[![back-to-top](https://i.imgur.com/wJEM2Vt.png)](#table-of-contents)


## 10. End Credit
This project is a part of subject DADS7202. Data Analytics and Data Science. NIDA
