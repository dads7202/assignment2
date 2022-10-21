# DADS7202 Assignment 2 (Group Moo-ka-ta)

The objective of this project is `multi-class image classification`of `Fine-Art painting` (5 categories) with `CNN models`. <br>
To build the best model that gives the highest accuracy for this task, we experiment with pre-trained models including VGG16, NASNetMobile, and DenseNet121 
and compare the results between the original pre-trained and the model after fine-tuning.

## 🌟 Highlight

## Table of Contents
1. [Introduction](##1.-introduction)<br>
2. [Data](##2.-data)<br>
3. Network Architechture and Training <br>
   - [3.1 VGG16](##3.1-vgg16)<br>
   - [3.2 NASNetMobile](##3.2-nasnetmobile)<br>
   - [3.3 DenseNet121](##3.3-densenet121)<br>
4. [Result](##4.-Result)<br>
5. [Discussion](##5.-discussion)<br>
6. [Conclusion](##6.-conclusion)<br>
7. [Reference](##7.-reference)


## 1. Introduction
**The 5 Painting Genres**
Traditional Classification of Paintings

> What are Genres?

Paintings are traditionally divided into **five categories or 'genres'**. The establishment of these genres and their relative status in relation to one other, stems from the philosophy of arts promoted by the great European Academies of Fine Art, like the Royal Academy in London, and the influential French Academy of Fine Arts (Academie des Beaux-Arts).

The five categories of fine art painting, listed in **order of their official ranking or importance**, are as follows:

1. **History Painting**  Traditionally the most-respected of all the genres, refers to paintings showing the exemplary deeds and struggles of moral figures. It includes Christian imagery involving Biblical figures, as well as mythological painting involving mythical or pagan divinities, and real-life historical figures. History paintings - traditionally large-scale public works - aim to elevate the morals of the community. **Famous historical painting artists** include Sir David Wilkie, Paul Delaroche, Eugène Delacroix

2. **Portrait Art**  Includes pictures of people, deities or mythological figures in human form. The genre includes group-portraits as well as those of individuals. A portrait of an individual may be face-only, or head and shoulders, or full-body.**Famous portrait artists** include Frida Kahlo, Rembrandt, Leonardo da Vinci

3. **Genre Painting**  or "genre-scenes" refers to pictures that portray ordinary scenes of everyday life. Subjects include domestic settings, interiors, celebrations, tavern scenes, markets and other street situations. Whatever the precise content, the scene is typically portrayed in a non-idealized way, and characters are not endowed with any heroic or dramatic attributes. **Famous genre painting artists** include Jan Vermeer, Edward Hopper, Edgar Degas

4. **Landscape Painting**  Any picture whose main subject is the depiction of a scenic view, such as fields, hillscapes, mountain-scapes, trees, riverscapes, forests, sea views and seascapes. Many famous landscape paintings include human figures, but their presence should be a secondary element in the composition. **Famous landscape artists** include Vincent Van Gogh, Claude Monet, J.M.W. Turner.

5. **Still Life Painting**  Typically comprises an arrangement of objects (such as flowers or any group of mundane objects) laid out on a table. A form of still life painting that contains biblical or moral messages. **Famous still life artists** include Paul Cezanne, Salvador Dali, Gorges Braque


## 2. Data

Data in this assignment are painting images which classified into 5 categories.
Total images are 1,106 images, including,
1. Genre Painting 223 images
2. History Painting 210 images
3. Landscape Painting 210 images
4. Portrait Painting 235 images
5. Still Life Painting 228 images

- Data Collection <br>
We collected images from various sources most of photos are public domain painting. <br>
You can see the reference of each photo in [datasource]( https://github.com/dads7202/assignment2/blob/main/fileReference/Reference.xlsx).

- Data preparation <br>
   - 1. Get the Image Dataset Paths and Load Image Datasets from path.
   - 2. Apply Augmentations, for traing dataset we have augmented images by rescale image to 1/255 from original image, shear image by 0.2 degrees, random zoom image, do both horizontal and wertical flip randomly. 
   - 3. Split data into training_set and testing_set with testing set = 0.1 of total images.


## 3.1 VGG16

## 3.2 NASNetMobile

## 3.3 DenseNet16

## 4. Result

## 5. Discussion

## 6. Conclusion

## 7. Reference
