# Bachelor Thesis: Facial Emotion Recognition with Convolutional Neural Networks
## Abstract
Deep Learning and Neuromarketing, two disciplines representing the state of the art in their corresponding fields of knowledge, act together for the first time to solve a problem that has been addressed multiple times. To understand what the consumers really want. By using advanced artificial intelligence techniques, such as convolutional neural networks, this work tries (and succeeds) to detect emotions through the analysis of a person’s facial expressions. With the outcome of said analysis, it is possible to predict the effectiveness of ads depending on which emotions advertisers want their consumers to feel.

### Keywords
Deep Learning, Neuromarketing, convolutional neural networks, CNN, facial expression recognition, emotions, advertisement

## Content of this repository
This repository contains my bachelor thesis, finished during the academic year 2016-2017. All source code for the training and evaluation of the different networks are included, plus the code used to process the results obtained during the experimentation phase with real subjects.

This project is built using mainly `R` and `MXNet`. However, one script also uses `Python` and `OpenCV` for pre-processing (face detection and cropping). All networks follow the Resnet-28-small architecture extracted from MXNet examples, which is also included in the corresponding directories.

Three different approaches were evaluated for facial emotion recognition, namely **classification**, **verification** and **arousal**. The **classification** approach (directory `Classification`) simply consisted on training a model that would classify each facial image according to the emotion that the person is most likely to be feeling (or showing). The **verification** (directory `Verification`) approach, on the other hand, involved training one model for each basic emotion, in the form of *one vs all*. Lastly, the **arousal** (directory `Arousal`) approach tried to measure the level of arousal of each image instead of assigning an emotion label. According to the literature, all emotions can be classified as having a high or a low arousal (in reality, this is a decimal value, not strictly 0 or 1).

For license reasons, the original data, the `CK+` database, was not included. However, it can be obtained from the [official webpage](http://www.consortium.ri.cmu.edu/ckagree/).
