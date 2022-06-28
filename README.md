# GTSRB Traffic Sign Recognition

This repository contains all of the models developed for Computer Vision and Image Processing lab project. These DL models were trained to classify road signs using the GTSRB dataset.

**Primary Features Employed**

1. Offline pre-processing with image filters such as Sobel's, Local Histogram Equalization, Unsharp Masking ...
2. Real-time data augmentation (translations, rotations, shearing, color manipulation) embedded as in-model layers
3. Ensembles of stacked CNN's

Frameworks/packages utilized/needed
========

- Tensorflow 2.6
- OpenCV and scikit-image
- Numpy and Pandas

Project Structure
========

## Data

---
The `data` directory contain all of the used training and testing images.

Besides the training and test data, the notebooks included can be used to create new pre-processed images by applying various filters (Sobel, Unsharp Mask, LHE, etc...).

**The download of this repo can take some time and a bit of storage. If you'd prefer, you can also just download the source from the second [branch](https://github.com/AsuosOnurb/traffic-sign-recognition/tree/source_only).**

## CNNs

---

The CNN models are all located under the `src/cnns` directory.

At the moment, there are 2 different architectures, each trained with images with different pre-processing filters.

## Ensembles

---
The CNN models are all located under the `src/ensembles` directory.

At the moment, there are 2 different architectures, each trained with images with different pre-processing filters.

