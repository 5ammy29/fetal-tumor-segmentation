# Fetal Tumor Segmentation in Ultrasound Images

This repository implements a U-Net model for identifying tumor regions in fetal ultrasound images. The model takes an ultrasound image as input and produces a pixel-wise probability map of tumor regions.

---

## Problem Overview

Fetal ultrasound images are difficult to analyze because they contain noise, low contrast, and unclear boundaries. As a result, identifying tumor regions manually is slow and can vary between observers.

This project labels each pixel in an ultrasound image as tumor or background.
- **Tumor (1)**
- **Background (0)**

The goal is to automatically locate tumor regions at the pixel level.

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset  
- **Annotations:** Pixel-level binary masks  
- **Annotation type:** Approximate tumor boundaries  

Only malignant cases that have corresponding tumor masks are used for training and analysis.

---

## Project Pipeline

The project follows the pipeline described below.

1. **Dataset Loading and Directory Structuring**  
   Ultrasound images and corresponding masks are loaded and organized into train, validation, and test splits.

2. **Image and Mask Preprocessing**  
   Images are normalized to a fixed intensity range, masks are binarized, and all samples are resized to a consistent resolution.

3. **Dataset and Mask Verification**  
   The dataset is checked for missing masks, empty masks, shape mismatches, tumor regions near image borders, and intensity differences between tumor and background areas.

4. **Model Architecture Implementation**  
   A U-Net–based segmentation model is implemented for pixel-level tumor prediction.

5. **Model Training**  
   The model is trained on annotated malignant ultrasound images using a combined Dice and BCE loss.

6. **Evaluation and Visualization**  
   The trained model is evaluated using Dice and IoU metrics, and predicted masks are generated for images in the test set.

---

## Model Architecture

The segmentation model is based on the original U-Net architecture proposed in 2015.

It follows an encoder–decoder structure with skip connections that preserve spatial information between upsampling and downsampling. The model takes a grayscale ultrasound image as input and produces a pixel-wise tumor probability map as output.

Applying a threshold converts the heatmap into a binary tumor mask.

**Reference:**  
Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation (2015)

---

## Training Configuration

- **Loss Function:** Dice Loss + 0.2 × BCEWithLogitsLoss  
- **Optimizer:** Adam  
- **Epochs:** 35 (early stabilization at epoch 29)  

The combined Dice and BCE loss improves overlap accuracy while maintaining stable gradients.

---

## Evaluation Results

Evaluation was conducted on a separate test set.

- **Mean Dice Score:** 0.9367  
- **Mean IoU:** 0.9000  

---

## Disclaimer

This project is intended for academic and learning purposes only.  
The annotations in the dataset are approximate, and the results produced by the model are not meant for clinical use or medical decision-making.
