# Fetal Tumor Segmentation in Ultrasound Images

This project focuses on identifying tumor regions in fetal ultrasound images using a classic U-Net architecture. The main idea is to take an ultrasound image and produce a heatmap, where each pixel shows how likely it is to belong to a tumor.

Current stage: Data preprocessing and dataset validation completed

---

## Dataset

- **Source:** https://www.kaggle.com/datasets/orvile/ultrasound-fetus-dataset  
- **Annotations:** Pixel-level binary masks  
- **Annotation type:** Approximate tumor boundaries  

Only malignant cases that have corresponding tumor masks are used for training and analysis.

---

## Project Pipeline

The project follows a step-by-step pipeline, starting from raw data and moving toward model training and evaluation.

1. **Dataset loading and directory structuring (completed)**  
   Ultrasound images and masks are loaded and organized into train, validation, and test folders.

2. **Image and mask preprocessing (completed)**  
   Images are normalized, masks are binarized, and all files are resized to a fixed shape.

3. **Dataset and mask verification (completed)**  
   The dataset is checked for missing masks, empty masks, shape mismatches, tumor locations near image borders, and intensity differences between tumor and background regions.

4. **Model architecture implementation (planned)**  
   A U-Net–based segmentation model will be implemented for pixel-level tumor prediction.

5. **Planned model training**  
   The model will be trained using the available annotated ultrasound images.

6. **Evaluation and visualization (planned)**  
   The results will be evaluated using segmentation metrics and visual overlays.

At present, the project is transitioning from step 3 to step 4.

---

## Preprocessing (Completed)

This stage prepares the data so that it can be safely used for training a segmentation model. The goal is to remove inconsistencies and avoid errors later.

Steps performed include normalization of ultrasound image intensities, binarization of segmentation masks, resizing images and masks to a fixed resolution, verifying filename-based image–mask pairing, and checking shape consistency between images and masks.

---

## Dataset & Mask Analysis (Completed)

This stage analyzes the dataset to understand the quality of the images and masks before training.

The checks performed include confirming that every image has a corresponding annotation, identifying cases where no tumor pixels are present, ensuring image and mask dimensions match, checking whether tumor regions are too close to image borders, and comparing pixel intensity values between tumor and background areas.

These checks help confirm that the dataset is usable and meaningful for segmentation.

---

## Data Verification (Completed)

The verification code runs on each dataset split (train, validation, and test) and prints summary statistics.

The output mainly includes basic information such as how many samples are present, whether any masks are empty, how often tumors appear near image borders, and how different tumor regions are from the background in terms of intensity.

This step serves as a final sanity check before model training begins.

---

## Planned Model Architecture

The segmentation model will initially be based on the original U-Net architecture proposed in 2015.

Planned characteristics:
- Single-channel (grayscale) image input  
- Pixel-wise probability output in the form of a heatmap  
- Skip connections between encoder and decoder layers  
- Sigmoid activation for binary segmentation  

The output heatmap can be thresholded to obtain a final binary tumor mask.

**Reference:**  
Ronneberger, O., Fischer, P., and Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation* (2015)

---

## Training and Evaluation (Planned)

- **Loss Functions:** Dice Loss and Binary Cross-Entropy combined with Dice Loss  
- **Metrics:** Dice Coefficient and Intersection over Union (IoU)  
- **Evaluation:** Quantitative metrics along with visual inspection of predicted segmentation outputs  

---

## Disclaimer

This project is intended for **academic and learning purposes only**.  
The annotations in the dataset are approximate, and the results produced by the model are **not meant for clinical use or medical decision-making**.
