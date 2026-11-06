# Metal Nut Anomaly Detection using PatchCore

An image-based anomaly detection project for identifying defects in metal nuts using PatchCore, ONNX Runtime, and C++.

The project explores how deep learning-based feature extraction and distance-based anomaly detection can be applied to visual inspection tasks.

## Demo

[Watch the demonstration video](https://drive.google.com/file/d/1tYTPK982gbc4pCzoyvGLgsD2gqk7OVY9/view?usp=sharing)

## Results

The C++ application processes an input image and produces an anomaly score, a classification result, and an anomaly map highlighting potentially defective regions.

![Anomaly Map](anomaly-map.png)

## Why C++?

I chose C++ for three reasons:

- Gain a deeper understanding of data structures, memory organization, and image processing operations.
- Strengthen my C++ foundation for future applications in robotics, embedded systems, and Industrial AI.
- Build on my existing programming knowledge

## Development Process

### 1. Dataset Preparation

This project uses the **metal_nut** category from the MVTec Anomaly Detection (MVTec AD) dataset.

- **Training:** 220 normal images
- **Testing:** 22 normal images and 93 defective images

The original dataset is not included in this repository. It is available on the [official MVTec AD dataset page](https://www.mvtec.com/company/research/datasets/mvtec-ad).

### 2. PatchCore Model Preparation

I chose PatchCore because it detects anomalies by learning the feature patterns of normal objects, without requiring examples of every possible defect.

As I had limited experience with Python, preparing the pretrained backbone and model outputs was initially challenging. I used existing scripts, documentation, and AI assistance to set up this stage, while gradually learning how the model extracts and uses image features.

### 3. C++ Implementation (Visual Studio)

The main part of this project was implementing the inference pipeline in C++ using OpenCV and ONNX Runtime.

**3.1. Image Preprocessing**

Images are loaded, resized, and normalized before being converted into the required channel-first (CHW) tensor format. Understanding how to arrange pixel data using nested loops took some time.

**3.2. Feature Extraction**

The pretrained CNN backbone extracts features from Layer 2 and Layer 3. Since their feature maps have different spatial dimensions, they must be resized and aligned before being combined.

**3.3. Memory Bank and Nearest-Neighbor Search**

Features from normal training images are stored in a memory bank. During inference, each test-image patch is compared with these stored features using nearest-neighbor distance.

Working with multidimensional vectors, matrices, and `cv::Mat` was one of the more challenging parts of the C++ implementation.

**3.4. Anomaly Scoring and Visualization**

Nearest-neighbor distances are used to calculate image-level anomaly scores and generate anomaly maps highlighting potentially defective regions. These maps are then overlaid on the original images for visualization.

### 4. Testing and Evaluation

I compared the C++ outputs with those from the original Python implementation to check their consistency. I also tested different images, inspected anomaly scores and heatmaps, and refined the code based on the results.

Finally, I used Python to evaluate different classification thresholds, exploring the trade-off between detecting defective samples and avoiding false alarms.

## Evaluation Results

The selected threshold was chosen to prioritize detecting defective samples.

| Metric | Result |
|---|---:|
| ROC AUC | 0.9560 |
| Threshold | 76.670616 |
| Recall | 88.17% |
| False Positive Rate | 4.55% |
| Detected anomalies | 82 / 93 |
| Missed anomalies | 11 |
| Good images incorrectly flagged | 1 / 22 |

[View Python Results](python-results.png)

## Model Files

The C++ inference application uses the following files:

| File | Description |
|---|---|
| `patchcore_backbone.onnx` | CNN backbone for feature extraction |
| `memory_bank.bin` | Stored normal patch feature vectors |

[View Model Files](model-files.png)

[Download memory_bank.bin](memory_bank.bin)

[Download patchcore_backbone.onnx](https://drive.google.com/file/d/1XLOI0bCrdDjUd9QXy5N0JNEBCKiGrbi8/view?usp=sharing)

## Requirements

- C++
- OpenCV
- ONNX Runtime
- Python
- NumPy
- PyTorch
- ONNX Runtime Python package

## How to Run

1. Install OpenCV and ONNX Runtime.
2. Place the ONNX model and memory bank files in the expected locations.
3. Configure the C++ project to link against the required libraries.
4. Build and run the C++ application.
5. Enter the path to an input image.
6. View the anomaly score, classification result, and anomaly map.

## Limitations: 
The model may still miss certain defective samples, particularly those with subtle or less distinctive anomalies. Its performance has only been evaluated on the MVTec AD Metal Nut dataset, and further validation under different objects and imaging conditions is needed.

## Dataset Attribution

Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger,  
"A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,"  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

**Dataset:** [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

**License:** CC BY-NC-SA 4.0  
Copyright 2019 MVTec Software GmbH.

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)



