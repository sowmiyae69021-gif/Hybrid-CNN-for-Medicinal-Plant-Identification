# Hybrid CNN for Medicinal Plant Identification

## Table of Contents

1. Overview
2. Repository Structure
3. Environment Setup
4. Dataset Preparation
5. Dataset Splits
6. Training the Model
7. Model Evaluation
8. Cross-Validation Experiments
9. Ablation Experiments
10. Model Export for Mobile Deployment
11. Reproducing the Results
12. Dataset Sources

---

# 1 Overview

This repository provides the **complete implementation of the hybrid CNN framework** proposed in the manuscript:

**Hybrid Inception–ResNet CNN for Medicinal Plant Identification**

The proposed model integrates **InceptionV3** and **ResNet50** feature extractors using **feature fusion through concatenation** to improve classification performance on medicinal plant leaf images.

The repository contains:

* dataset preparation scripts
* preprocessing and augmentation pipeline
* hybrid CNN architecture implementation
* training and evaluation scripts
* cross-validation experiments
* ablation studies
* TensorFlow Lite model export for mobile inference
* example inference notebook

The code enables **full reproduction of all experiments reported in the manuscript**.

---

# 2 Repository Structure

```
medicinal-plant-hybrid-cnn
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs
│   └── default.yaml
│
├── scripts
│   ├── prepare_dataset.py
│   ├── make_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── crossval.py
│   ├── ablation.py
│   └── export_tflite.py
│
├── src
│   ├── data.py
│   ├── augment.py
│   ├── model.py
│   ├── metrics.py
│   ├── utils.py
│   └── reproducibility.py
│
├── notebooks
│   └── inference_demo.ipynb
│
└── mobile
    └── android_tflite
```

---

# 3 Environment Setup

Python version used:

```
Python 3.10
```

Install dependencies:

```
pip install -r requirements.txt
```

The project was tested using:

* TensorFlow 2.13
* CUDA GPU (optional)
* Ubuntu / Google Colab

The implementation also runs on CPU systems.

---

# 4 Dataset Preparation

Two publicly available datasets were used in the experiments.

### PlantCLEF 2015 Dataset

Large-scale plant species image dataset containing multiple plant categories.

### Indian Medicinal Plants Dataset

Leaf images of medicinal plant species.

After downloading both datasets, place them inside:

```
data/raw/
```

Example structure:

```
data/raw
├── plantclef2015
│   ├── aloe_vera
│   ├── neem
│   └── ...
│
└── indian_medicinal_plants
    ├── tulsi
    ├── brahmi
    └── ...
```

Then run:

```
python scripts/prepare_dataset.py
```

This script:

* validates images
* removes corrupted files
* standardizes class folder structure
* creates the processed dataset

Output:

```
data/processed/
```

---

# 5 Dataset Splits

To ensure reproducibility, the dataset is split using a deterministic procedure.

Run:

```
python scripts/make_splits.py
```

Generated files:

```
data/splits/
├── train.csv
├── val.csv
└── test.csv
```

Split ratios:

```
Train: 70%
Validation: 15%
Test: 15%
```

Stratified sampling preserves class distribution.

---

# 6 Training the Model

Train the hybrid CNN model:

```
python scripts/train.py
```

Training procedure:

1. load dataset splits
2. apply data augmentation
3. initialize pretrained backbones
4. perform feature fusion
5. train classifier layers

Training outputs:

```
runs/
├── checkpoints
│   ├── best_model.h5
│   └── final_model
│
└── logs
    └── training_history.json
```

---

# 7 Model Evaluation

Evaluate the trained model on the test dataset:

```
python scripts/evaluate.py
```

Generated reports:

```
reports/
├── classification_report.txt
├── confusion_matrix.png
└── metrics_summary.json
```

Metrics computed:

* Accuracy
* Precision
* Recall
* F1 score

---

# 8 Cross-Validation Experiments

To reproduce cross-validation experiments:

```
python scripts/crossval.py
```

This performs **5-fold stratified cross-validation**.

Generated results:

```
reports/crossval_results.json
```

Example output:

```
Accuracy: 96.95 ± 0.42
Precision: 96.72 ± 0.35
Recall: 96.50 ± 0.38
F1 Score: 96.61 ± 0.33
```

---

# 9 Ablation Experiments

Ablation studies analyze the contribution of different components of the architecture.

Run:

```
python scripts/ablation.py
```

Compared architectures:

* InceptionV3 only
* ResNet50 only
* Hybrid without feature fusion
* Full hybrid CNN

Results saved in:

```
reports/ablation_results.csv
```

---

# 10 Model Export for Mobile Deployment

Export the trained model to TensorFlow Lite:

```
python scripts/export_tflite.py
```

Generated files:

```
exports/
├── model.tflite
└── labels.txt
```

These files can be used for **mobile inference applications**.

---

# 11 Reproducing the Results

Complete reproduction pipeline:

Step 1 — install dependencies

```
pip install -r requirements.txt
```

Step 2 — prepare dataset

```
python scripts/prepare_dataset.py
```

Step 3 — create dataset splits

```
python scripts/make_splits.py
```

Step 4 — train model

```
python scripts/train.py
```

Step 5 — evaluate model

```
python scripts/evaluate.py
```

Step 6 — export model

```
python scripts/export_tflite.py
```

---

# 12 Dataset Sources

PlantCLEF 2015
[https://www.imageclef.org/lifeclef/2015/plant](https://www.imageclef.org/lifeclef/2015/plant)

Indian Medicinal Plants Dataset
[https://www.kaggle.com/datasets](https://www.kaggle.com/datasets)

Due to dataset licensing restrictions, the datasets are not redistributed in this repository.
Scripts are provided to reconstruct the dataset structure locally.


