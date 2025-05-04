# Reproducing MIMIC-SBDH: Baseline Models and Extension Study

This repository contains three Python scripts to reproduce and extend the findings of the MIMIC-SBDH paper on social and behavioral determinants of health (SBDH) classification using clinical notes from the MIMIC-III dataset.

We reproduce baseline models from the original paper—Random Forest, XGBoost and Bio-ClinicalBERT —and further extend the study by introducing four comparison models: Logistic Regression (LR), BERT-LR, BERT-RF, and BERT-XGBoost.

##  Repository Structure
### 1. SaveClinicBERTEmbedding.py

This script extracts [CLS] token embeddings from the pre-trained Bio-ClinicalBERT model using discharge summaries from MIMIC-III. The embeddings are saved as bert-embed.npy and are used as input features for downstream classification models.

### 2.rf_xgb_extension.py
This script implements both baseline and extension study models to classify 8 SBDH labels.
Models Implemented:

#### Baseline models (from the original paper):

-Random Forest

-XGBoost

#### Extension models:
-Logistic Regression (LR) using traditional features

-BERT-LR: LR on BERT embeddings

-BERT-RF: Random Forest on BERT embeddings

-BERT-XGBoost: XGBoost on BERT embeddings

### 3.Bio-ClinicalBERT.py
This script implements the Bio-ClinicalBERT baseline model as described in the original MIMIC-SBDH paper. It fine-tunes the Bio-ClinicalBERT model using an MLP classifier for each label.

## How to Run the Code
### Prerequisites

Install required packages, you can run the following command: 
pip install transformers torch pandas numpy scikit-learn xgboost

You must have access to:
MIMIC-SBDH.csv
NOTEEVENTS.csv (from MIMIC-III)

📁 Note: In our implementation, we run the code in Google Colab and load the CSV files from Google Drive or Dropbox for accessibility.
We recommend running the code in Google Colab for ease of setup and GPU access.

### Step-by-Step Instructions

Step 1: Generate BERT Embeddings
Run SaveClinicBERTEmbedding.py to generate bert-embed.npy, which contains Bio-ClinicalBERT embeddings used by downstream classifiers.

Step 2: Train and Evaluate Baseline and Extension Models
Run rf_xgb_extension.py to evaluate two baseline models — Random Forest and XGBoost — and four comparison models from the extension study: Logistic Regression (LR), BERT-LR, BERT-RF, and BERT-XGBoost.

Step 3: Train and Evaluate Bio-ClinicalBERT Model
Run Bio-ClinicalBERT.py to fine-tune Bio-ClinicalBERT for SBDH classification. This serves as the third baseline model from the original paper.


