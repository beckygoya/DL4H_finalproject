#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import re
import numpy as np

#%% 
# Load the MIMIC-SBDH dataset 
def load_annotation_data():
    annotation_path = "/home/youyexie/Dropbox/UIUC-MCS/CS598DeepLearningForHealthcare/project/MIMIC-SBDH.csv"
    columns = ['row_id', 'sdoh_community_present', 'sdoh_community_absent',
               'sdoh_education', 'sdoh_economics', 'sdoh_environment',
               'behavior_alcohol', 'behavior_tobacco', 'behavior_drug']
    return pd.read_csv(annotation_path, usecols=columns)

# Load the original MIMIC-III dataset and the discharge summaries
def load_mimic_data():
    mimic_path = "/home/youyexie/Dropbox/UIUC-MCS/CS598DeepLearningForHealthcare/project/NOTEEVENTS.csv"
    mimic_data = pd.read_csv(
        mimic_path,
        usecols=['ROW_ID', 'TEXT', 'CATEGORY']
    )
    return mimic_data[mimic_data['CATEGORY'] == 'Discharge summary']

# Extract social history section from discharge summary
def extract_social_history(text):
    if pd.isna(text):
        return ""
    pattern = r"social history:\s*(.*?)(?=\n[a-z\s]+:|\Z)"
    match = re.search(pattern, text.lower(), re.DOTALL)
    return match.group(1).strip() if match else ""

# Merge and prepare the dataset
def prepare_dataset(annotated_data, mimic_data):
    annotated_data = annotated_data.rename(columns={'row_id':'ROW_ID'})
    merged = pd.merge(
        annotated_data,
        mimic_data[['ROW_ID', 'TEXT']],
        on='ROW_ID'
    )
    merged['social_history'] = merged['TEXT'].apply(extract_social_history)
    return merged[merged['social_history'] != ""]

#%% Calculate the embedding using pre-trained bio clinical BERT and save the result

data = prepare_dataset(load_annotation_data(), load_mimic_data())
print(f"Number of samples: {len(data)}")

texts = data['social_history']
texts_list = texts.tolist()
X = np.empty((0, 768)) 
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
for i in range(len(texts_list)):
    print("processing ",i)
    inputs = tokenizer(texts_list[i], return_tensors="pt", padding="max_length", truncation=True,max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # Take [CLS] token
        X = np.append(X, embeddings, axis=0) 

np.save('bert-embed.npy', X)

