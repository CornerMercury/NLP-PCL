import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading model...")
model_path = "./BestModel"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

THRESHOLD = 0.500 

print("Loading Dev Data...")
main_df = pd.read_csv('./data/dontpatronizeme_pcl.tsv', sep='\t', header=None,
                      names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], skiprows=4)
main_df['par_id'] = main_df['par_id'].astype(str)
main_df['label'] = pd.to_numeric(main_df['label'], errors='coerce')
main_df['binary_label'] = main_df['label'].apply(lambda x: 1 if x >= 2 else 0)

dev_split = pd.read_csv('./data/dev_semeval_parids-labels.csv', header=None, names=['par_id', 'label'])
dev_split['par_id'] = dev_split['par_id'].astype(str)

dev_df = pd.merge(dev_split[['par_id']], main_df[['par_id', 'keyword', 'country', 'text', 'binary_label']], on='par_id', how='inner')
dev_df['text_with_meta'] = "Target: " + dev_df['keyword'].fillna('none') + " | Country: " + dev_df['country'].fillna('none') + " </s> " + dev_df['text'].fillna('').astype(str)

print("Predicting...")
predictions = []
batch_size = 16
texts = dev_df['text_with_meta'].tolist()

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
    predictions.extend((probs >= THRESHOLD).astype(int))

dev_df['prediction'] = predictions

# confusion matrix
cm = confusion_matrix(dev_df['binary_label'], dev_df['prediction'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No PCL (0)', 'PCL (1)'], yticklabels=['No PCL (0)', 'PCL (1)'])
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title(f'Confusion Matrix (Threshold = {THRESHOLD})')
plt.savefig('confusion_matrix.png')
print("Saved confusion_matrix.png")

false_positives = dev_df[(dev_df['binary_label'] == 0) & (dev_df['prediction'] == 1)]
false_negatives = dev_df[(dev_df['binary_label'] == 1) & (dev_df['prediction'] == 0)]

false_positives[['keyword', 'text']].to_csv('false_positives.csv', index=False)
false_negatives[['keyword', 'text']].to_csv('false_negatives.csv', index=False)
print(f"Saved {len(false_positives)} False Positives and {len(false_negatives)} False Negatives.")