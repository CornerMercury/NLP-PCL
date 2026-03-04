import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading model and tokenizer from ./BestModel ...")
model_path = "./BestModel"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

THRESHOLD = 0.500 
print(f"Using Threshold: {THRESHOLD}")

print("Loading datasets from ./data/ ...")

main_df = pd.read_csv('./data/dontpatronizeme_pcl.tsv', sep='\t', header=None,
                      names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], skiprows=4)
main_df['par_id'] = main_df['par_id'].astype(str)

official_dev_split = pd.read_csv('./data/dev_semeval_parids-labels.csv', header=None, names=['par_id', 'label'])
official_dev_split['par_id'] = official_dev_split['par_id'].astype(str)

test_df = pd.read_csv('./data/task4_test.tsv', sep='\t', header=None, 
                      names=['par_id', 'art_id', 'keyword', 'country', 'text'])

dev_df = pd.merge(official_dev_split[['par_id']], main_df[['par_id', 'keyword', 'country', 'text']], on='par_id', how='inner')

# FIX: Added .fillna('').astype(str) to the end of the text column to prevent the TypeError
dev_df['text_with_meta'] = "Target: " + dev_df['keyword'].fillna('none') + " | Country: " + dev_df['country'].fillna('none') + " </s> " + dev_df['text'].fillna('').astype(str)
test_df['text_with_meta'] = "Target: " + test_df['keyword'].fillna('none') + " | Country: " + test_df['country'].fillna('none') + " </s> " + test_df['text'].fillna('').astype(str)

# Inference function
def predict(texts):
    predictions = []
    batch_size = 16
    
    for i in range(0, len(texts), batch_size):
        # FIX: Explicitly cast every element to string just in case
        batch_texts = [str(text) for text in texts[i:i+batch_size].tolist()]
        
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        
        # Apply the threshold
        batch_preds = (probs >= THRESHOLD).astype(int)
        predictions.extend(batch_preds)
        
    return predictions

print("\nGenerating predictions for Dev set...")
dev_preds = predict(dev_df['text_with_meta'])

print("Generating predictions for Test set...")
test_preds = predict(test_df['text_with_meta'])

print("Saving dev.txt...")
with open("dev.txt", "w") as f:
    for pred in dev_preds:
        f.write(f"{pred}\n")

print("Saving test.txt...")
with open("test.txt", "w") as f:
    for pred in test_preds:
        f.write(f"{pred}\n")

print(f"dev.txt lines:  {len(dev_preds)}")
print(f"test.txt lines: {len(test_preds)}")