import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import glob
import json
import shutil

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
print("Loading data...")

main_df = pd.read_csv('dontpatronizeme_pcl.tsv', sep='\t', header=None,
                      names=['par_id', 'art_id', 'keyword', 'country', 'text', 'label'], skiprows=4)
main_df['par_id'] = main_df['par_id'].astype(str)
main_df = main_df.dropna(subset=['text', 'label'])
main_df['label'] = pd.to_numeric(main_df['label'], errors='coerce')
main_df['binary_label'] = main_df['label'].apply(lambda x: 1 if x >= 2 else 0)

# 🚀 UPGRADE 1: Richer Context Engineering
main_df['text_with_meta'] = "Target: " + main_df['keyword'].fillna('none') + \
                            " | Country: " + main_df['country'].fillna('none') + \
                            " </s> " + main_df['text']

train_split = pd.read_csv('train_semeval_parids-labels.csv', header=None, names=['par_id', 'label'])
official_dev_split = pd.read_csv('dev_semeval_parids-labels.csv', header=None, names=['par_id', 'label'])

train_split['par_id'] = train_split['par_id'].astype(str)
official_dev_split['par_id'] = official_dev_split['par_id'].astype(str)

train_df = pd.merge(train_split[['par_id']], main_df[['par_id', 'text_with_meta', 'binary_label']], on='par_id', how='inner')
dev_df = pd.merge(official_dev_split[['par_id']], main_df[['par_id', 'text_with_meta', 'binary_label']], on='par_id', how='inner')

print(f"Train: {len(train_df)} (Pos: {train_df['binary_label'].sum()})")
print(f"Dev:   {len(dev_df)} (Pos: {dev_df['binary_label'].sum()})")

# ==========================================
# 2. TOKENIZATION (UPGRADED TO LARGE)
# ==========================================
model_name = "roberta-large" 
tokenizer = AutoTokenizer.from_pretrained(model_name)

class PCLDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=256)
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = PCLDataset(train_df['text_with_meta'], train_df['binary_label'], tokenizer)
dev_dataset = PCLDataset(dev_df['text_with_meta'], dev_df['binary_label'], tokenizer)

# ==========================================
# 3. CUSTOM TRAINER AND CALLBACKS
# ==========================================
class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = torch.tensor([1.0, 4.0], device=model.device, dtype=logits.dtype)
        ce_loss = F.cross_entropy(logits.view(-1, self.model.config.num_labels), labels.view(-1), weight=weights, reduction='none')
        
        pt = torch.exp(-ce_loss)
        gamma = 2.0  
        focal_loss = ((1 - pt) ** gamma) * ce_loss
        
        loss = focal_loss.mean()
        return (loss, outputs) if return_outputs else loss

# 🚀 NEW: The Storage Saver! 
# This watches the checkpoints and deletes the bad ones so your disk never fills up.
class KeepBestCheckpointCallback(TrainerCallback):
    def __init__(self):
        self.best_f1 = 0.0
        self.best_ckpt_path = None

    def on_save(self, args, state, control, **kwargs):
        # 1. Get the F1 score of the epoch that just finished
        current_f1 = 0.0
        for log in state.log_history:
            if 'eval_f1_pos' in log:
                current_f1 = log['eval_f1_pos']

        # 2. Get a list of all current checkpoint folders on disk
        checkpoints = sorted(glob.glob(f"{args.output_dir}/checkpoint-*"), key=lambda x: int(x.split('-')[-1]))
        if not checkpoints:
            return

        latest_ckpt = checkpoints[-1]

        # 3. Update our record of the "best" checkpoint
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self.best_ckpt_path = latest_ckpt

        # 4. DELETE any checkpoint that is NOT the absolute best, and NOT the current one
        for ckpt in checkpoints:
            if ckpt != self.best_ckpt_path and ckpt != latest_ckpt:
                shutil.rmtree(ckpt, ignore_errors=True)
                print(f"\n[Storage Saver] Deleted worse checkpoint: {ckpt} (Saved 1.4 GB!)")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, pos_label=1)
    precision = precision_score(labels, preds, pos_label=1, zero_division=0)
    recall = recall_score(labels, preds, pos_label=1, zero_division=0)
    return {'f1_pos': f1, 'precision': precision, 'recall': recall}

# ==========================================
# 4. TRAINING
# ==========================================
print(f"Initializing {model_name}...")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

training_args = TrainingArguments(
    output_dir='./pcl_best_model',
    num_train_epochs=10,              
    
    per_device_train_batch_size=8,   
    gradient_accumulation_steps=2,   
    per_device_eval_batch_size=16,
    
    learning_rate=1e-5,              
    warmup_ratio=0.1,
    weight_decay=0.05,               
    
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    
    save_total_limit=None, 
    
    fp16=True, 
    report_to="none"
)

trainer = FocalLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    callbacks=[KeepBestCheckpointCallback()] # 🚀 Activating the Storage Saver
)

print("Starting training...")
trainer.train()

# ==========================================
# 5. MANUALLY LOAD THE BEST CHECKPOINT
# ==========================================
print("\n--- Finding best checkpoint ---")
best_f1 = 0
best_checkpoint = None

checkpoints = sorted(glob.glob("./pcl_best_model/checkpoint-*"))
for ckpt_path in checkpoints:
    trainer_state_path = os.path.join(ckpt_path, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        with open(trainer_state_path) as f:
            state = json.load(f)
        
        last_f1 = 0
        for log_entry in state["log_history"]:
            if "eval_f1_pos" in log_entry:
                last_f1 = log_entry["eval_f1_pos"]
        if last_f1 > best_f1:
            best_f1 = last_f1
            best_checkpoint = ckpt_path

print(f"Best checkpoint: {best_checkpoint} (F1: {best_f1:.4f})")

best_model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint, num_labels=2).to(trainer.args.device)
trainer.model = best_model

# ==========================================
# 6. THRESHOLD TUNING
# ==========================================
print("\n--- Threshold Tuning ---")

predictions = trainer.predict(dev_dataset)
logits = torch.tensor(predictions.predictions)
probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
true_labels = predictions.label_ids

best_f1 = 0
best_threshold = 0.5
best_p = 0
best_r = 0

for threshold in np.arange(0.15, 0.85, 0.005): 
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(true_labels, preds, pos_label=1)
    p = precision_score(true_labels, preds, pos_label=1, zero_division=0)
    r = recall_score(true_labels, preds, pos_label=1, zero_division=0)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_p = p
        best_r = r

print(f"Best Threshold: {best_threshold:.3f}")
print(f"Best F1:        {best_f1:.4f}")
print(f"Precision:      {best_p:.4f}")
print(f"Recall:         {best_r:.4f}")

# ==========================================
# 7. SAVE FINAL ASSETS
# ==========================================
best_model.save_pretrained("./BestModel")
tokenizer.save_pretrained("./BestModel")

with open("./BestModel/threshold.txt", "w") as f:
    f.write(str(best_threshold))

print(f"\nModel + threshold saved to ./BestModel/")