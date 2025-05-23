# -*- coding: utf-8 -*-
"""train_bert_finetune.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YDvcpEj99Xz7s3WJZ3HBlgC2XYhlzy-Z
"""

# -*- coding: utf-8 -*-
"""
Fine-tunes a ClinicalBERT model (using custom layer) for multi-label
ICD code classification.
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
from transformers import AutoTokenizer, TFAutoModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
import os
import gc
import argparse
import traceback
# Import evaluation utilities if created
# from src.evaluation.evaluate_model import find_optimal_threshold, plot_history

# --- Configuration (Example) ---
# BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
# MAX_BERT_LENGTH = 512
# N_TOP_CODES = 100
# BATCH_SIZE = 8
# EPOCHS = 5 # Start low for fine-tuning
# LEARNING_RATE = 3e-5
# EARLY_STOPPING_PATIENCE = 2
# OPTIMIZATION_THRESHOLD_STEP = 0.05
# MODEL_OUTPUT_DIR = '../../models/bert_finetuned/' # Example

# --- Custom BertLayer Class ---
# (Include the BertLayer class definition here as provided previously)
class BertLayer(Layer):
   """Custom Keras layer to wrap Hugging Face TFBertModel."""
   def __init__(self, bert_model_name, trainable=True, **kwargs):
       super().__init__(**kwargs)
       self.bert_model_name = bert_model_name
       self.trainable_bert = trainable
       self.bert = None
       try:
           self.bert = TFAutoModel.from_pretrained(self.bert_model_name, from_pt=True) # Assuming PyTorch weights
           self.bert.trainable = self.trainable_bert
           print(f"HF model '{self.bert_model_name}' loaded. Trainable: {self.bert.trainable}")
       except Exception as e: print(f"Error loading HF model in BertLayer: {e}"); raise e
   def call(self, inputs):
       try: return self.bert(inputs).last_hidden_state
       except Exception as e: print(f"Error during BertLayer call: {e}"); raise e
   # def compute_output_shape... (optional)


def build_finetuning_model_custom(bert_model_name, num_labels, max_length, bert_trainable=True):
   """Builds Keras Model using the custom BertLayer."""
   bert_layer_instance = BertLayer(bert_model_name, trainable=bert_trainable, name='bert_layer')
   input_ids = Input(shape=(max_length,), dtype=tf.int32, name='input_ids')
   attention_mask = Input(shape=(max_length,), dtype=tf.int32, name='attention_mask')
   bert_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
   bert_output_last_hidden_state = bert_layer_instance(bert_inputs)
   cls_output = bert_output_last_hidden_state[:, 0, :]
   x = Dropout(0.1)(cls_output)
   classifier_output = Dense(num_labels, activation='sigmoid', name='classifier')(x)
   model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
   model = Model(inputs=model_inputs, outputs=classifier_output)
   return model

def fine_tune_bert(processed_data_path, config):
   """Orchestrates the BERT fine-tuning process."""
   print("\n--- Starting ClinicalBERT Fine-Tuning ---")
   print(f"Loading processed data from: {processed_data_path}")
   try:
       df = pd.read_csv(processed_data_path)
       # Convert list-like strings back to lists
       label_col = 'ICD9_CODE_filtered' if 'ICD9_CODE_filtered' in df.columns else 'ICD9_CODE'
       from ast import literal_eval
       if isinstance(df[label_col].iloc[0], str):
            df[label_col] = df[label_col].apply(literal_eval)
       df['TEXT'] = df['TEXT'].fillna('').astype(str)
   except Exception as e:
       print(f"Error loading or processing data file {processed_data_path}: {e}")
       return

   N_TOP_CODES = config.get('N_TOP_CODES', 100)
   BERT_MODEL_NAME = config.get('BERT_MODEL_NAME', "emilyalsentzer/Bio_ClinicalBERT")
   MAX_BERT_LENGTH = config.get('MAX_BERT_LENGTH', 512)
   BATCH_SIZE = config.get('BATCH_SIZE', 8)
   EPOCHS = config.get('EPOCHS', 5)
   LEARNING_RATE = config.get('LEARNING_RATE', 3e-5)
   EARLY_STOPPING_PATIENCE = config.get('EARLY_STOPPING_PATIENCE', 2)
   OPTIMIZATION_THRESHOLD_STEP = config.get('OPTIMIZATION_THRESHOLD_STEP', 0.05)
   MODEL_OUTPUT_DIR = config.get('MODEL_OUTPUT_DIR', '../../models/bert_finetuned')

   # --- Filter Top N Codes (if not done during preprocessing) ---
   if 'ICD9_CODE_filtered' not in df.columns:
       print(f"\nFiltering for Top {N_TOP_CODES} codes...")
       # (Add filtering logic here if needed)
       label_col = 'ICD9_CODE_filtered' # Assume filtering happens and adds this col
       if label_col not in df.columns: label_col = 'ICD9_CODE'

   # --- Binarize Labels ---
   print("\nBinarizing labels...")
   mlb = MultiLabelBinarizer()
   y = mlb.fit_transform(df[label_col])
   num_classes = len(mlb.classes_)
   print(f"Number of classes: {num_classes}")

   # --- Train/Val/Test Split ---
   print("\nSplitting data...")
   df_train, df_temp, y_train, y_temp = train_test_split(df, y, test_size=0.2, random_state=42)
   df_val, df_test, y_val, y_test = train_test_split(df_temp, y_temp, test_size=0.5, random_state=42)
   print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
   del df, df_temp, y_temp; gc.collect()

   # --- Tokenize ---
   print("\nLoading Tokenizer and Tokenizing text data...")
   tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
   def batch_tokenize(texts, tokenizer, max_len):
       texts = [str(text) if pd.notna(text) else "" for text in texts]
       return tokenizer(texts, add_special_tokens=True, max_length=max_len, padding='max_length', truncation=True, return_token_type_ids=False, return_attention_mask=True, return_tensors='np')
   try:
       X_train_inputs = {'input_ids': batch_tokenize(df_train['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['input_ids'], 'attention_mask': batch_tokenize(df_train['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['attention_mask']}
       X_val_inputs = {'input_ids': batch_tokenize(df_val['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['input_ids'], 'attention_mask': batch_tokenize(df_val['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['attention_mask']}
       X_test_inputs = {'input_ids': batch_tokenize(df_test['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['input_ids'], 'attention_mask': batch_tokenize(df_test['TEXT'].tolist(), tokenizer, MAX_BERT_LENGTH)['attention_mask']}
       print("Tokenization complete.")
       del df_train, df_val, df_test; gc.collect()
   except Exception as e: print(f"Tokenization failed: {e}"); traceback.print_exc(); exit()


   # --- Build & Compile ---
   print("\nBuilding & Compiling Model...")
   try:
       tf.keras.backend.clear_session() # Clear session before building
       model = build_finetuning_model_custom(BERT_MODEL_NAME, num_classes, MAX_BERT_LENGTH, bert_trainable=True)
       optimizer = Adam(learning_rate=LEARNING_RATE)
       model.compile(optimizer=optimizer, loss='binary_crossentropy',
                     metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc_roc', multi_label=True),
                              tf.keras.metrics.AUC(name='auc_pr', curve='PR', multi_label=True)])
       print("Model built and compiled.")
       # print(model.summary())
   except Exception as e: print(f"Model build/compile failed: {e}"); traceback.print_exc(); exit()

   # --- Train ---
   print("\nStarting Training...")
   early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
   try:
       history = model.fit(X_train_inputs, y_train, validation_data=(X_val_inputs, y_val),
                           epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], verbose=1)
       print("Training finished.")
   except Exception as e: print(f"Training failed: {e}"); traceback.print_exc(); exit()

   # --- Evaluate ---
   print("\nEvaluating Model...")
   # (Add evaluation logic: predict, threshold optimize, report)
   # ... similar to logreg script's evaluation section ...
   y_val_pred_proba = model.predict(X_val_inputs, verbose=1, batch_size=BATCH_SIZE)
   y_test_pred_proba = model.predict(X_test_inputs, verbose=1, batch_size=BATCH_SIZE)

   # Threshold optimization...
   print("Finding optimal threshold...")
   best_threshold = 0.5; best_f1 = -1.0
   thresholds = np.arange(0.1, 0.6, OPTIMIZATION_THRESHOLD_STEP)
   for threshold in thresholds:
       f1 = f1_score(y_val, (y_val_pred_proba >= threshold).astype(int), average='micro', zero_division=0)
       if f1 > best_f1: best_f1 = f1; best_threshold = threshold
   print(f"Best threshold: {best_threshold:.2f} (Val Micro F1: {best_f1:.4f})")

   y_test_pred_bin = (y_test_pred_proba >= best_threshold).astype(int)

   # Reporting...
   print("\n--- Final Test Set Performance ---")
   print(f"Using threshold: {best_threshold:.2f}")
   # ... print F1 scores ...
   print("\nClassification Report:")
   print(classification_report(y_test, y_test_pred_bin, target_names=mlb.classes_, zero_division=0))

   # --- Save ---
   if not os.path.exists(MODEL_OUTPUT_DIR): os.makedirs(MODEL_OUTPUT_DIR)
   # Save the fine-tuned model (consider saving weights only or full model)
   model_save_path = os.path.join(MODEL_OUTPUT_DIR, 'fine_tuned_bert_model')
   print(f"\nSaving fine-tuned model to {model_save_path}...")
   model.save(model_save_path)
   # Save the MLB object
   mlb_path = os.path.join(MODEL_OUTPUT_DIR, 'mlb.joblib')
   joblib.dump(mlb, mlb_path)
   print("Model artifacts saved.")


if __name__ == '__main__':
    # Example usage if run as a script
   parser = argparse.ArgumentParser(description='Fine-tune ClinicalBERT for ICD coding.')
   parser.add_argument('--data_path', type=str, required=True, help='Path to the processed CSV/Pickle file')
   parser.add_argument('--output_dir', type=str, default='../../models/bert_finetuned', help='Directory to save model artifacts')
   parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
   parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
   parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')

   args = parser.parse_args()

   config = {
       'N_TOP_CODES': 100,
       'BERT_MODEL_NAME': "emilyalsentzer/Bio_ClinicalBERT",
       'MAX_BERT_LENGTH': 512,
       'BATCH_SIZE': args.batch_size,
       'EPOCHS': args.epochs,
       'LEARNING_RATE': args.lr,
       'EARLY_STOPPING_PATIENCE': 2,
       'OPTIMIZATION_THRESHOLD_STEP': 0.05,
       'MODEL_OUTPUT_DIR': args.output_dir
   }

   fine_tune_bert(args.data_path, config)