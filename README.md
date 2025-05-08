# Automated ICD-9 Coding from Clinical Text using Machine Learning

## Overview

This project explores the development and evaluation of various machine learning and deep learning models for automating the assignment of ICD-9 diagnosis codes to hospital discharge summaries. Using the publicly available MIMIC-III clinical database, specifically discharge summaries (`NOTEEVENTS`), this work focuses on applying Natural Language Processing (NLP) techniques to predict relevant ICD-9 codes based on unstructured clinical text.

The primary goals were:
1.  To investigate the feasibility of automating ICD coding using ML/DL.
2.  To compare the performance of traditional ML approaches (TF-IDF + Logistic Regression) against deep learning models (BiGRU, ClinicalBERT).
3.  To understand and address challenges inherent in this task, such as severe class imbalance and model evaluation in a multi-label setting.

## Dataset

* **Source:** [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/)
* **Input Text:** Discharge summaries from the `NOTEEVENTS.csv.gz` file.
* **Target Labels:** ICD-9 diagnosis codes from the `DIAGNOSES_ICD.csv.gz` file.
* **Preprocessing:**
   * Discharge summaries were extracted and linked to their corresponding hospital admissions (`HADM_ID`).
   * Text data was cleaned (lowercase, removal of specific patterns like `[**...**]`, punctuation, extra whitespace).
   * ICD-9 codes were aggregated per admission.
   * The analysis focused on predicting the **Top 100 most frequent** ICD-9 codes across the dataset.
   * Labels were multi-label binarized.
* **Access:** Note that access to the MIMIC-III database requires credentialing through PhysioNet.

## Methodology

Several models were implemented and compared for this multi-label text classification task:

1.  **TF-IDF + Logistic Regression:**
   * **Features:** TF-IDF vectors generated from the cleaned text, using uni-grams and bi-grams (`ngram_range=(1, 2)`) and limiting features (`max_features=20000`).
   * **Model:** `OneVsRestClassifier` wrapping `LogisticRegression`.
   * **Key Parameters:** `solver='liblinear'`, `max_iter=1000`, `class_weight='balanced'` (crucial for imbalance).

2.  **Bi-directional GRU (BiGRU):**
   * **Features:** Learned embeddings trained from scratch.
   * **Architecture:** Embedding Layer -> BiGRU Layer -> Dense Layers -> Sigmoid Output Layer.
   * **Framework:** TensorFlow/Keras.

3.  **ClinicalBERT (Feature Extraction):**
   * **Model:** Pre-trained `emilyalsentzer/Bio_ClinicalBERT` (via Hugging Face Transformers).
   * **Features:** Used the fixed `[CLS]` token embedding (768 dimensions) from the *frozen* BERT model as input to downstream classifiers.
   * **Classifier:** Simple Keras Sequential model (Dense -> Dropout -> Dense -> Sigmoid Output).
   * **Framework:** Hugging Face Transformers (PyTorch backend for extraction), TensorFlow/Keras for classifier.

4.  **ClinicalBERT (Fine-tuning):**
   * **Model:** Pre-trained `emilyalsentzer/Bio_ClinicalBERT` integrated into a Keras model using a custom layer, with BERT layers set to `trainable=True`.
   * **Architecture:** BERT Model -> `[CLS]` Token Output -> Dropout -> Dense Sigmoid Output Layer.
   * **Training:** Required small batch sizes (e.g., 8), low learning rates (e.g., 3e-5), and careful monitoring due to high computational cost. *(Class weights were prepared but not included in the final successful run shown in logs).*
   * **Framework:** Hugging Face Transformers, TensorFlow/Keras.

**Evaluation:** All models were evaluated using standard multi-label classification metrics, including Micro F1, Macro F1, Samples F1, and detailed per-class precision/recall/F1 scores. Prediction threshold optimization based on validation set Micro F1 was performed.

## Key Findings & Results

* **Strong Baseline Performance:** The TF-IDF + Logistic Regression model, particularly when using `class_weight='balanced'`, established a surprisingly strong baseline, significantly outperforming the initial deep learning models tested.
   * *Best LogReg Result:* Micro F1 ~0.51, **Macro F1 ~0.46** (with Threshold ≈ 0.55)
* **Deep Learning Performance:**
   * BiGRU and Frozen ClinicalBERT achieved similar, modest overall performance (Micro F1 ~0.35-0.36).
   * Initial Fine-tuning of ClinicalBERT (15 epochs, LR=3e-5, no class weights) did *not* show significant improvement over the frozen or BiGRU models (Micro F1 ~0.35, Macro F1 ~0.16).
   * All deep learning models struggled significantly with rarer classes, resulting in low Macro F1 scores (~0.16-0.18).
* **Importance of Imbalance Handling:** The success of the weighted Logistic Regression model highlights that directly addressing class imbalance is critical for this task, especially for improving Macro F1.
* **Threshold Optimization:** Optimizing the decision threshold significantly impacted results, often requiring low thresholds (e.g., 0.15) for deep learning models to balance precision/recall based on Micro F1 on validation data.
* **Fine-tuning Challenges:** Effective fine-tuning of large models like ClinicalBERT requires careful hyperparameter selection (esp. learning rate), longer training, potentially class weighting, and significant computational resources.

## Installation

1.  **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/[YourRepoName].git
   cd [YourRepoName]
   ```
2.  **Set up Environment:** It's recommended to use a virtual environment (like `conda` or `venv`).
   ```bash
   python -m venv venv
   source venv/bin/activate # Linux/macOS
   # venv\Scripts\activate # Windows
   ```
3.  **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Make sure to create a `requirements.txt` file with necessary libraries like: pandas, numpy, scikit-learn, tensorflow, transformers, torch [if needed for HF backend], matplotlib, tqdm)*

4.  **Dataset:** Obtain access to MIMIC-III v1.4 via PhysioNet and place the relevant `.csv.gz` files (e.g., `ADMISSIONS.csv.gz`, `DIAGNOSES_ICD.csv.gz`, `NOTEEVENTS.csv.gz`) in an accessible location (e.g., a `data/mimic_iii/` directory - **ensure this data is NOT committed to Git**). Update data paths in scripts if necessary.

## Usage

1.  **Data Preparation:** Run the data cleaning and preprocessing script (e.g., `python src/data/preprocess_mimic.py`) to generate the `merged_df` containing cleaned text and ICD code lists. (Ensure the output path is specified or the resulting DataFrame is available).
2.  **Model Training & Evaluation:**
   * Navigate to the relevant script (e.g., `src/models/train_logreg.py`, `src/models/train_bert_finetune.py`).
   * Adjust configuration parameters (like `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`) within the script if needed.
   * Run the script: `python src/models/train_bert_finetune.py`
   * The script will typically perform data loading (using the preprocessed data), model building, training, evaluation, and generate results/plots.

*(Note: Adapt script names and paths based on your actual project structure)*

## Visualizations

The scripts include code to generate visualizations to aid analysis:

* **Training History:** Plots loss, binary accuracy, AUC ROC, and AUC PR curves for training and validation sets over epochs.
   *(Placeholder for user-provided image `image_9073a0.png` showing training history)*
   ```
   [Insert Image: image_9073a0.png]
   *Caption: Example Training History Plot from ClinicalBERT Fine-tuning*
   ```
* **Threshold Optimization:** Shows Validation Micro F1 score versus different prediction thresholds.
* **F1 Score Distribution:** Displays a histogram and bar charts illustrating the range and distribution of per-class F1 scores.

## Future Work

* Apply **class weights** during ClinicalBERT fine-tuning to improve Macro F1.
* **Train fine-tuning model longer** with adjusted learning rates or schedulers.
* **Hyperparameter optimization** for both the TF-IDF/Logistic Regression model and the fine-tuned BERT model.
* Explore **hierarchical classification** approaches that leverage the ICD code structure.
* Experiment with different **pre-trained embeddings** (BioBERT, PubMedBERT) or embedding aggregation methods (e.g., mean/max pooling instead of just `[CLS]`).
* Investigate strategies for handling **texts longer than BERT's limit** (e.g., chunking, Longformer).

## License

Distributed under the MIT License.

## Acknowledgments

* This project utilizes the MIMIC-III database. Access requires credentialing via PhysioNet.
* Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). PhysioNet. https://doi.org/10.13026/C2XW26.
* Hugging Face Transformers library.

## Contact

[Emmanuel Leonce] - [deoleojr016@gmail.com] - [(https://www.linkedin.com/in/emmanuel-leonce-963bbb247/)]
Project Link: [Link to this GitHub repository]
