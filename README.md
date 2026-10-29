#  News Article Categorisation (NLP Mini-Project)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Colab-Ready%20to%20Run-orange.svg)](https://colab.research.google.com/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML-blue.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

An end-to-end Natural Language Processing (NLP) mini-project for classifying news articles and headlines into **4 categories**:
-  **Sports**
-  **Education**
-  **Politics**
-  **Technology**


---

## Project Overview & Pipeline

The project implements a complete NLP text classification workflow across a sequence of structured Google Colab notebooks:

```text
[Raw News Articles / Headlines]
       ↓
[1. Data Collection & Augmentation] ── (AG News dataset + Indian news augmentation)
       ↓
[2. Exploratory Data Analysis] ── (Class balance, token lengths, N-gram frequencies & word clouds)
       ↓
[3. Text Preprocessing] ── (Lowercasing, punctuation/number cleanup, stopword removal, spaCy lemmatization)
       ↓
[4. Feature Extraction & Classical ML] ── (TF-IDF with Unigram + Bigram features)
   ├── Multinomial Naive Bayes (Baseline)
   ├── Logistic Regression (One-vs-Rest)
   └── Support Vector Machine (LinearSVC with Platt Scaling)
       ↓
[5. Deep Learning with Transformers] ── (DistilBERT WordPiece Tokenization & Fine-Tuning)
       ↓
[6. Evaluation & Interactive Error Analysis] ── (Accuracy, Precision, Recall, Macro F1 & Confusion Matrices)
```

---

##  Model Comparison & Results

All models were evaluated on an 80/20 stratified test split:

| Model | Feature Extraction | Accuracy (%) | Macro F1 (%) | Key Advantage |
|---|---|---|---|---|
| **Multinomial Naive Bayes** | TF-IDF (1–2 n-grams) | ~86.5% | ~85.9% | Ultra-fast baseline; works with limited data |
| **Logistic Regression** | TF-IDF (1–2 n-grams) | ~91.2% | ~90.8% | High interpretability with calibrated probabilities |
| **Support Vector Machine (LinearSVC)** | TF-IDF (1–2 n-grams) | ~93.1% | ~92.8% | Maximum-margin boundary in high-dimensional text space |
| **DistilBERT (Fine-Tuned)** | Transformer WordPiece | **~96.4%** | **~96.1%** | True bidirectional context; understands subtle semantics |

---

##  Repository Structure

```text
├── 01_data_collection.ipynb       # AG News dataset filtering & Indian news augmentation
├── 02_data_exploration.ipynb      # Class balance checks, length analysis & word clouds
├── 03_preprocessing.ipynb         # Step-by-step cleaning (lowercase, stopwords, lemmatization)
├── 04_modelling.ipynb             # TF-IDF vectorization, NB, Logistic Regression, SVM training
├── 05_demo.ipynb                  # Interactive prediction pipeline, batch tests & viva guide
├── 06_distilbert_finetuning.ipynb # DistilBERT transformer fine-tuning with Hugging Face & PyTorch
├── requirements.txt               # Project Python dependencies
├── .gitignore                     # Git ignore rules for datasets, models, and cache
└── README.md                      # Project documentation
```

---

##  How to Run the Notebooks

### Running in Google Colab (Recommended)
Each notebook is self-contained and designed to run directly in **Google Colab**:

1. Open [Google Colab](https://colab.research.google.com/).
2. Open or upload the notebooks in sequential order (`01` ➔ `02` ➔ `03` ➔ `04` ➔ `05` ➔ `06`).
3. For **Notebook 06 (DistilBERT)**, make sure to enable the free GPU:
   ```text
   Runtime ➔ Change runtime type ➔ Hardware accelerator: T4 GPU ➔ Save
   ```

### Running Locally (Jupyter Notebook / VS Code)
```bash
# Clone the repository
git clone https://github.com/<your-username>/news-article-categorisation.git
cd news-article-categorisation

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Launch Jupyter
jupyter notebook
```

---

##  Error Analysis & Key Observations

During model evaluation, specific inter-category confusions were systematically analyzed:
1. **Politics ↔ Education:** Government education policies (e.g., Union Budget allocations, National Education Policy, university funding) share vocabulary with political governance.
2. **Technology ↔ Education:** Online learning platforms, EdTech startups, and university AI degree announcements trigger strong signals from both categories.
3. **Contextual Advantage of DistilBERT:** While classical TF-IDF models often misclassified multi-domain headlines, DistilBERT utilized bidirectional attention across the whole sentence to correctly determine the core subject.

---
