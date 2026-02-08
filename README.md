# Comparative Analysis of Text Classification with Multiple Embeddings

**Course assignment — Text Classification**  
This repository contains the **group project** code and supports the accompanying **research report (PDF)**. The report includes the full methodology, comparative analysis, and a **contribution tracker** for each team member.

---

## 1. Problem Definition & Dataset

### Classification task
Multiclass text classification: predict **product cluster/category** from **Product Title** (short text). The setup is suitable for comparing embedding–model combinations on real-world product data.

### Dataset: Product Classification and Clustering (PriceRunner)

- **Source**: Collected from [PriceRunner](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering), a product comparison platform. Donated 8/6/2023.
- **Instances**: 35,311 product offers.
- **Categories**: 10 categories; 306 merchants.
- **Characteristics**: Tabular, text; subject area: Business. Associated tasks: Classification, Clustering, Entity matching.
- **Preprocessing (original)**: Case folding and punctuation removal were applied to the titles (column 2). No missing values.
- **Location**: `data/pricerunner_aggregate.csv`

| Variable      | Role    | Type       | Description        |
|---------------|--------|------------|--------------------|
| Product ID    | Feature| Integer    | —                  |
| **Product Title** | Feature | Categorical | **Input text** |
| Merchant ID   | Feature| Integer    | —                  |
| Cluster ID    | Feature| Integer    | —                  |
| **Cluster Label** | Feature | Categorical | **Target label** |
| Category ID   | Feature| Integer    | —                  |
| Category Label| Feature| Categorical| —                  |

**Citation (introductory paper)**  
L. Akritidis, A. Fevgas, P. Bozanis, C. Makris (2020). *A self-verifying clustering approach to unsupervised matching of product titles.* Artificial Intelligence Review.

### Why this dataset (Team 11)

This dataset is relevant to Team 11 because most members work on product-related projects (e.g. products we sell or similar e-commerce settings). Although the exact products differ, the domain is product-oriented short text, so the methodology and embedding comparisons transfer well to our own use cases.

---

## 2. Methodology

### 2.1 Dataset and task

We use the **PriceRunner Product Classification and Clustering** dataset (35,311 product offers, 10 categories, 306 merchants). The **input** is the **Product Title** (short text); the **target** is the **Cluster Label** (product category). The task is **multiclass text classification**. Original data had case folding and punctuation removal on titles; we apply additional preprocessing as below.

### 2.2 Preprocessing (shared pipeline)

A single preprocessing pipeline is agreed upon and used across all model–embedding experiments to ensure fair comparison:

- **Cleaning**: Lowercasing; removal of URLs, emails, and non-alphabetic characters; normalization of whitespace.
- **Tokenization**: NLTK tokenizer (`punkt_tab` or `punkt`).
- **Filtering**: Minimum word length; optional stopword removal (NLTK English stopwords).
- **Lemmatization**: NLTK WordNet lemmatizer.

Implementation is in `utils/preprocessing.py`. The same pipeline produces (1) space-joined strings for TF-IDF and (2) token lists for Word2Vec-based embeddings (Skip-gram, CBOW).

### 2.3 Embedding strategy (per embedding type)

- **TF-IDF**: Preprocessed text as documents; scikit-learn `TfidfVectorizer` with configurable `max_features`, `ngram_range` (e.g. unigrams and bigrams), and `sublinear_tf`. No scaling before SVM (raw TF-IDF used).
- **Skip-gram (Word2Vec)**: Tokenized sentences; gensim Word2Vec with `sg=1`, configurable `vector_size`, `window`, `min_count`, `epochs`. Document representation: mean of word vectors (or zero vector if no known words).
- **CBOW (Word2Vec)**: Same tokenized input; Word2Vec with `sg=0`. Document vectors again by averaging word vectors.

Embeddings are fitted on the training (or full) text data; train/validation/test splits are transformed consistently. This design allows direct comparison of the same classifier (or same family of classifiers) across the three embedding types.

### 2.4 Models and experimental design

- **Traditional ML (SVM)**: **Support Vector Classification** (linear kernel, `class_weight='balanced'`). Hyperparameters (e.g. `C`) tuned via GridSearchCV. Three pipelines: SVM on TF-IDF, Skip-gram, CBOW — same splits and metrics.
- **RNN**: Recurrent neural network for sequence classification (notebook: `Thierry_SHYAKA_RNN_Product_Classification_ENHANCED (1).ipynb`). TF-IDF, Skip-gram, CBOW.
- **LSTM / GRU**: **LSTM** and **GRU** for sequence classification. Each is trained with TF-IDF, Skip-gram, CBOW (LSTM also supports GloVe). Same dataset and shared preprocessing; split and number of classes may differ per notebook as documented.

For each model–embedding combination we train, tune where applicable, and evaluate on a held-out test set. Results are saved to `outputs/tables/` as JSON for the **evaluation notebook**.

### 2.5 Evaluation

- **Metrics**: Accuracy; macro and weighted **precision**, **recall**, and **F1**; classification report; confusion matrix.
- **Splits**: Train / validation / test (e.g. 70% / 15% / 15%), with stratification when feasible. SVM notebook uses configurable class and sample caps (e.g. top 100 classes, 12k train samples) for reproducibility and runtime.
- **Analysis**: Comparison tables (performance across embeddings and, where applicable, across models); validation vs test curves or scatter (generalization); train vs validation scatter (overfitting); per-embedding confusion matrices and error analysis.

This methodology supports the comparative analysis required by the assignment and the structure of the final report.

---

## 3. Repository Structure (Code Organization)

```
MLT1_text-classification/
├── data/
│   └── pricerunner_aggregate.csv      # Shared dataset
├── embeddings/                         # Reusable embedding modules (≥3 types)
│   ├── tfidf_embedding.py             # TF-IDF
│   ├── skipgram_embedding.py          # Skip-gram (Word2Vec)
│   ├── cbow_embedding.py              # CBOW (Word2Vec)
│   └── saved/                         # Fitted embeddings (tfidf_model.pkl, skipgram_model.bin, cbow_model.bin)
├── utils/
│   └── preprocessing.py               # Shared preprocessing (tokenization, stopwords, lemmatization)
├── notebooks/                          # Separate implementations per model + evaluation
│   ├── text_classification_svm.ipynb  # SVM: TF-IDF, Skip-gram, CBOW
│   ├── Thierry_SHYAKA_RNN_Product_Classification_ENHANCED (1).ipynb  # RNN experiments
│   ├── lstm_model.ipynb               # LSTM: TF-IDF, Skip-gram, CBOW, GloVe
│   ├── gru_notebook.ipynb             # GRU: TF-IDF, Skip-gram, CBOW
│   └── evaluation_compare_models.ipynb # Compares all four models using outputs/tables/*.json
├── outputs/
│   └── tables/                        # Result JSONs: results_<model>_<embedding>.json (SVM, RNN, LSTM, GRU)
├── models/                             # Saved models (in .gitignore; regenerate via notebooks)
├── requirements.txt
└── README.md
```

**Rubric alignment**: Separate notebooks per model (SVM, RNN, LSTM, GRU); shared embeddings (TF-IDF, Skip-gram, CBOW) for fair comparison; single evaluation notebook aggregates results.

---

## 4. Dataset Exploration, Preprocessing & Embedding Strategy

- **Exploratory data analysis (EDA)** is in the SVM notebook (`notebooks/text_classification_svm.ipynb`): label distribution, text length distribution, train/val/test split sizes, class balance. Cross-model comparison is in `notebooks/evaluation_compare_models.ipynb`.
- **Preprocessing** (shared): lowercasing, removal of non-alphabetic characters, tokenization (NLTK), stopword removal, lemmatization. Implemented in `utils/preprocessing.py` and used consistently across notebooks.
- **Embedding-specific adaptation**:
  - **TF-IDF**: Preprocessed text as space-joined strings; `TfidfVectorizer` with configurable `max_features`, `ngram_range`, `sublinear_tf`.
  - **Skip-gram / CBOW**: Same tokenized output from the shared preprocessor; Word2Vec (gensim) with configurable `vector_size`, `window`, `min_count`, `epochs`. Document vectors by averaging word vectors.

---

## 5. Model Implementation & Experimental Design

This repo contains **four model implementations**:

| Notebook | Model | Embeddings |
|----------|--------|------------|
| `text_classification_svm.ipynb` | SVM (scikit-learn SVC, linear kernel) | TF-IDF, Skip-gram, CBOW |
| `Thierry_SHYAKA_RNN_Product_Classification_ENHANCED (1).ipynb` | RNN | TF-IDF, Skip-gram, CBOW |
| `lstm_model.ipynb` | LSTM | TF-IDF, Skip-gram, CBOW, GloVe |
| `gru_notebook.ipynb` | GRU | TF-IDF, Skip-gram, CBOW |

- **SVM**: GridSearchCV over `C`, `kernel`, `class_weight`; configurable class/sample caps (e.g. top 100 classes, 12k train samples) and FAST_MODE.
- **RNN / LSTM / GRU**: Sequence models; each notebook fits embeddings and trains on the same dataset with shared preprocessing where applicable. Results are written to `outputs/tables/results_<model>_<embedding>.json` for the evaluation notebook.

---

## 6. Experiment Tables & Results

- **Result files**: Each model notebook writes JSON files to `outputs/tables/` with the naming pattern `results_<model>_<embedding>.json` (e.g. `results_svm_tfidf.json`, `results_lstm_skipgram.json`). Each JSON includes at least `model`, `embedding`, `accuracy`, and optionally `precision_macro`, `recall_macro`, `f1_macro`, `train_time_sec`.
- **Evaluation notebook** (`evaluation_compare_models.ipynb`): Loads all available result JSONs, reports which models are present or missing, and produces:
  - Pivot tables (accuracy and F1 macro by model × embedding)
  - Bar charts: accuracy and F1 by embedding, grouped by model (order: SVM, RNN, LSTM, GRU)
  - Heatmap and ranking visuals
- **Per-notebook visuals** (e.g. SVM notebook): Performance comparison table, grouped bars (all metrics), heatmap, validation vs test, train vs validation scatter, confusion matrices, per-class F1, confused pairs.
- **Metrics**: Accuracy; macro/weighted precision, recall, F1; classification report; confusion matrix — aligned across notebooks.

---

## 7. How to Run (Reproducibility)

1. **Environment**
   ```bash
   pip install -r requirements.txt
   ```
2. **NLTK data** (once): `punkt_tab` (or `punkt`), `stopwords`, `wordnet`.
3. **Run from project root** so paths to `data/`, `embeddings/`, `utils/` resolve correctly.
4. **Model notebooks** (run in any order; each writes to `outputs/tables/`):
   - `notebooks/text_classification_svm.ipynb` — SVM (use `FAST_MODE = True` for a quicker run).
   - `notebooks/Thierry_SHYAKA_RNN_Product_Classification_ENHANCED (1).ipynb` — RNN.
   - `notebooks/lstm_model.ipynb` — LSTM.
   - `notebooks/gru_notebook.ipynb` — GRU.
5. **Evaluation**: Run `notebooks/evaluation_compare_models.ipynb` after at least one model notebook has produced JSONs in `outputs/tables/`. It will list models found and missing and plot comparisons (SVM, RNN, LSTM, GRU).
6. **Saved artifacts**: Embeddings in `embeddings/saved/`; trained models in `models/` (in .gitignore; regenerate via notebooks).

---

## 8. Deliverables (Assignment Requirements)

| Deliverable | Status |
|-------------|--------|
| **Research report (PDF)** | See submission; includes link to this GitHub repo. |
| **GitHub repo** | This repository — code, README, and structure as required. |
| **Comparative analysis** | Report contains comparison across embeddings and models; `evaluation_compare_models.ipynb` compares SVM, RNN, LSTM, GRU using `outputs/tables/*.json`. |
| **Code quality** | Modular code, shared preprocessing, separate notebooks per model (SVM, RNN, LSTM, GRU), single evaluation notebook, and this README. |

---

## 9. Individual Technical Contribution

- **SVM** (`text_classification_svm.ipynb`): Full pipeline — data load, EDA, preprocessing, TF-IDF/Skip-gram/CBOW embeddings, SVM training and tuning, evaluation tables and figures (grouped bars, heatmap, validation vs test, overfitting check, confusion matrices, per-class F1, confused pairs). Writes `results_svm_*.json` to `outputs/tables/`.
- **RNN** (`Thierry_SHYAKA_RNN_Product_Classification_ENHANCED (1).ipynb`): RNN experiments; results in `outputs/tables/results_rnn_*.json`.
- **LSTM** (`lstm_model.ipynb`): LSTM with TF-IDF, Skip-gram, CBOW, GloVe; writes `results_lstm_*.json`.
- **GRU** (`gru_notebook.ipynb`): GRU with TF-IDF, Skip-gram, CBOW; writes `results_gru_*.json`.

The **evaluation notebook** aggregates all four models for the comparative analysis in the report. The PDF report includes the group contribution statement; this README summarizes the repo contents.

---

## 10. References & Citation

Methodology and embedding choices are justified in the **report** with references (e.g. TF-IDF, Word2Vec, SVM for text). This README does not duplicate the references section; see the submitted PDF.

---

## 11. License & Contact

For course use only. Contact the team via the repository or the report for questions.
