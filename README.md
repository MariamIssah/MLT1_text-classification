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

- **Traditional ML (SVM)**: One team member implements **Support Vector Classification** (linear kernel, `class_weight='balanced'`). Hyperparameters (e.g. `C`) are tuned via GridSearchCV (e.g. 2-fold in fast mode). Three pipelines: SVM on TF-IDF, on Skip-gram, on CBOW — same splits and metrics.
- **Sequence models (LSTM, GRU)**: Other members implement **LSTM** and **GRU** for sequence classification. Each model is trained separately with TF-IDF, Skip-gram, and CBOW (and optionally GloVe in LSTM). Same dataset and shared preprocessing; split and number of classes may differ per notebook (e.g. 10-class vs many-class setup) as documented in each notebook.

For each model–embedding combination we train, tune where applicable, and evaluate on a held-out test set.

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
├── notebooks/                          # Separate implementations per model
│   ├── text_classification_svm.ipynb  # Traditional ML: SVM with TF-IDF, Skip-gram, CBOW
│   ├── lstm_model.ipynb               # LSTM experiments
│   └── gru_notebook.ipynb             # GRU experiments
├── outputs/
│   └── tables/                        # Result tables (e.g. JSON) for report
├── models/                             # Saved models (optional; not in Git; regenerate via notebooks)
├── requirements.txt
└── README.md
```

**Rubric alignment**: Separate implementations for each model; embeddings kept consistent (TF-IDF, Skip-gram, CBOW) for fair comparison.

---

## 4. Dataset Exploration, Preprocessing & Embedding Strategy

- **Exploratory data analysis (EDA)** is in the SVM notebook (`notebooks/text_classification_svm.ipynb`), including:
  - Label distribution (top classes)
  - Text length (word count) distribution
  - Train / validation / test split sizes
  - Class balance and missing values
- **Preprocessing** (shared): lowercasing, removal of non-alphabetic characters, tokenization (NLTK), stopword removal, lemmatization. Implemented in `utils/preprocessing.py` and used consistently.
- **Embedding-specific adaptation**:
  - **TF-IDF**: Preprocessed text as space-joined strings; `TfidfVectorizer` with configurable `max_features`, `ngram_range`, `sublinear_tf`.
  - **Skip-gram / CBOW**: Same tokenized output from the shared preprocessor; Word2Vec (gensim) with configurable `vector_size`, `window`, `min_count`, `epochs`. Document vectors by averaging word vectors.

---

## 5. Model Implementation & Experimental Design (This Repo: SVM)

- **Model**: Support Vector Machine (SVC, scikit-learn).
- **Embeddings**: **TF-IDF**, **Skip-gram (Word2Vec)**, **CBOW (Word2Vec)** — at least three as required.
- **Hyperparameter tuning**: GridSearchCV over `C`, `kernel` (e.g. linear), `class_weight`; 2-fold CV in fast mode, more folds when disabled.
- **Training**: Each embedding produces a separate pipeline (fit on same train/val/test splits). All choices (splits, metrics, preprocessing) are documented in the notebook.

---

## 6. Experiment Tables & Results

- **Tables**: The SVM notebook produces:
  - **Performance comparison table** on the test set (accuracy, precision_macro, recall_macro, f1_macro, precision_weighted, recall_weighted, f1_weighted) for TF-IDF, Skip-gram, CBOW.
  - **Best hyperparameters** per embedding (in the same results structure).
- **Visualizations**: Multiple figures support the report:
  - Bar charts: accuracy, precision, recall, F1 across embeddings
  - Grouped bars: all metrics by embedding; heatmap (embeddings × metrics)
  - Validation vs test accuracy and F1 (generalization)
  - Train vs validation scatter (overfitting check)
  - Confusion matrices (per embedding); per-class F1; most confused class pairs
- **Metrics**: Accuracy, macro/weighted F1, precision, recall, classification report, confusion matrix — aligned with the group’s evaluation plan.

---

## 7. How to Run (Reproducibility)

1. **Environment**
   ```bash
   pip install -r requirements.txt
   ```
2. **NLTK data** (once): `punkt_tab` (or `punkt`), `stopwords`, `wordnet` 
3. **SVM notebook**: Open `notebooks/text_classification_svm.ipynb` from the **project root** so paths to `data/`, `embeddings/`, `utils/` are correct. Run all cells. Use `FAST_MODE = True` for a quicker run.
4. **Saved embeddings**: Optional reuse via `embeddings/saved/` (see notebook for load examples). Models can be re-saved under `models/` (folder is in .gitignore for size).

---

## 8. Deliverables (Assignment Requirements)

| Deliverable | Status |
|-------------|--------|
| **Research report (PDF)** | See submission; includes link to this GitHub repo. |
| **GitHub repo** | This repository — code, README, and structure as required. |
| **Comparative analysis** | Report contains comparison tables and discussion across embeddings and models. |
| **Code quality** | Modular code, shared preprocessing, separate notebooks per model, and this README. |

---

## 9. Individual Technical Contribution (SVM Member)

- **Assigned model**: Traditional ML — **SVM**.
- **Embeddings used**: **TF-IDF**, **Skip-gram (Word2Vec)**, **CBOW (Word2Vec)** (three required).
- **Scope**: Full implementation in `notebooks/text_classification_svm.ipynb`: data load, EDA, preprocessing, embedding fit/transform, SVM training with tuning, evaluation (tables and figures), overfitting check, and error analysis. Results are suitable for inclusion in the group report and comparison tables.

**Contribution tracker**: The PDF report includes the group contribution statement; this README summarizes the SVM component only.

---

## 10. References & Citation

Methodology and embedding choices are justified in the **report** with references (e.g. TF-IDF, Word2Vec, SVM for text). This README does not duplicate the references section; see the submitted PDF.

---

## 11. License & Contact

For course use only. Contact the team via the repository or the report for questions.
