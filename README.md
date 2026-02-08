# Comparative Analysis of Text Classification with Multiple Embeddings

## Project Overview
This project investigates the impact of different **word embedding techniques** on **text classification performance** across multiple model architectures. Each team member implements and evaluates **one classification model** using the **same dataset, preprocessing pipeline, and evaluation metrics**, enabling a fair and meaningful comparative analysis.

This repository contains the **Traditional ML (SVM)** implementation: **Support Vector Machine** for text classification using TF-IDF, Skip-gram, and CBOW embeddings.

---

## Objectives
- Compare the performance of multiple **word embedding techniques** for text classification
- Analyze how embeddings interact with different **model architectures**
- Evaluate models using consistent metrics and experimental settings
- Produce a reproducible, well-documented academic-style study

---

## Dataset
**PriceRunner Aggregate Dataset**
- **Task**: Multiclass product category classification  
- **Text field**: Product Title  
- **Labels**: Cluster Label (product category)  
- **Source**: Aggregated e-commerce product listings  

The dataset is stored in the `data/` directory.

---

## This Repo: SVM Model

### Project Structure

```
MLT1_text-classification/
├── data/
│   └── pricerunner_aggregate.csv    # Dataset
├── embeddings/
│   ├── __init__.py
│   ├── tfidf_embedding.py            # TF-IDF embedding module
│   ├── skipgram_embedding.py         # Skip-gram embedding module
│   ├── cbow_embedding.py             # CBOW embedding module
│   └── saved/                        # Saved embedding models (generated)
├── utils/
│   └── preprocessing.py             # Text preprocessing utilities
├── text_classification_svm.ipynb     # Main notebook (local)
├── text_classification_svm_kaggle.ipynb  # Kaggle-ready notebook
├── requirements.txt
└── README.md
```

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. NLTK data (run once; or the notebook will prompt):
```python
import nltk
nltk.download('punkt_tab')  # or 'punkt'
nltk.download('stopwords')
nltk.download('wordnet')
```

### Usage

1. Open `text_classification_svm.ipynb` and run all cells, or use `text_classification_svm_kaggle.ipynb` on Kaggle.
2. Set `FAST_MODE = True` for a quicker run (fewer classes, smaller grid); set `False` for full data and tuning.
3. The notebook loads data, builds embeddings (TF-IDF, Skip-gram, CBOW), trains SVM with tuning, and evaluates.

### Embeddings for Team Members

Saved embeddings in `embeddings/saved/` can be reused:

```python
from embeddings.tfidf_embedding import TFIDFEmbedding
from embeddings.skipgram_embedding import SkipGramEmbedding
from embeddings.cbow_embedding import CBOWEmbedding

tfidf = TFIDFEmbedding()
tfidf.load('embeddings/saved/tfidf_model.pkl')
skipgram = SkipGramEmbedding()
skipgram.load('embeddings/saved/skipgram_model.bin')
cbow = CBOWEmbedding()
cbow.load('embeddings/saved/cbow_model.bin')
```

### Model and Evaluation

- **Algorithm**: Support Vector Machine (SVC)
- **Hyperparameter Tuning**: GridSearchCV (C, kernel, class_weight; full grid when `FAST_MODE=False`)
- **Embeddings**: TF-IDF, Skip-gram (Word2Vec), CBOW (Word2Vec)
- **Metrics**: Accuracy, Precision, Recall, F1 (macro and weighted), classification report, confusion matrices

---

## Preprocessing (Shared)
- Lowercasing, remove non-alphabetic characters, tokenization
- Consistent train / validation / test split
- NLTK stopwords and lemmatization

---

## Evaluation Metrics (Group)
- Accuracy
- Macro F1-score
- Classification report
- Confusion matrix

Results from this notebook can be combined with other members’ results for the comparative report.
