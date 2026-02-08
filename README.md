# MLT1 Text Classification - Traditional ML Model (SVM)

This repository contains the implementation of **Support Vector Machine (SVM)** for text classification using multiple embedding techniques.

## Project Structure

```
MLT1_text-classification/
├── data/
│   └── pricerunner_aggregate.csv    # Dataset
├── embeddings/
│   ├── __init__.py
│   ├── tfidf_embedding.py          # TF-IDF embedding module
│   ├── skipgram_embedding.py       # Skip-gram embedding module
│   ├── cbow_embedding.py           # CBOW embedding module
│   └── saved/                       # Saved embedding models (generated)
├── utils/
│   └── preprocessing.py             # Text preprocessing utilities
├── text_classification_svm.ipynb   # Main notebook
├── requirements.txt                 # Dependencies
└── README.md
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download NLTK data (will be done automatically on first run):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

1. Open `text_classification_svm.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load and preprocess the dataset
   - Create and save embeddings (TF-IDF, Skip-gram, CBOW)
   - Train SVM with hyperparameter tuning for each embedding
   - Evaluate and compare results

## Embeddings for Team Members

The embeddings are saved in `embeddings/saved/` and can be reused by other team members:

```python
from embeddings.tfidf_embedding import TFIDFEmbedding
from embeddings.skipgram_embedding import SkipGramEmbedding
from embeddings.cbow_embedding import CBOWEmbedding

# Load saved embeddings
tfidf = TFIDFEmbedding()
tfidf.load('embeddings/saved/tfidf_model.pkl')

skipgram = SkipGramEmbedding()
skipgram.load('embeddings/saved/skipgram_model.bin')

cbow = CBOWEmbedding()
cbow.load('embeddings/saved/cbow_model.bin')
```

## Results

The notebook generates:
- Performance comparison table
- Visualizations (bar charts, confusion matrices)
- Classification reports
- Best hyperparameters for each embedding

## Model: SVM

- **Algorithm**: Support Vector Machine
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Embeddings**: TF-IDF, Skip-gram (Word2Vec), CBOW (Word2Vec)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score (macro and weighted)