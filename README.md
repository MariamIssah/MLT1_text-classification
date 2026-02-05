# Comparative Analysis of Text Classification with Multiple Embeddings

## Project Overview
This project investigates the impact of different **word embedding techniques** on **text classification performance** across multiple model architectures. Each team member implements and evaluates **one classification model** using the **same dataset, preprocessing pipeline, and evaluation metrics**, enabling a fair and meaningful comparative analysis.

The study focuses on how traditional feature representations and distributed word embeddings interact with different model architectures, particularly sequence-based neural networks.

## Objectives
- Compare the performance of multiple **word embedding techniques** for text classification
- Analyze how embeddings interact with different **model architectures**
- Evaluate models using consistent metrics and experimental settings
- Produce a reproducible, well-documented academic-style study

## Dataset
**PriceRunner Aggregate Dataset**

- **Task**: Multiclass product category classification  
- **Text field**: Product Title  
- **Labels**: Product Category  
- **Source**: Aggregated e-commerce product listings  

The dataset is stored in the `data/` directory and is shared across all experiments to ensure consistency.

## Models and Embeddings

### Models
Each notebook focuses on **one model architecture**:
- Traditional Machine Learning model (e.g., Logistic Regression / SVM)
- RNN
- LSTM
- GRU

### Embeddings
Each model is evaluated using at least **three embedding techniques**:
- TF-IDF
- Word2Vec (Skip-gram)
- Word2Vec (CBOW)
- GloVe (pretrained)

Additional embeddings (e.g., FastText) may be included where applicable.

## Preprocessing
All notebooks follow a **shared preprocessing strategy**:
- Lowercasing text
- Removing non-alphabetic characters
- Tokenization
- Consistent train / validation / test split
- Shared tokenizer settings for embedding-based models

Embedding-specific adaptations are documented within each notebook and discussed in the final report.

## Evaluation Metrics
Models are evaluated using:
- Accuracy
- Macro F1-score
- Classification report
- Confusion matrix (visualized for selected models)

Results are saved in the `outputs/` directory and summarized in comparative tables in the final report.
