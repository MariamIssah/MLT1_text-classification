"""
TF-IDF Embedding Module
Reusable for all team members
"""
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFEmbedding:
    """
    TF-IDF embedding class that can be saved and loaded
    for consistent use across team members
    """
    
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95):
        """
        Initialize TF-IDF vectorizer
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to use (default: unigrams and bigrams)
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=False,  # Assume text is already preprocessed
            min_df=min_df,
            max_df=max_df
        )
        self.is_fitted = False
    
    def fit(self, texts):
        """
        Fit TF-IDF vectorizer on texts
        
        Args:
            texts: List of preprocessed text strings
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"TF-IDF fitted on {len(texts)} documents")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF vectors
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            TF-IDF feature matrix (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transformation")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts):
        """
        Fit and transform texts
        
        Args:
            texts: List of preprocessed text strings
            
        Returns:
            TF-IDF feature matrix (numpy array)
        """
        return self.vectorizer.fit_transform(texts).toarray()
    
    def save(self, filepath):
        """
        Save the fitted TF-IDF model
        
        Args:
            filepath: Path to save the model (e.g., 'embeddings/tfidf_model.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df
            }, f)
        print(f"TF-IDF model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a saved TF-IDF model
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.vectorizer = data['vectorizer']
        self.max_features = data['max_features']
        self.ngram_range = data['ngram_range']
        self.min_df = data['min_df']
        self.max_df = data['max_df']
        self.is_fitted = True
        print(f"TF-IDF model loaded from {filepath}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def get_feature_names(self):
        """Get feature names"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        return self.vectorizer.get_feature_names_out()