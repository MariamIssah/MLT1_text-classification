"""
CBOW (Word2Vec) Embedding Module
Reusable for all team members
"""
import numpy as np
from gensim.models import Word2Vec
import os


class CBOWEmbedding:
    """
    CBOW Word2Vec embedding class that can be saved and loaded
    for consistent use across team members
    """
    
    def __init__(self, vector_size=300, window=5, min_count=2, workers=4, epochs=10):
        """
        Initialize CBOW Word2Vec model
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum word count threshold
            workers: Number of worker threads
            epochs: Number of training epochs
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.model = None
        self.is_fitted = False
    
    def fit(self, tokenized_texts):
        """
        Train CBOW Word2Vec model
        
        Args:
            tokenized_texts: List of lists of tokens (preprocessed)
        """
        print("Training CBOW Word2Vec model...")
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=0,  # 0 for CBOW
            epochs=self.epochs
        )
        self.is_fitted = True
        print(f"CBOW model trained. Vocabulary size: {len(self.model.wv)}")
    
    def transform(self, tokenized_texts):
        """
        Transform tokenized texts to document vectors (average of word vectors)
        
        Args:
            tokenized_texts: List of lists of tokens
            
        Returns:
            Document embedding matrix (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        embeddings = []
        for tokens in tokenized_texts:
            word_vectors = []
            for token in tokens:
                if token in self.model.wv:
                    word_vectors.append(self.model.wv[token])
            
            if word_vectors:
                # Average word vectors to get document vector
                doc_vector = np.mean(word_vectors, axis=0)
            else:
                # Zero vector if no words found
                doc_vector = np.zeros(self.vector_size)
            
            embeddings.append(doc_vector)
        
        return np.array(embeddings)
    
    def fit_transform(self, tokenized_texts):
        """
        Fit and transform
        
        Args:
            tokenized_texts: List of lists of tokens
            
        Returns:
            Document embedding matrix (numpy array)
        """
        self.fit(tokenized_texts)
        return self.transform(tokenized_texts)
    
    def save(self, filepath):
        """
        Save the trained CBOW model
        
        Args:
            filepath: Path to save the model (e.g., 'embeddings/cbow_model.bin')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"CBOW model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a saved CBOW model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = Word2Vec.load(filepath)
        self.vector_size = self.model.wv.vector_size
        self.is_fitted = True
        print(f"CBOW model loaded from {filepath}")
        print(f"Vocabulary size: {len(self.model.wv)}")
    
    def get_word_vector(self, word):
        """Get vector for a single word"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        return self.model.wv[word] if word in self.model.wv else None