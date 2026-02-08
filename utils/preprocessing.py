"""
Text preprocessing utilities
Shared preprocessing pipeline for all team members
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

# Download required NLTK data
# For newer NLTK versions (3.8+), punkt_tab is required
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        # Try downloading punkt_tab first
        nltk.download('punkt_tab', quiet=True)
    except:
        # Fallback to punkt for older NLTK versions
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Text preprocessing pipeline - shared across team"""
    
    def __init__(self, min_word_length=2, remove_stopwords=True, lemmatize=True):
        """
        Initialize text preprocessor
        
        Args:
            min_word_length: Minimum word length to keep
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.min_word_length = min_word_length
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits (keep only letters and spaces)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return tokens
    
    def preprocess(self, text):
        """
        Full preprocessing pipeline
        
        Returns:
            List of preprocessed tokens
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(cleaned)
        
        # Filter by length and stopwords
        filtered_tokens = [
            token for token in tokens
            if len(token) >= self.min_word_length
            and (not self.remove_stopwords or token not in self.stop_words)
        ]
        
        # Lemmatize if enabled
        if self.lemmatize and self.lemmatizer:
            filtered_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        return filtered_tokens
    
    def preprocess_for_tfidf(self, text):
        """
        Preprocess text for TF-IDF (returns string)
        """
        tokens = self.preprocess(text)
        return ' '.join(tokens)
    
    def preprocess_for_embeddings(self, text):
        """
        Preprocess text for word embeddings (returns list of tokens)
        """
        return self.preprocess(text)