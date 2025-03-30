# Created by rishijeet at 30/03/25
import re
from collections import Counter

# Preprocess the Data
def preprocess_text(text):
    """
    Cleans and tokenizes the input text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    return tokens

def build_vocab(texts, min_freq=5):
    """
    Builds a vocabulary from the tokenized texts.
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(text)
    # Filter out words below the minimum frequency
    vocab = {word: i for i, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    return vocab

def text_to_sequence(text, vocab):
    """
    Converts a text (list of tokens) into a sequence of vocabulary indices.
    """
    return [vocab.get(word, len(vocab)) for word in text] # Use len(vocab) for unknown words
