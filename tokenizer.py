from nltk.tokenize import word_tokenize
import os
from transformers import BertTokenizer

class SimpleTokenizer:
    """
    A simple tokenizer class that builds a vocabulary from the given text and encodes/decodes text into indices.
    """

    def __init__(self, text):
        """Initialize the tokenizer with the initial text to build vocabulary."""
        self.vocab = set()
        self.stoi = {}
        self.itos = {}
        self.build_vocab(text)

    def build_vocab(self, text):
        """Build vocabulary from the given text."""
        tokens = word_tokenize(text)
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab) + 2
        self.stoi = {word: i for i, word in enumerate(self.vocab, start=2)}
        self.stoi['<pad>'] = 0
        self.stoi['<unk>'] = 1
        self.itos = {i: word for word, i in self.stoi.items()}

    def encode(self, text):
        """Encode the text into a list of indices."""
        tokens = word_tokenize(text)
        return [self.stoi.get(word, self.stoi['<unk>']) for word in tokens]

    def decode(self, indices):
        """Decode the list of indices back into text."""
        return ' '.join([self.itos.get(index, '<unk>') for index in indices])
    

class AdvancedTokenizer:
    """
    An advanced tokenizer class that uses BertTokenizer from the transformers library.
    """

    def __init__(self, model_name='bert-base-uncased'):
        """Initialize the tokenizer with a specified pre-trained model."""
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
    
    def encode(self, text, add_special_tokens=True):
        """Encode the text into a list of indices."""
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def decode(self, indices, skip_special_tokens=True):
        """Decode the list of indices back into text."""
        return self.tokenizer.decode(indices, skip_special_tokens=skip_special_tokens)
