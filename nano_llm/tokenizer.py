"""
A tokenizer that maps characters to integers IDS.
"""

class CharTokenizer:
    def __init__(self, text: str):
        """
        Initializes the tokenizer.

        Attributes:
            chars (list): Sorted list of unique characters found in the training text.
            vocab_size (int): Total number of unique characters in the vocabulary.
            stoi (dict): Mapping from character to integer index.
            itos (dict): Mapping from integer index to character.
        """
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        """
        Converts a string into a list of integer token IDz.
        """
        return [self.stoi[c] for c in s]
    
    def decode(self, l: list[int]) -> str:
        """
        Convert a list of integer token IDs back into readable string.
        """
        return ''.join([self.itos[i] for i in l])