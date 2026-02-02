import torch
from torch.utils.data import Dataset
from nano_llm.tokenizer import CharTokenizer

class TextDataset(Dataset):
    """
    Dataset for language modeling that provides shifted sequences.

    Takes a raw text file and prepares chunks of data where the target sequence (y)
    is the input sequence (x) shifted by one position.
    """
    def __init__(self, data_path: str, block_size: int, tokenizer: CharTokenizer):
        """
        data_path (str): Path to the .txt file containing the training data.
        block_size (int): The context window size (max sequence length).
        tokenizer (CharTokenizer): An instance of the tokenizer to encode the text.
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        """
        Returns the total number of possible sequences that can be extracted.
        """
        return len(self.data) - self.block_size

    def __getitem__(self, index):
        """
        Retrieves a single pair of (input, target) sequences.
        """
        chunk = self.data[index: index + self.block_size + 1]
        x = chunk[: -1]
        y = chunk[1:]
        return x, y
