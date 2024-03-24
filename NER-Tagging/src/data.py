# src/data.py

import torch
from torch.utils.data import Dataset
from utils import pad_collate
import os


def build_vocab(data_files: list[str]) -> tuple[dict[str, int], dict[str, int]]:

    """Builds word and tag vocabularies from the given list of data files.

    Args:
        data_files (list): A list of file paths containing the data.

    Returns:
        tuple: A tuple containing two dictionaries: `word_vocab` and `tag_vocab`.
            - `word_vocab` (dict): A dictionary mapping words to their indices.
            - `tag_vocab` (dict): A dictionary mapping tags to their indices.
    """

    word_vocab = {'<PAD>': 0, '<UNK>': 1}
    tag_vocab = {}
    word_idx, tag_idx = 2, 0  

    for file_path in data_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    _, word, tag = line.split()
                    if word not in word_vocab:
                        word_vocab[word] = word_idx
                        word_idx += 1
                    if tag not in tag_vocab:
                        tag_vocab[tag] = tag_idx
                        tag_idx += 1
    return word_vocab, tag_vocab


class IndexedNERDataset(Dataset):

    """Dataset class for Indexed Named Entity Recognition.

    This class prepares data for Named Entity Recognition tasks by indexing words and tags.

    Attributes:
        word_vocab (dict[str, int]): A dictionary mapping words to their indices.
        tag_vocab (Optional[dict[str, int]]): A dictionary mapping tags to their indices if `use_tags` is True, otherwise None.
        use_tags (bool): Whether to include tags.
        data (list): A list to store the processed data.

    Methods:
        __init__: Initialize the dataset.
        _load_data: Load data from the dataset file.
        __len__: Return the number of data instances in the dataset.
        __getitem__: Retrieve a specific data instance from the dataset.
    """


    def __init__(self, file_path, word_vocab, tag_vocab = None, use_tags = True):
        self.word_vocab = word_vocab
        self.tag_vocab = tag_vocab if use_tags else None
        self.use_tags = use_tags
        self.data = []
        self._load_data(file_path)
        
    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentence = []
            for line in f:
                line = line.strip()
                if line:
                    if self.use_tags:
                        _, word, tag = line.split()
                        tag_idx = self.tag_vocab.get(tag, -1)  
                    else:
                        word = line
                        tag_idx = -1 
                    sentence.append((self.word_vocab.get(word, self.word_vocab['<UNK>']), tag_idx))
                else:
                    self.data.append(sentence)
                    sentence = []
            if sentence:  # Handle the case where the file doesn't end with a newline
                self.data.append(sentence)
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence, tags = zip(*self.data[idx])
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(tags, dtype=torch.long)