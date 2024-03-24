# src/utils.py

import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import numpy as np
from typing import Tuple, List , Optional, Dict
# from data import build_vocab

# data_files = ['../../data/lstm-data/train', '../../data/lstm-data/dev']
# word_vocab, tag_vocab = build_vocab(data_files)

def pad_collate(batch, word_vocab):
    sentences, tags = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=word_vocab['<PAD>'])
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)  # -1 or another index for padding in tags
    return sentences_padded, tags_padded


def get_class_weights(file_paths: list[str], tag_vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate class weights for a set of tags based on their frequency in the dataset.

    Args:
        file_paths (list[str]): A list of file paths containing the data.
        tag_vocab (dict[str, int]): A dictionary mapping tags to their indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - `regular_weights_tensor` (torch.Tensor): A tensor containing the regular class weights.
            - `inv_weights_tensor` (torch.Tensor): A tensor containing the inverse class weights.
    """

    tag_counts = Counter()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    _, _, tag = parts
                    if tag in tag_vocab:  
                        tag_counts[tag] += 1

    total_tags = sum(tag_counts.values())
    regular_weights = {tag: (count / total_tags) for tag, count in tag_counts.items()}
    inv_weights = {tag: (total_tags / (count + 1e-9)) for tag, count in tag_counts.items()} # Calculate inverse class weights (inversely proportional to frequency)

    num_tags = len(tag_vocab)
    regular_weights_tensor = torch.zeros(num_tags, dtype=torch.float) # Initialize weights tensors based on tag_vocab ordering
    inv_weights_tensor = torch.zeros(num_tags, dtype=torch.float)

    # Populate the tensors from tag_vocab indices
    for tag, idx in tag_vocab.items():
        regular_weights_tensor[idx] = regular_weights.get(tag, 0)
        inv_weights_tensor[idx] = inv_weights.get(tag, 0)

    return regular_weights_tensor, inv_weights_tensor


def load_glove_embeddings(path: str, word_vocab: dict[str, int], embedding_dim: int) -> torch.Tensor:
    """Load pre-trained GloVe embeddings from the specified path and create an embedding matrix.

    Args:
        path (str): The path to the GloVe embeddings file.
        word_vocab (dict[str, int]): A dictionary mapping words to their indices.
        embedding_dim (int): The dimensionality of the word embeddings.

    Returns:
        torch.Tensor: A tensor representing the embedding matrix.
    """

    embedding_dict = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]  
            vector = np.asarray(values[1:], "float32")
            embedding_dict[word] = vector
    
    vocab_size = len(word_vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    
    for word, idx in word_vocab.items():
        embedding_vector = embedding_dict.get(word, embedding_dict.get(word.lower()))
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
        else:
            embedding_matrix[idx] = np.random.randn(embedding_dim) 
    
    return torch.tensor(embedding_matrix, dtype=torch.float)
