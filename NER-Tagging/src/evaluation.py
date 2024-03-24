# src/evaluation.py

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Optional, Dict
from data import build_vocab
import os

data_files = ['../../data/lstm-data/train', '../../data/lstm-data/dev']
word_vocab, tag_vocab = build_vocab(data_files)


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float, float, float]:
    """Evaluate the model on the given dataloader using the specified criterion.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing the evaluation data.
        criterion (nn.Module): The loss criterion.
        device (torch.device): The device (CPU or GPU) to perform evaluation on.

    Returns:
        Tuple[float, float, float, float, float]: A tuple containing the average loss, accuracy, and F1 scores for macro, micro, and weighted averages.
    """

    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader: # (batch sz, seq len) 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(torch.max(outputs, dim=2))
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim = 2)
            all_preds.extend(preds.view(-1).cpu().numpy())
            all_labels.extend(labels.view(-1).cpu().numpy())

    valid_indices = [i for i, label in enumerate(all_labels) if label != -1]
    valid_preds = [all_preds[i] for i in valid_indices]
    valid_labels = [all_labels[i] for i in valid_indices]

    accuracy = np.mean(np.array(valid_preds) == np.array(valid_labels))
    f1_mac, f1_mic, f1_weight = compute_metrics(valid_preds, valid_labels)

    return total_loss / len(dataloader), accuracy, f1_mac, f1_mic, f1_weight


def get_eval_preds(model, file_paths, word_vocab, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), idx_to_tag={idx: tag for tag, idx in tag_vocab.items()}, output_dir='../../data/lstm-data/preds/', output_postfix = None):   
    """Generate predictions for evaluation using the provided model.

    Args:
        model (nn.Module): The trained model.
        file_paths (list[str]): A list of file paths containing evaluation data.
        word_vocab (dict[str, int]): A dictionary mapping words to their indices.
        device (torch.device, optional): The device (CPU or GPU) to perform evaluation on. Defaults to GPU if available, otherwise CPU.
        idx_to_tag (dict[int, str], optional): A dictionary mapping tag indices to tag names. Defaults to None.
        output_dir (str, optional): The directory to save prediction files. Defaults to '../../data/lstm-data/preds/'.
        output_postfix (str, optional): A postfix to append to the output file names. Defaults to None.

    Returns:
        list[str]: A list of file paths where the predictions are saved.
    """

    model.eval()
    model.to(device)
    output_paths = []

    for file_path in file_paths:
        # print(file_path)
        output_file_name = os.path.basename(file_path) + "_preds_" + output_postfix
        output_path = os.path.join(output_dir, 'new', output_file_name)
        output_paths.append(output_path)
        # print("error may be: ", file_path, 'Or: ', output_path)
        
        with open(file_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:     
            sentences = []
            current_sentence = []
            for line in f:
                if line.strip():  # if line contains stripable parts 
                    parts = line.strip().split()
                    original_word = parts[1]  
                    current_sentence.append(word_vocab.get(original_word, word_vocab['<UNK>']))
                elif current_sentence:  # Empty line and current sentence is not empty
                    sentences.append(current_sentence)
                    current_sentence = []

            # Add the last sentence if the file doesn't end with a newline
            if current_sentence:
                sentences.append(current_sentence)

            # Predict and write to file
            for sentence in sentences: 
                sentence_tensor = torch.tensor([sentence], dtype = torch.long, device = device)
                outputs = model(sentence_tensor)
                _, preds = torch.max(outputs, dim = 2)
                pred_tags = [idx_to_tag[pred.item()] for pred in preds[0]]  # Convert indices to tags


                # Prediction writing 
                for i, word_idx in enumerate(sentence):
                    word = list(word_vocab.keys())[list(word_vocab.values()).index(word_idx)]  # Inverse lookup
                    tag = pred_tags[i]
                    out_f.write(f"{i+1}\t{word}\t{tag}\n")
                out_f.write("\n")  # New line after each sentence
    return output_paths


def compute_metrics(preds: List[int], labels: List[int]) -> Tuple[float, float, float]:
    """Compute F1 scores for the given predictions and labels.

    Args:
        preds (List[int]): Predicted labels.
        labels (List[int]): True labels.

    Returns:
        Tuple[float, float, float]: A tuple containing F1 scores for macro, micro, and weighted averages.
    """

    f1_mac = f1_score(labels, preds, average='macro', zero_division = 0)
    f1_mic = f1_score(labels, preds, average='micro', zero_division = 0)
    f1_weight = f1_score(labels, preds, average='weighted', zero_division = 0)

    return f1_mac, f1_mic, f1_weight