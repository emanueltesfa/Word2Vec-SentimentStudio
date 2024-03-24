# src/main.py
from model import BILSTMForNER
from data import build_vocab, IndexedNERDataset
from train import train_model
from utils import load_glove_embeddings, get_class_weights, pad_collate
from evaluation import evaluate_model, get_eval_preds, compute_metrics
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial

import warnings 
warnings.filterwarnings('ignore')

###  C:/Users/amant/anaconda3/envs/pytorch_dl/python.exe ./main.py
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    # Custom Embedding Model Training
    data_files = ['../../data/lstm-data/train', '../../data/lstm-data/dev']
    word_vocab, tag_vocab = build_vocab(data_files)
    input_dim = len(word_vocab)  
    output_dim = len(tag_vocab)

    regular_class_weight, inv_class_weight = get_class_weights(data_files, tag_vocab)
    regular_class_weight, inv_class_weight = regular_class_weight.to(device),  inv_class_weight.to(device)

    model = BILSTMForNER(input_dim = input_dim, embedding_dim = 100, hidden_dim = 256, dropout = 0.33, output_dim = output_dim)
    criterion = nn.CrossEntropyLoss(weight = inv_class_weight, ignore_index = tag_vocab.get('<PAD>', -1))  
    optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)

    pad_collate_with_vocab = partial(pad_collate, word_vocab = word_vocab)

    dataloaders = {
        'train': DataLoader(IndexedNERDataset('../../data/lstm-data/train', word_vocab, tag_vocab), batch_size = 16, shuffle = True, collate_fn = lambda batch: pad_collate_with_vocab(batch)),
        'dev': DataLoader(IndexedNERDataset('../../data/lstm-data/dev', word_vocab, tag_vocab), batch_size = 16, shuffle = False, collate_fn=lambda batch: pad_collate_with_vocab(batch)),
        # 'test': DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    }
    
    train_model(model, dataloaders, optimizer, criterion, device, ckpt_name = '../ckpts/custom_BiLSTM.pth', patience = 30, num_epochs = 100) # MODEL SAVED AT '/NER-Tagging/notebooks/ckpts/custom_BiLSTM.pth'


    ### GLOVE Model Training
    glove_path = '../../data/lstm-data/glove.6B.100d/glove.6B.100d.txt' 
    glove_embeddings = load_glove_embeddings(glove_path, word_vocab, embedding_dim = 100)

    glove_dataloaders = {
        'train': DataLoader(IndexedNERDataset('../../data/lstm-data/train', word_vocab, tag_vocab), batch_size = 32, shuffle = True, collate_fn=lambda batch: pad_collate_with_vocab(batch)),
        'dev': DataLoader(IndexedNERDataset('../../data/lstm-data/dev', word_vocab, tag_vocab), batch_size = 32, shuffle = False, collate_fn=lambda batch: pad_collate_with_vocab(batch)),
        # 'test': DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate)
    }

    glove_model = BILSTMForNER(input_dim = input_dim, embedding_dim = 100, hidden_dim = 256, glove_embeddings = glove_embeddings, dropout = 0.33, output_dim = output_dim)
    glove_criterion = nn.CrossEntropyLoss(weight = inv_class_weight, ignore_index = tag_vocab.get('<PAD>', -1))  
    glove_optimizer = optim.SGD(glove_model.parameters(), lr = 0.05, momentum = 0.9, weight_decay = 0.0001)

    train_model(glove_model, glove_dataloaders, glove_optimizer, glove_criterion, device, ckpt_name = '../ckpts/glove_BiLSTM.pth', patience = 50, num_epochs = 150)


    ### Make Predicition 
    checkpoint = torch.load('../ckpts/custom_BiLSTM.pth') 
    model.load_state_dict(checkpoint)
    get_eval_preds(model, data_files, word_vocab, output_postfix = 'custom_BiLSTM')

    checkpoint = torch.load('../ckpts/glove_BiLSTM.pth') 
    model.load_state_dict(checkpoint)
    get_eval_preds(model, data_files, word_vocab, output_postfix = 'glove_BiLSTM')


if __name__ == "__main__":
    main()