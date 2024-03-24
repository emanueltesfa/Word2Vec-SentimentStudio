# src/training.py

from evaluation import evaluate_model
from tqdm import tqdm
import torch
import torch.nn as nn 
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from typing import Tuple, List, Optional, Dict


def train_model(model: nn.Module, dataloaders: dict[str, DataLoader], optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, num_epochs: int = 50, patience: int = 10, scheduler: Optional[optim.lr_scheduler._LRScheduler] = None, ckpt_name: str = 'best_model.pth') -> Tuple[float, float, float]:
    """Train the model using the provided dataloaders, optimizer, and criterion.

    Args:
        model (nn.Module): The model to train.
        dataloaders (dict[str, DataLoader]): A dictionary containing DataLoader objects for training and validation datasets.
        optimizer (optim.Optimizer): The optimizer for updating the model's parameters.
        criterion (nn.Module): The loss criterion.
        device (torch.device): The device (CPU or GPU) to perform training on.
        num_epochs (int, optional): Number of epochs for training. Defaults to 50.
        patience (int, optional): Number of epochs to wait for improvement in validation F1 score before early stopping. Defaults to 10.
        scheduler (Optional[optim.lr_scheduler._LRScheduler], optional): Learning rate scheduler. Defaults to None.
        ckpt_name (str, optional): File name to save the best model checkpoint. Defaults to 'best_model.pth'.

    Returns:
        Tuple[float, float, float]: A tuple containing the accuracy on the validation set, accuracy on the training set, and the best validation F1 score achieved during training.
    """

    best_val_f1 = -float('inf')
    patience_counter = 0
    model.to(device)

    for epoch in tqdm( range(num_epochs) ):
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            a_max, a_indx = torch.max(outputs, dim = 2)
            # print(torch.max(outputs, dim=2))
            # print('predition shape ',outputs.shape, 'label shape', labels.shape)
            # print('predition',outputs, 'label', labels) 
            loss = criterion(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        train_loss, train_acc, train_f1_mac, train_f1_mic, train_f1_weighted = evaluate_model(model, dataloaders['train'], criterion, device)
        val_loss, val_acc, val_f1_mac, val_f1_mic, val_f1_weighted = evaluate_model(model, dataloaders['dev'], criterion, device)

        if epoch % 5 == 0: 
            print(f'Epoch {epoch+1}:')
            print(f'Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f},  F1Mac: {train_f1_mac:.4f}')
            print(f'Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1Mac: {val_f1_mac:.4f}')

            
        # Early stopping based on validation F1 score
        if val_f1_mac > best_val_f1:
            best_val_f1 = val_f1_mac
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_name)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
    ### after saving location of model in ckpt_name
        #checkpoint = torch.load(ckpt_name)
        # model.load_state_dict(checkpoint)
    ### and run lower eval, retrieve predictions => return dev prediction file path custom/glove,
        # file_paths = ['../../data/lstm-data/dev']#, '../../data/lstm-data/dev']
        # word_vocab, _ = build_vocab(data_files)
        # output_paths = get_eval_preds(model, file_paths, word_vocab)
    ###  the run   # evaluate_fb1_model(preds_file_path, gold_file_path)
        # evaluate_fb1_model(output_paths, gold_file_path = '../../data/lstm-data/dev')
    return val_acc, train_acc, best_val_f1