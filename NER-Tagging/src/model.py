# src/models.py
import torch.nn as nn
import torch.nn.functional as F

class BILSTMForNER(nn.Module):
    """A Bidirectional LSTM model for Named Entity Recognition.

    Args:
        input_dim (int): The dimensionality of the input data.
        embedding_dim (int): The dimensionality of the word embeddings.
        hidden_dim (int): The dimensionality of the hidden state of the LSTM.
        output_dim (int): The dimensionality of the output.
        num_layers (int, optional): Number of LSTM layers. Defaults to 1.
        dropout (float, optional): Dropout probability. Defaults to 0.33.

    Returns:
        BILSTMForNER: A Bidirectional LSTM model for Named Entity Recognition.
    """
    
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, glove_embeddings = None, num_layers = 1, dropout = 0.33):
        super(BILSTMForNER, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        if glove_embeddings is not None:
            self.embedding.weight = nn.Parameter(glove_embeddings)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first = True, bidirectional = True, dropout = dropout)
        self.fc = nn.Linear(hidden_dim * 2, 128)
        self.classifier = nn.Linear(128, output_dim)
        self.elu = nn.ELU()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.elu(self.fc(lstm_out))  
        logits = self.classifier(out)  
        return logits  # batch, seq, embeddim