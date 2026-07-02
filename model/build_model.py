from turtle import forward
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from pathlib  import Path

# root directory
ROOT_DIR = Path.cwd().parent
sys.path.append(ROOT_DIR)


from base_model.MLP import*
from base_model.ResNet_model import *
class ViCaptionEncoder(ResNet):
    def __init__(self, version: int = 34, num_classes=1000):
        super().__init__(version, num_classes)
        # bỏ lớp Layer cuối cùng
        self.model = self.model[:-1]
    def __call__(self,X):
        return self.forward(X)

class ViCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding_matrix = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn_layer = nn.LSTM(input_size=embedding_dim, hidden_size= hidden_size, num_layers=num_layers, batch_first=True)
    def forward(self,X, h0, c0):
        X_embedded = self.embedding_matrix(X)
        output,_ = self.rnn_layer(X_embedded, (h0, c0))
        return output


class ViCaptioningImgModel(nn.Module):
    def __init__(self, version_ResNet: int = 34, vocab_size: int = 500, embedding_dim: int =100, hidden_size: int =100, num_layer: int =1 ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = ViCaptionEncoder(version=34)
        self.decoder = ViCaptionDecoder(vocab_size= vocab_size, embedding_dim= embedding_dim, hidden_size= hidden_size, num_layers= num_layer)
        self.LinearF2h0 = nn.Linear(in_features= self.encoder.list_channel[-1], out_features= hidden_size)
        self.LinearF2c0 = nn.Linear(in_features= self.encoder.list_channel[-1], out_features= hidden_size)
    def forward(self, X, y):
        X, y = X.to(self.device), y = y.to(self.device)
        FeatureMaps = self.encoder(X)
        c0, h0 = self.LinearF2c0(FeatureMaps), self.LinearF2h0(FeatureMaps)
        logits = self.decoder(y, (h0, c0))
        return logits

class TrainModel:
    def __init__(self) -> None:
        pass
    def fit(self, model, X, y, dataset, X_val, y_val, val_dataset, n_epochs: int = 100, batch_size: int = 256, verbose: int = 2, is_shuffle : bool = True, lr: float = 0.01 , criterion, optimizer):
        if X