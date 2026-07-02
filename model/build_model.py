from turtle import forward
from sympy import evaluate
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from pathlib  import Path
from torch.utils.data import TensorDataset
from torch.amp import GradScaler

# root directory
ROOT_DIR = Path.cwd().parent
sys.path.append(str(ROOT_DIR))


from base_model.MLP import*
from base_model.ResNet_model import *
class ViCaptionEncoder(ResNet):
    def __init__(self, version: int = 34, num_classes=1000):
        super().__init__(version, num_classes)
        # bỏ lớp Layer cuối cùng
        self.model = self.model[:-1]
class ViCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding_matrix = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn_layer = nn.LSTM(input_size=embedding_dim, hidden_size= hidden_size, num_layers=num_layers, batch_first=True)
        self.out2class = nn.Linear(in_features = hidden_size, out_features  = vocab_size)
    def forward(self,X, h0, c0):
        X_embedded = self.embedding_matrix(X)
        output,_ = self.rnn_layer(X_embedded, (h0, c0))
        logits = self.out2class(output)
        return logits


class ViCaptioningImgModel(nn.Module):
    def __init__(self, version_ResNet: int = 34, vocab_size: int = 500, embedding_dim: int =100, hidden_size: int =100, num_layer: int =1 ):
        super().__init__()
        self.encoder = ViCaptionEncoder(version=34)
        self.decoder = ViCaptionDecoder(vocab_size= vocab_size, embedding_dim= embedding_dim, hidden_size= hidden_size, num_layers= num_layer)
        self.LinearF2h0 = nn.Linear(in_features= self.encoder.list_channel[-1], out_features= hidden_size)
        self.LinearF2c0 = nn.Linear(in_features= self.encoder.list_channel[-1], out_features= hidden_size)
        self.num_layer = num_layer
    def forward(self, X, y):
        FeatureMaps = self.encoder(X)
        c0, h0 = self.LinearF2c0(FeatureMaps), self.LinearF2h0(FeatureMaps)
        # Thêm một chiều vào dim =0 cho đúng đầu vào của LSTM, RNN
        c0 = c0.unsqueeze(0)
        h0 = h0.unsqueeze(0)

        c0 = c0.repeat(self.num_layer, 1, 1)
        h0 = h0.repeat(self.num_layer, 1, 1)
        logits = self.decoder(y, h0, c0)
        #  giống như tensor.view() -> thay đổi đúng về shape quy định (Batch_size, C, ...)
        # output của LSTM có shape(N, L, hidden_size)
        return logits.permute(0, 2, 1)

class TrainModel:
    def __init__(self) -> None:
        self.dataset = None
        self.val_dataset = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Losses = []
        self.Accuracies =[]
        self.Val_Losses = []
        self.Val_Accuracies = []
        self.scaler = GradScaler()
    def get_accuracy(self, logits, y):
        return torch.mean((torch.argmax(logits, dim = 1) == y).float())
    def evaluate(self, X_val, y_val):
        with torch.no_grad():
            logits = self.model(X_val, y_val)
            loss = self.criterion(logits, y_val)
            acc = self.get_accuracy(logits, y_val)
        return loss.item(), acc.item()

    def fit(self, model = None, X = None, y = None, dataset = None, X_val = None, y_val = None, val_dataset = None, n_epochs: int = 100, batch_size: int = 256, is_shuffle : bool = True, lr: float = 0.01 , criterion = None, optimizer = None):
        if model is None:
            raise ValueError(f"TrainModel.fit require the parameter model !")
        else:
            self.model = model.to(self.device)

        # train set
        if X is not None and y is not None:
            self.dataset = TensorDataset(X,y)
        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError(f"TrainModel.fit require the data input ")

        # val set
        if X_val is not None and y_val is not None:
            self.val_dataset = TensorDataset(X_val, y_val)
        if val_dataset is not None:
            self.val_dataset = val_dataset        

        self.criterion = criterion
        self.dataloader = DataLoader(self.dataset, batch_size= batch_size, shuffle = is_shuffle, num_workers=2, pin_memory=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size= batch_size, shuffle = is_shuffle, num_workers=2, pin_memory=True) if self.val_dataset is not None else None
        for epoch in tqdm(range(n_epochs)):
            model.train()
            loss_epochs = 0
            num_sample_train = 0
            num_sample_val = 0
            acc_epoch = 0
            for  X_batch_train, y_batch_train in self.dataloader:
                X_batch_train, y_batch_train = X_batch_train.to(self.device), y_batch_train.to(self.device)
                num_sample_train += X_batch_train.shape[0]

                # forward
                with torch.autocast(device_type="cuda"):
                    logits = self.model(X_batch_train, y_batch_train)
                    loss = criterion(logits, y_batch_train)
                loss_epochs += loss.item()*X_batch_train.shape[0]
                acc = self.get_accuracy(logits, y_batch_train)*X_batch_train.shape[0]
                acc_epoch += acc.item()

                optimizer.zero_grad()

                # compute gradient
                self.scaler.scale(loss).backward()

                # update weight
                self.scaler.step(optimizer)

                self.scaler.update()
            self.Losses.append(loss_epochs/num_sample_train)
            self.Accuracies.append(acc_epoch/num_sample_train)
            loss_val_epoch = 0
            acc_val_epoch = 0
            if self.val_dataloader is not None:
                self.model.eval()
                for X_batch_val, y_batch_val in self.val_dataloader:
                    num_sample_val += X_batch_val.shape[0]
                    X_batch_val, y_batch_val = X_batch_val.to(self.device), y_batch_val.to(self.device)
                    loss_val, acc_val = self.evaluate(X_batch_val, y_batch_val)
                    loss_val_epoch += loss_val*X_batch_val.shape[0]
                    acc_val_epoch += acc_val*X_batch_val.shape[0]
                self.Val_Losses.append(loss_val_epoch/num_sample_val)
                self.Val_Accuracies.append(acc_val_epoch/num_sample_val)
            if len(self.Val_Losses):
                print(f"Epoch [{epoch+1:>4}/ {n_epochs}]  - Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = {self.Val_Losses[-1]:.4f} - Accuracy = {self.Val_Accuracies[-1]:.4f}")
            else:
                print(f"Epoch [{epoch+1:>4}/ {n_epochs}]  - Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = Nan - Accuracy = Nan")