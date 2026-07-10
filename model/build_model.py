import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from pathlib  import Path
from torch.amp import GradScaler
import torchvision.models as models
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# root directory
ROOT_DIR = Path.cwd().parent
sys.path.append(str(ROOT_DIR))


from base_model.MLP import*
from base_model.ResNet_model import *

################# with model ResNet self-deployment ########################
'''class ViCaptionEncoder(ResNet):
    def __init__(self, version: int = 34, num_classes=1000):
        super().__init__(version, num_classes)
        # bỏ lớp Layer cuối cùng
        self.model = self.model[:-1]'''
###########################################################################
class ViCaptionEncoder(nn.Module):
    def __init__(self, version: int = 34, pretrained: bool = True):
        super().__init__()
        ResNet_dict = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152
        }
        weight_pretrained ={
            18: models.ResNet18_Weights.IMAGENET1K_V1,
            34: models.ResNet34_Weights.IMAGENET1K_V1,
            50: models.ResNet50_Weights.IMAGENET1K_V1,
            101: models.ResNet101_Weights.IMAGENET1K_V1,
            152: models.ResNet152_Weights.IMAGENET1K_V1
        }
        base_model = ResNet_dict[version](weights = weight_pretrained[version] if pretrained else None)
        list_layers_resnet = list(base_model.children())[:-1]
        list_layers_resnet.append(nn.Flatten())
        self.model = nn.Sequential(*list_layers_resnet)

    def forward(self,X):
        return self.model(X)
class ViCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding_matrix = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        try:
            self.rnn_layer = nn.LSTM(input_size= embedding_dim + 512, hidden_size= hidden_size, num_layers=num_layers, batch_first=True)
        except:
            self.rnn_layer = nn.LSTM(input_size= embedding_dim + 2048, hidden_size= hidden_size, num_layers=num_layers, batch_first=True)
        self.out2class = nn.Linear(in_features = hidden_size, out_features  = vocab_size)
        self.dropout_layer = nn.Dropout(p = 0.4)
    def forward(self,X, h0, c0, img_feature):
        X_embedded = self.embedding_matrix(X) # have shape (N, L, embedding_dim)
        L = X_embedded.shape[1]
        # img_feature have  shape (N, feature_map_dim)
        # Thêm một chiều vào vị trí index = 1 của img_feature (N, 1, feature_map_dim)
        img_feature_append = img_feature.unsqueeze(1).repeat(1, L, 1)
        input_lstm = torch.cat((X_embedded,img_feature_append), dim = -1)
        output,(hn, cn) = self.rnn_layer(input_lstm, (h0, c0))
        output = self.dropout_layer(output)
        logits = self.out2class(output)
        return logits, (hn, cn)


class ViCaptioningImgModel(nn.Module):
    def __init__(self, version_ResNet: int = 34, pretrained_ResNet : bool = True, vocab_size: int = 500, embedding_dim: int = 100, hidden_size: int = 100, num_layer_LSTM: int = 1 ):
        super().__init__()
        if version_ResNet not in [18, 34, 50, 101, 152]:
            raise  ValueError(f"ResNet only support version 18, 34, 50, 101, 152!")
        self.encoder = ViCaptionEncoder(version=version_ResNet, pretrained=pretrained_ResNet)
        self.decoder = ViCaptionDecoder(vocab_size= vocab_size, embedding_dim = embedding_dim, hidden_size= hidden_size, num_layers= num_layer_LSTM)
        self.LinearF2h0 = nn.Linear(in_features= 512 if version_ResNet in [18, 34]  else 2048, out_features= hidden_size)
        self.LinearF2c0 = nn.Linear(in_features= 512 if version_ResNet in [18, 34]  else 2048, out_features= hidden_size)
        self.num_layer = num_layer_LSTM
        self.activation_layer1 = nn.Tanh()
        self.activation_layer2 = nn.Tanh()
    def forward(self, X, y):
        FeatureMaps = self.encoder(X)
        c0, h0 = self.LinearF2c0(FeatureMaps), self.LinearF2h0(FeatureMaps)
        # Thêm một chiều vào dim =0 cho đúng đầu vào của LSTM, RNN
        c0 = self.activation_layer1(c0)
        h0 = self.activation_layer2(h0)
        
        c0 = c0.unsqueeze(0)
        h0 = h0.unsqueeze(0)

        c0 = c0.repeat(self.num_layer, 1, 1)
        h0 = h0.repeat(self.num_layer, 1, 1)
        logits, _ = self.decoder(y, h0, c0, FeatureMaps)

        # Có chuỗi <start> tôi đi học <end>
        # y_train có dạng <start> tôi đi học
        # y có dạng       tôi      đi học <end> 

        #  giống như tensor.view() -> thay đổi đúng về shape quy định (Batch_size, C, ...)
        # output của LSTM có shape(N, L, hidden_size)
        return logits.permute(0, 2, 1)

class TrainModel:
    def __init__(self) -> None:
        self.dataset = None
        self.val_dataset = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Losses = []
        self.Accuracies = []
        self.scaler = GradScaler()
    def get_accuracy(self, logits, y, idx_padding = 0):
        preds = (torch.argmax(logits, dim = 1) == y)
        mask = (y!= idx_padding)
        correct = preds[mask]
        return correct.sum().float()/mask.sum().float()
    
    def generate_caption(self, model, img, vocab, max_length = 20):
        if model is None:
            raise ValueError(f"Generate_caption require the parameter model !")
        else:
            self.model = model.to(self.device)
        # Thêm dim vào vị trí index = 0 -> shape(N,C,W,H)
        img = img.to(self.device)
        img = img.unsqueeze(0)
        FeatureMap = self.model.encoder(img)
        c0, h0 = self.model.LinearF2c0(FeatureMap), self.model.LinearF2h0(FeatureMap)
        # Thêm một chiều vào dim =0 cho đúng đầu vào của LSTM, RNN
        c0 = self.model.activation_layer1(c0)
        h0 = self.model.activation_layer2(h0)
        
        c0 = c0.unsqueeze(0)
        h0 = h0.unsqueeze(0)

        c0 = c0.repeat(self.model.num_layer, 1, 1)
        h0 = h0.repeat(self.model.num_layer, 1, 1)

        start_idx = vocab.vocab['<start>']
        end_idx = vocab.vocab['<end>']
        list_idx_generated =[start_idx]
        input_token = torch.tensor([[start_idx]]).to(self.device)
        while end_idx not in list_idx_generated and len(list_idx_generated) < max_length:
            logit, (h0,c0)= self.model.decoder(input_token, h0, c0, FeatureMap)
            next_token = torch.argmax(logit, dim = -1).item()
            list_idx_generated.append(next_token)
            input_token = torch.tensor([[next_token]]).to(self.device)
        list_words = [vocab.idx2str[idx] for idx in list_idx_generated[1:-1]]
        return ' '.join(list_words)

    def evaluate(self, model, val_dataset, vocab, max_length, num_sample=200):
        if model is None:
            raise ValueError(f" evaluate method require the parameter model !")
        with torch.no_grad():
            self.model.eval()
            smoothie = SmoothingFunction().method4
            Scores = {'bleu1': [], 'bleu2': [], 'bleu3': [], 'bleu4': []}
            for i in range(min(num_sample, len(val_dataset))):
                img = val_dataset[i][0]
                references = val_dataset[i][1]
                img = img.to(self.device)            
                candidate = self.generate_caption(model = model ,img=img, vocab=vocab, max_length= max_length).split()
                Scores['bleu1'].append(sentence_bleu(references, candidate, weights=(1,0,0,0), smoothing_function=smoothie))
                Scores['bleu2'].append(sentence_bleu(references, candidate, weights=(0.5,0.5,0,0), smoothing_function=smoothie))
                Scores['bleu3'].append(sentence_bleu(references, candidate, weights=(0.33,0.33,0.33,0), smoothing_function=smoothie))
                Scores['bleu4'].append(sentence_bleu(references, candidate, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoothie))

            return {k: sum(v)/len(v) for k, v in Scores.items()}
    def fit(self, model = None, dataset = None, val_dataset = None, n_epochs: int = 100, batch_size: int = 256, is_shuffle : bool = True, criterion = None, optimizer = None, scheduler =None, idx_padd = 0, vocab = None, max_length = 20):
        if model is None:
            raise ValueError(f"TrainModel.fit require the parameter model !")
        else:
            self.model = model.to(self.device)

        # train set
        if dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError(f"TrainModel.fit require the data input ")
        
        # valset
        if val_dataset is not None:
            self.val_dataset = val_dataset

        self.criterion = criterion
        self.dataloader = DataLoader(self.dataset, batch_size= batch_size, shuffle = is_shuffle, num_workers=2, pin_memory=True)
        for epoch in tqdm(range(n_epochs)):
            model.train()
            loss_epochs = 0
            num_tokens = 0
            acc_epoch = 0
            for  X_batch_train, y_batch_train, y_ in self.dataloader:
                X_batch_train, y_batch_train, y_ = X_batch_train.to(self.device), y_batch_train.to(self.device), y_.to(self.device)
                num_tokens += (y_ != idx_padd).sum().item()

                # forward
                with torch.autocast(device_type="cuda"):
                    logits = self.model(X_batch_train, y_batch_train)
                    loss = criterion(logits, y_)
                loss_epochs += loss.item()*(y_ != idx_padd).sum().item()
                acc = self.get_accuracy(logits, y_)*(y_ != idx_padd).sum().item()
                acc_epoch += acc.item()

                optimizer.zero_grad()

                # compute gradient
                self.scaler.scale(loss).backward()

                # update weight
                self.scaler.step(optimizer)

                self.scaler.update()
            if scheduler is not None:
                scheduler.step()
            self.Losses.append(loss_epochs/num_tokens)
            self.Accuracies.append(acc_epoch/num_tokens)
            dict_score = {}
            if self.val_dataset is not None:
                if epoch%10 ==0:
                    dict_score = self.evaluate(model = self.model,val_dataset = val_dataset, vocab = vocab, max_length = max_length)
            print(f"Epoch [{epoch+1:>4}/ {n_epochs}]  - Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f}")
            print(f"BLEU1  = {dict_score.get('bleu1',float('nan')):.4f} - BLEU2 = {dict_score.get('bleu2',float('nan')):.4f} - BLEU3 = {dict_score.get('bleu3',float('nan')):.4f} - BLEU4 ={dict_score.get('bleu4',float('nan')):.4f}")