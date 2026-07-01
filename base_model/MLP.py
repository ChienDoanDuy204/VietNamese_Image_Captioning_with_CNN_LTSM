from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset, Subset
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler

##########################   kỹ thuật AMP - Automatic Mixed Precision -> Giảm kiểu DL từ float 32 -> float 16 -> tiết kiệm VRAM tăng tốc độ tính toán  #########################
# Ép các operations như Linear, Conv2d -> float 16 nhưng w thật vẫn ở 32
# Giữ nguyên float 32 với BatchNorm, SoftMax, loss function
# giải quyết vấn đề underflow của gradient bằng cách nhân loss với hệ số scale gọi là scaler_factor
# Khi backward mọi gradient trung gian (gradient nhân chập các lớp) được ở dạng float16 cũng được hệ số scale_factor
# Trước khi update các weight các gradient được scale_factor để trả về đún giá trị gradient 




# using when have skip connection - Residual Block
class ResidualBlock(nn.Module):
    r'''
    this class to add the technique skipconnection on model
    '''
    def __init__(self, block : nn.Sequential, downsample : nn.Sequential = None):
        super().__init__()
        self.block = block
        # downsample is used to change the shape of the input to the same shape as the output of the block
        self.downsample = downsample 
    def forward(self,X):
        # Lưu feature maps ban đầu lại để tránh bị thay đổi
        return self.block(X) + X if self.downsample is None else self.downsample(X) + self.block(X)



# Model Base for classification and Regression
class BaseMLP(ABC):
    r"""
    abstract class - Lớp trừu tượng - Tạo bộ khung, mẫu cho class BASEMLP

    All subclass should overwrite metthod 'predict' to get prediction of model throught
    logits = self.Forward().
    Subclass also should overwrite metthod 'get_accuracy' to get accuracy from 2 parameter logits and y
    Subclass also should overwrite metthod 'compute_loss' to get loss_value from self.criterion
    """


    def __init__(self):
        self.Layers = []
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.criterion = None
        
        self.scaler = GradScaler()
    @abstractmethod
    def predict(self,X):
        pass
    @abstractmethod
    def get_accuracy(self,logits,y):
        pass
    @abstractmethod
    def compute_loss(self,logits, y):
        pass


    def Add_layer(self,layers):
        if not isinstance(layers, list):
            raise TypeError("layers must be a list")
        self.Layers.extend(layers)
        # dấu * ở đây có nghĩa là mỗi phần tử của list là 1 tham số của hàm nn.Sequential
        self.model = nn.Sequential(*self.Layers)
    def forward(self,X):
        if self.model is not None:
            return self.model(X)
        raise ValueError("BaseMLP.model is None !")

    def print_fmt(self,Value):
        if Value is None or len(Value) ==0:
            return float('nan')
        return Value[-1]
    def fit(self, X = None, y = None, X_val = None, y_val = None, dataset = None, val_dataset = None, lr = 0.01, n_epochs = 100, batch_size = 1, verbose = 0, validation_split = None, is_shuffle = True, optimizer = 'SGD', criterion = 'MSE', random_state = 14):
        self.dataset = None
        self.val_dataset = None
        
        # Kiểm tra chuẩn bị dataset train
        if X is not None and y is not None:
            self.dataset = TensorDataset(X,y)
        elif dataset is not None:
            self.dataset = dataset
        else:
            raise ValueError("BaseMLP.dataset is empty !")
        
        # Kiểm tra chuẩn bị DL validation nếu có
        if X_val is not None and y_val is not None:
            self.val_dataset = TensorDataset(X_val,y_val)
        elif val_dataset is not None:
            self.val_dataset = val_dataset


        # Chuyển mô hình lên GPU
        if self.model is not None:
            self.model = self.model.to(self.device)
        else:
            raise ValueError("BaseMLP.model is None !")
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.is_shuffle = is_shuffle
        self.random_state = random_state
        # dict các optimizer và criterion hỗ trợ
        optimizers = {
            'sgd':torch.optim.SGD,
            'adam':torch.optim.Adam
        }
        
        criterions = {
            'mse': nn.MSELoss,
            'ce': nn.CrossEntropyLoss,
            'bce': nn.BCEWithLogitsLoss,
        }
        # Kiểm tra một biến có kiểu Dl là gì đó
        if isinstance(criterion,str):
            crit_class = criterions.get(criterion.lower())
            if crit_class is None:
                raise ValueError(f"criterion unsupport {criterion}")
            self.criterion = crit_class()
        else:
            if not isinstance(criterion, nn.Module):
                raise TypeError("criterion must be a subclass of torch.nn.Module")
            self.criterion = criterion
        
        if isinstance(optimizer,str):
            optim_class = optimizers.get(optimizer.lower())
            if optim_class is None:
                raise ValueError(f"optimizer unsupport {optimizer}")
            self.optimizer = optim_class(self.model.parameters(),lr = self.lr)
        else:
            if not isinstance(optimizer,Optimizer):
                raise TypeError("optimizer must be a subclass of torch.optim.Optimizer")
            self.optimizer = optimizer

        self.Val_Loader = None
        self.Train_Loader = None
        self.Losses = []
        self.Accuracies = []
        self.Val_Losses = []
        self.Val_Accuracies = []

        # Tách DL validation từ train set nếu muốn và không có val_dataset được truyền vào
        if self.validation_split is not None and self.validation_split>0 and self.val_dataset is None:
            '''
            Val_size = int(self.validation_split*len(self.dataset))
            Train_size = len(self.dataset) - Val_size
            # Chia DL trong train set và val set nếu có
            # Cách chia theo random_split sẽ làm mất cân bằng nhãn -> mô hình học không tốt
            self.TrainSet, self.ValSet = random_split(self.dataset,[Train_size,Val_size])
            self.Val_Loader = DataLoader(self.ValSet,batch_size=Val_size)
            '''
            idxs = range(len(self.dataset))
            labels = list(zip(*self.dataset))[1]
            # Tham số stratify=labels sẽ giữ nguyên phân bố của nhãn khi split giữa các tập được chia -> phân bố nhãn giữa các tập giống nhau 
            try: 
                # Split train, test follow index of dataset then use function Subset to split Traindaset and Valdataset
                train_idx, val_idx = train_test_split(idxs, train_size= (1- self.validation_split), stratify=labels, shuffle=True ,random_state=self.random_state)
            except:
                train_idx, val_idx = train_test_split(idxs, train_size= (1- self.validation_split), shuffle=True ,random_state=self.random_state)
            self.TrainSet, self.ValSet = Subset(self.dataset,train_idx), Subset(self.dataset,val_idx)
            self.Val_Loader = DataLoader(self.ValSet,batch_size=self.batch_size)
        else:
            self.TrainSet = self.dataset
        self.Train_Loader = DataLoader(self.TrainSet,batch_size=self.batch_size, shuffle= self.is_shuffle, num_workers = 2, pin_memory=True)


        # Kiêm tra val_dataset có tồn tại hay không
        if self.val_dataset is not None:
            self.Val_Loader = DataLoader(self.val_dataset,batch_size=self.batch_size, num_workers = 2, pin_memory= True)


        # Tính số batch trong train_loader
        num_batch = len(self.Train_Loader)

        for epoch in tqdm(range(self.n_epochs)):
            self.model.train()
            if self.verbose:
                print(f"Epoch [{epoch+1:>4}/{self.n_epochs}]")
            batch_count = 1
            Loss_epoch = 0
            Acc_epoch = 0
            for X_batch_train, y_batch_train in self.Train_Loader:
                X_batch_train, y_batch_train = X_batch_train.to(self.device), y_batch_train.to(self.device)

                # forward 
                with torch.autocast(device_type='cuda'):
                    logits = self.forward(X_batch_train)
                    loss = self.compute_loss(logits,y_batch_train)
                Loss_epoch += loss.item()
                acc = self.get_accuracy(logits, y_batch_train)
                Acc_epoch += acc.item()

                # 
                self.optimizer.zero_grad()

                # compute Gradient
                #loss.backward()
                self.scaler.scale(loss).backward()

                # update weight
                #self.optimizer.step()
                self.scaler.step(self.optimizer)

                #
                self.scaler.update()
                
                if self.verbose == 1 and batch_count< num_batch:
                    print(f"Batch {batch_count:>4}/{num_batch} - Loss = {loss.item():.4f} - Accuracy = {acc.item():.4f}")
                
                batch_count +=1
            self.Losses.append(Loss_epoch/num_batch)
            self.Accuracies.append(Acc_epoch/num_batch)

            # Đánh giá trên validation set nếu có
            if self.Val_Loader is not None:
                num_batch_val = len(self.Val_Loader)
                Loss_val_epoch = 0
                Acc_val_epoch = 0
                self.model.eval()
                for X_val, y_val in self.Val_Loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    with torch.no_grad():
                        logits_Val = self.model(X_val)
                        val_loss = self.compute_loss(logits_Val,y_val)
                        val_acc = self.get_accuracy(logits_Val, y_val)
                        Loss_val_epoch += val_loss.item()
                        Acc_val_epoch += val_acc.item()
                self.Val_Losses.append(Loss_val_epoch/num_batch_val)
                self.Val_Accuracies.append(Acc_val_epoch/num_batch_val)
            if self.verbose == 1:
                print(f"Batch {num_batch}/{num_batch} - Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = {self.print_fmt(self.Val_Losses):.4f} - Accracy_Validation = {self.print_fmt(self.Val_Accuracies):.4f}")
            if self.verbose >1:
                print(f"Loss = {self.Losses[-1]:.4f} - Accuracy = {self.Accuracies[-1]:.4f} - Loss_Validation = {self.print_fmt(self.Val_Losses):.4f} - Accracy_Validation = {self.print_fmt(self.Val_Accuracies):.4f}")