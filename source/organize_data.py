from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
from vocabulary.vocab import Tokenizer
from preprocessing.preprocessing_text import *
import torch
class VicaptioningDataSet(Dataset):
    def __init__(self, dataset = None, split = 'train',  val_split = 0.1, transform: list = None, img_size = 224, vocab = None, max_length = 30):
        super().__init__()
        # Kiếm tra xem dataset có rỗng không
        if dataset is None:
            raise ValueError(f"the parameter dataset is not None !")
        self.list_idx_sample = []
        self.split_idx = None
        self.dataraw = dataset['train']
        if split not in ['train', 'valid']:
            raise ValueError(f"split only support value 'train' or 'valid' !")
        self.split = split
        if val_split:
            self.idx_train, self.idx_val = train_test_split(range(len(self.dataraw)), test_size = val_split, shuffle = False)
            self.split_idx = self.idx_train
        else:
            self.split_idx = range(len(self.dataraw))
        ##################### tổ chức DL 1 ảnh có nhiều caption -> 1 ảnh - 1 caption tương ứng ########################################## 
        for idx_img in self.split_idx:
            for idx_caption in range( len(self.dataraw[idx_img]['segment_caption_vi'])):
                # tổ chức data dưới dạng [(img1idx - caption1idx), (img1idx - caption2idx)]
                self.list_idx_sample.append((idx_img, idx_caption))
        #################################################################################################################################

        # transform mặc định là Resize, chuyển thành Tensor
        list_transforms =[transforms.Resize(size=(img_size, img_size)), transforms.ToTensor()]
        if transform is not None:
            list_transforms.extend(transform)
        self.transformer = transforms.Compose(list_transforms)
        # dict string to idx
        self.vocab = vocab
        self.tokenizer = Tokenizer()
        self.max_length = max_length
        self.processor = Preprocessing()
    def __len__(self):
        if self.split == 'train':
            return len(self.list_idx_sample)
        if self.split == 'valid':
            return len(list(self.idx_val))
    def __getitem__(self, index):
        if self.split == 'train':
            idx_img, idx_caption = self.list_idx_sample[index]
            img = self.transformer(self.dataraw[idx_img]['image'])
            caption =self.processor.tranfer2Lower(self.processor.remove_punctuation_digit(self.dataraw[idx_img]['segment_caption_vi'][idx_caption]))
            if self.vocab is None:
                return img , caption
            else:
                caption2idx = self.tokenizer(caption)
                num_padd = self.max_length - len(caption2idx) -2
                # Thêm token '<padd>' nếu câu chưa đủ dài 
                caption2idx = caption2idx[:(self.max_length-2)] + ['<end>'] + ['<padd>']*num_padd
                # Thêm token '<start>' để mở đầu sentence và '<end>' để kết thúc câu
                caption2idx = self.vocab(['<start>'] + caption2idx)
                return img, torch.tensor(caption2idx[:-1]), torch.tensor(caption2idx[1:])
        if self.split == 'valid':
            idx_img = list(self.idx_val)[index]
            img = self.transformer(self.dataraw[idx_img]['image'])
            list_caption = [self.processor.tranfer2Lower(self.processor.remove_punctuation_digit(caption)).split() for caption in self.dataraw[idx_img]['segment_caption_vi']]
            return img, list_caption