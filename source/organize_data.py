from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

class VicaptioningDataSet(Dataset):
    def __init__(self, dataset = None, split: str = 'train', val_split: float = 0.0, is_shuffle: bool = True, sheet = 12, transform: list = None, img_size = 224):
        super().__init__()
        # Kiếm tra xem dataset có rỗng không
        if dataset is None:
            raise ValueError(f"the parameter dataset is not None !")
        self.list_idx_sample = []
        self.dataraw = None
        self.split_idx = None
        # kiểm tra xem split có là 1 trong 3 tập train, test, valid không
        if split != 'train' and split != 'test' and split != 'valid':
            raise ValueError(f"the split parameter unsupport {split}")
        # Nếu split = 'valid' nhưng quên điền val_split -> Lỗi
        if split =='valid' and not val_split:
            raise ValueError(f"The split = 'valid requirment val_split > 0! ")
        # Nếu có giá trị val_slpit và split là train hoặc test -> tiến hành tách DL từ tập train 
        if val_split and split != 'test':
            self.dataraw = dataset['train']
            idx_train, idx_val = train_test_split(range(len(self.dataraw)), test_size=val_split, shuffle=is_shuffle, random_state=sheet)
            if split == 'train':
                self.split_idx = idx_train
            else:
                self.split_idx = idx_val
        # Nếu split = 'test' thì giữ nguyên không chia
        if split == 'test':
            self.dataraw = dataset['test']
            self.split_idx = range(len(self.dataraw))
        # Nếu val_split = 0 và split = 'train' thì giữ nguyên tập train không chia
        if not val_split and split =='train':
            self.dataraw = dataset['train']
            self.split_idx = range(len(self.dataraw))
        

        ##################### tổ chức DL 1 ảnh có nhiều caption -> 1 ảnh - 1 caption tương ứng ########################################## 
        for idx_img in self.split_idx:
            for idx_caption in range( len(self.dataraw[idx_img]['caption_vi'])):
                # tổ chức data dưới dạng [(img1idx - caption1idx), (img1idx - caption2idx)]
                self.list_idx_sample.append((idx_img, idx_caption))
        #################################################################################################################################

        # transform mặc định là Resize, chuyển thành Tensor
        list_transforms =[transforms.Resize(size=img_size), transforms.ToTensor()]
        if transform is not None:
            list_transforms.extend(transform)
        self.transformer = transforms.Compose(list_transforms)
    def __len__(self):
        return len(self.list_idx_sample)
    def __getitem__(self, index):
        idx_img, idx_caption = self.list_idx_sample[index]
        img = self.transformer(self.dataraw[idx_img]['image'])
        caption = self.dataraw[idx_img]['caption_vi'][idx_caption]
        return img , caption