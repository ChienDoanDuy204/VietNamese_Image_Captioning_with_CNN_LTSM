from torch.utils.data import DataLoader
import json
import torch
class ComputeMeanStd:
    def __init__(self, dataset, batch_size, num_channels) -> None:
        self.dataloader = DataLoader(dataset, batch_size=batch_size)
        self.batch_size = batch_size
        self.num_channels = num_channels
    def __call__(self):
        mean = 0
        std = 0
        num_samples = 0
        for images, _ in self.dataloader:
            num_samples += images.shape[0]
            images = images.view(self.batch_size, self.num_channels, -1)
            mean += images.mean(dim = (0,2))*images.shape[0]
            std += images.std(dim = (0,2))*images.shape[0]
        mean /= num_samples
        std /= num_samples
        return mean, std
    def save_mean_std(self, mean, std, path):
        with open(file = path, mode = 'w', encoding = 'utf-8') as f:
            img_config = {
                "mean": mean.tolist(),
                "std": std.tolist()
            }
    def load_mean_std(self, path):
        with open(file = path, mode = 'r', encoding = 'utf-8') as f:
            img_config = json.load(f)
            mean = torch.tensor(img_config['mean'])
            std = torch.tensor(img_config['std'])
            return mean, std