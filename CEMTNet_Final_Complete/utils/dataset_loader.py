import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os

class DepressionDataset(Dataset):
    def __init__(self, data_dir):
        try:
            self.text = np.load(os.path.join(data_dir, "text_features.npy"))
            self.audio = np.load(os.path.join(data_dir, "audio_features.npy"))
            self.labels = np.load(os.path.join(data_dir, "labels.npy"))
        except FileNotFoundError as e:
            raise RuntimeError(f"❌ 数据加载失败，确认路径正确且特征文件存在：{e}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.text[idx], dtype=torch.float32),
            torch.tensor(self.audio[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def get_dataloaders(data_dir, batch_size=16, split_ratio=0.8, seed=42):
    dataset = DepressionDataset(data_dir)
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    torch.manual_seed(seed)  # 保证每次划分一致
    train_set, val_set = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(val_set, batch_size=batch_size, shuffle=False)
    )
