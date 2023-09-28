from torch.utils.data import Dataset

filename = "./data/test.txt"

class TextModelDataset(Dataset):
    def __init__(self):
        with open(filename) as f:
            self.X = f.read().replace('\r', '')
    
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self, index):
        return ord(self.X[index])
