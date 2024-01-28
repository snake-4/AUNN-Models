from torch.utils.data import Dataset
from impl.utils import *


class BinaryIndexedTextDataset(Dataset):
    def __init__(self, filename, bits):
        self.__bits = bits
        with open(filename, "rb") as f:
            self.X = (
                f.read().decode("ascii", "replace").replace("\r", "?").replace("�", "?")
            )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        input = binary_encode_tensor(torch.tensor(index), self.__bits)
        target = ord(self.X[index])
        return (input, target)
