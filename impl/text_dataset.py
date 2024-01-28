from torch.utils.data import Dataset
from impl.utils import *
import mmap
import os


class BinaryIndexedTextDataset(Dataset):
    def __init__(self, filename, bits):
        # This is done so that each worker can create their own mmap at __getitem__ call.
        self.__memview = None
        with open(filename, mode="rb") as f:
            f.seek(0, os.SEEK_END)
            self.__len = f.tell()
        self.__filename = filename
        self.__bits = bits

    def __len__(self) -> int:
        return self.__len

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.__memview:
            with open(self.__filename, mode="rb") as f:
                self.__memview = memoryview(mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ))
        input = binary_encode_tensor(torch.tensor(index), self.__bits)
        return (input, self.__memview[index])
