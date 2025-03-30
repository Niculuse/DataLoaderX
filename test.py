import time
import torch
import numpy as np
from dataloaderX import DataLoaderX
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, size=100, shape=[24, 3, 704, 1280]):
        self.size = size
        self.data = torch.rand(*shape, dtype=torch.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data


if __name__ == "__main__":
    shape = [24, 3, 704, 1280]
    num_workers = 4
    dataset = RandomDataset(32, shape)
    dataloader = DataLoaderX(dataset, buffer_meta={torch.float32:int(np.prod(shape))}, batch_size=1, num_workers=num_workers)
    t1 = time.time()
    print(f'Testing DataloaderX...')
    for i, batch in enumerate(dataloader):
        print(f"Iter: {i}\tdata time: {time.time() - t1:.3f}s", flush=True)
        time.sleep(0.5)
        t1 = time.time()
    print(f'Testing Dataloader...')
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers)
    for i, batch in enumerate(dataloader):
        print(f"Iter: {i}\tdata time: {time.time() - t1:.3f}s", flush=True)
        time.sleep(0.5)
        t1 = time.time()