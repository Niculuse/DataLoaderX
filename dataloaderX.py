import torch
import collections
from torch.utils.data import DataLoader, Dataset
from typing import (
    Optional,
    Dict
)


class DatasetX(Dataset):
    r"""
        Args:
            dataset (Dataset): see torch.utils.data.Dataset for details.
            buffer_meta (dict, optional): dict with torch.dtype as key and tensor size as value.
                Default to empty dict.
            buffer_num (int, optional): the number of buffers. Default to 2.
            kwargs: (dict, optional): keyword arguments. See torch.utils.data.Dataloader for details.
    """
    def __init__(self, dataset: Dataset, buffer_meta: Optional[Dict] = None, buffer_num: Optional[int] = 2, **kwargs):
        self.dataset = dataset
        self.buffer_meta = buffer_meta
        self.buffer_num = buffer_num
        self.dataloader = iter(DataLoader(dataset, **kwargs))
        self.buffer_index = -1
        self.tensor_buffers = self.__init_tensor_buffers__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index=None):
        return next(self.dataloader)

    def __init_tensor_buffers__(self):
        if not self.buffer_meta:
            self.buffer_meta = {
                                torch.uint8: 1024**3, # for image tensors
                                torch.int64: 1024**2, # for label tensors
                                torch.float32: 1024**3, # for image tensors
                                }
        tensor_buffers = [{dtype:torch.empty(size, dtype=dtype).share_memory_() for dtype, size in self.buffer_meta.items()} for _ in range(self.buffer_num)]
        return tensor_buffers

    def copy_data_to_buffer(self, batch_data, buffer):
        start_index = 0
        def copy2buffer(data):
            if isinstance(data, torch.Tensor):
                nonlocal start_index
                size = data.numel()
                shape = data.shape
                dtype = data.dtype
                buffer[dtype][start_index:start_index+size].copy_(data.view((-1)))
                data = buffer[dtype][start_index:start_index+size].view(*shape)
                start_index += size
                return data
            elif isinstance(data, (tuple, list)):
                return [copy2buffer(sample) for sample in data]
            elif isinstance(data, collections.abc.Mapping):
                try:
                    return type(data)({key: copy2buffer(value) for key, value in data.items()})  # type: ignore[call-arg]
                except TypeError:
                    # The mapping type may not support `__init__(iterable)`.
                    return {key: copy2buffer(value) for key, value in data.items()}
            elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
                return type(data)(*(copy2buffer(sample) for sample in data))
            elif isinstance(data, collections.abc.Sequence):
                try:
                    return type(data)([copy2buffer(sample) for sample in data])  # type: ignore[call-arg]
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [copy2buffer(sample) for sample in data]
            else:
                return data
        batch_data = copy2buffer(batch_data)
        return batch_data
    
    def collate_fn(self, batch):
        batch = batch[0]
        self.buffer_index = (self.buffer_index + 1) % len(self.tensor_buffers)
        buffer = self.tensor_buffers[self.buffer_index]
        batch = self.copy_data_to_buffer(batch, buffer)
        return batch

class DataLoaderX(DataLoader):
    r"""
    Hierarchical DataLoader to reduce IPC time via tensors with shared_memory.

    Args:
        dataset (Dataset): dataset from which to load the data.
        buffer_meta (dict, optional): a dict with torch.dtype as key and tensor size as value.
                Default to an empty dict.
        prefetch_num (int, optional): the number of batches to prefetch for DataLoaderX.
                Default to 1. Note that a larger prefetch_num costs more memory!
        kwargs (dict, optional): see :py:mod:`torch.utils.data.Dataloader` for details.
.
    """
    def __init__(self, dataset: Dataset, buffer_meta: Optional[Dict] = None, prefetch_num: Optional[int] = 1, **kwargs):
        datasetX = DatasetX(dataset, buffer_meta, buffer_num=prefetch_num+1, **kwargs)
        super().__init__(datasetX, batch_size=1, shuffle=False, num_workers=1, collate_fn=datasetX.collate_fn,
                         pin_memory=True, prefetch_factor=prefetch_num, persistent_workers=True)
