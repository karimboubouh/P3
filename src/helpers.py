import numpy as np
import torch
from torch.utils.data import Dataset

from src.conf import DATASET_DUPLICATE


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        if DATASET_DUPLICATE:
            dataset, idxs = self.times_dataset(dataset, idxs, times=DATASET_DUPLICATE)
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if DATASET_DUPLICATE:
            while item > 60000:
                item -= 60000
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label)

    @staticmethod
    def times_dataset(dataset, idxs, times=1):
        _dataset = None
        _idxs = []
        for i in range(times):
            _dataset = dataset if _dataset is None else _dataset + dataset
            _idxs += list(np.array(idxs) + i * len(dataset))

        return _dataset, _idxs


class Map(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]
