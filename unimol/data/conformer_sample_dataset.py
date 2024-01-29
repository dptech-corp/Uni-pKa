# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
from . import data_utils


class ConformerSamplePKADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, charges, id="ori_smi"):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        atoms = np.array(self.dataset[index][self.atoms])
        charges = np.array(self.dataset[index][self.charges])
        assert len(atoms) > 0, 'atoms: {}, charges: {}, coordinates: {}, id: {}'.format(atoms, charges, coordinates, id)
        size = len(self.dataset[index][self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = self.dataset[index][self.coordinates][sample_idx]
        return {"atoms": atoms, "coordinates": coordinates.astype(np.float32),"charges":charges,"id": self.id}

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
