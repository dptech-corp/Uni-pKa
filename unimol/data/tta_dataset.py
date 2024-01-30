# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class TTADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, atoms, coordinates, charges, id="ori_smi", conf_size=10):
        self.dataset = dataset
        self.seed = seed
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self.conf_size = conf_size
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def __len__(self):
        return len(self.dataset) * self.conf_size

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_idx = index // self.conf_size
        coord_idx = index % self.conf_size
        atoms = np.array(self.dataset[mol_idx][self.atoms])
        charges = np.array(self.dataset[mol_idx][self.charges])
        coordinates = np.array(self.dataset[mol_idx][self.coordinates][coord_idx])
        id = self.dataset[mol_idx][self.id]
        target = self.dataset[mol_idx]["target"]
        smi = self.dataset[mol_idx][self.id]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "charges": charges.astype(str),
            "target": target,
            "smi":smi,
            "target": target,
            "id": id,
        }

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class TTAPKADataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, metadata, atoms, coordinates, charges, id="ori_smi"):
        self.dataset = dataset
        self.seed = seed
        self.metadata = metadata
        self.atoms = atoms
        self.coordinates = coordinates
        self.charges = charges
        self.id = id
        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = {}
        total_sz = 0
        for i in range(len(self.dataset)):
            size = len(self.dataset[i][self.metadata])
            for j in range(size):
                self.idx2key[total_sz] = (i, j)
                total_sz += 1
        self.total_sz = total_sz

    def get_idx2key(self):
        return self.idx2key

    def __len__(self):
        return self.total_sz

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        smi_idx, mol_idx = self.idx2key[index]
        atoms = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.atoms])
        coordinates = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.coordinates])
        charges = np.array(self.dataset[smi_idx][self.metadata][mol_idx][self.charges])
        smi = self.dataset[smi_idx]["ori_smi"]
        id = self.dataset[smi_idx][self.id]
        target = self.dataset[smi_idx]["target"]
        return {
            "atoms": atoms,
            "coordinates": coordinates.astype(np.float32),
            "charges": charges.astype(str),
            "smi": smi,
            "target": target,
            "id": id,
        }
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
