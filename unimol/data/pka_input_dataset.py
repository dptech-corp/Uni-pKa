# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset
import collections
import torch
from itertools import chain
from unicore.data.data_utils import collate_tokens, collate_tokens_2d
from .coord_pad_dataset import collate_tokens_coords


class PKAInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_pad_idx, charge_pad_idx, split='train', conf_size=10):
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self._init_rec2mol()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)

        return src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch
    

class PKAMLMInputDataset(BaseWrapperDataset):
    def __init__(self, idx2key, src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, dist_targets, coord_targets, token_pad_idx, charge_pad_idx, split='train', conf_size=10):
        self.idx2key = idx2key
        self.dataset = src_tokens
        self.src_tokens = src_tokens
        self.src_charges = src_charges
        self.src_coord = src_coord
        self.src_distance = src_distance
        self.src_edge_type = src_edge_type
        self.token_targets = token_targets
        self.charge_targets = charge_targets
        self.dist_targets = dist_targets
        self.coord_targets = coord_targets
        self.token_pad_idx = token_pad_idx
        self.charge_pad_idx = charge_pad_idx
        self.split = split
        self.conf_size = conf_size
        self.left_pad = False
        self._init_rec2mol()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch
    
    def _init_rec2mol(self):
        self.rec2mol = collections.defaultdict(list)
        if self.split in ['train','train.small']:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].append(i)
        else:
            total_sz = len(self.idx2key)
            for i in range(total_sz):
                smi_idx, _ = self.idx2key[i]
                self.rec2mol[smi_idx].extend([i * self.conf_size + j for j in range(self.conf_size)])


    def __len__(self):
        return len(self.rec2mol)

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        mol_list = self.rec2mol[index]
        src_tokens_list = []
        src_charges_list = []
        src_coord_list = []
        src_distance_list = []
        src_edge_type_list = []
        token_targets_list = []
        charge_targets_list = []
        coord_targets_list = []
        dist_targets_list = []
        for i in mol_list:
            src_tokens_list.append(self.src_tokens[i])
            src_charges_list.append(self.src_charges[i])
            src_coord_list.append(self.src_coord[i])
            src_distance_list.append(self.src_distance[i])
            src_edge_type_list.append(self.src_edge_type[i])
            token_targets_list.append(self.token_targets[i])
            charge_targets_list.append(self.charge_targets[i])
            coord_targets_list.append(self.coord_targets[i])
            dist_targets_list.append(self.dist_targets[i])

        return src_tokens_list, src_charges_list,src_coord_list,src_distance_list,src_edge_type_list, token_targets_list, charge_targets_list, coord_targets_list, dist_targets_list
    
    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
    
    def collater(self, samples):
        batch = [len(samples[i][0]) for i in range(len(samples))]

        src_tokens, src_charges, src_coord, src_distance, src_edge_type, token_targets, charge_targets, coord_targets, dist_targets  = [list(chain.from_iterable(i)) for i in zip(*samples)]
        src_tokens = collate_tokens(src_tokens, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_charges = collate_tokens(src_charges, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        src_coord = collate_tokens_coords(src_coord, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_distance = collate_tokens_2d(src_distance, 0, left_pad=self.left_pad, pad_to_multiple=8)
        src_edge_type = collate_tokens_2d(src_edge_type, 0, left_pad=self.left_pad, pad_to_multiple=8)
        token_targets = collate_tokens(token_targets, self.token_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        charge_targets = collate_tokens(charge_targets, self.charge_pad_idx, left_pad=self.left_pad, pad_to_multiple=8)
        coord_targets = collate_tokens_coords(coord_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)
        dist_targets = collate_tokens_2d(dist_targets, 0, left_pad=self.left_pad, pad_to_multiple=8)

        return src_tokens, src_charges, src_coord, src_distance, src_edge_type, batch, charge_targets, coord_targets, dist_targets, token_targets