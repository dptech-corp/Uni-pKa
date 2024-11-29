# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    SortDataset,
    TokenizeDataset,
    RawLabelDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSamplePKADataset,
    PKAInputDataset,
    PKAMLMInputDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    NormalizeDataset,
    CroppingDataset,
    FoldLMDBDataset,
    StackedLMDBDataset,
    SplitLMDBDataset,
    data_utils,
    MaskPointsDataset,
)

from unimol.data.tta_dataset import TTADataset, TTAPKADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)


@register_task("mol_pka_mlm")
class UniMolPKAMLMTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1.0,
            type=float,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--charge-dict-name",
            default="dict_charge.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            '--split-mode', 
            type=str, 
            default='predefine',
            choices=['predefine', 'cross_valid', 'random', 'infer'],
        )
        parser.add_argument(
            "--nfolds",
            default=5,
            type=int,
            help="cross validation split folds"
        )
        parser.add_argument(
            "--fold",
            default=0,
            type=int,
            help='local fold used as validation set, and other folds will be used as train set'
        )
        parser.add_argument(
            "--cv-seed",
            default=42,
            type=int,
            help="random seed used to do cross validation splits"
        )

    def __init__(self, args, dictionary, charge_dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.charge_dictionary = charge_dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.charge_mask_idx = charge_dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.split_mode !='predefine':
            self.__init_data()

    def __init_data(self):
        data_path = os.path.join(self.args.data, self.args.task_name + '.lmdb')
        raw_dataset = LMDBDataset(data_path)
        if self.args.split_mode == 'cross_valid':
            train_folds = []
            for _fold in range(self.args.nfolds):
                if _fold == 0:
                    cache_fold_info = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds).get_fold_info()
                if _fold == self.args.fold:
                    self.valid_dataset = FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info)
                if _fold != self.args.fold:
                    train_folds.append(FoldLMDBDataset(raw_dataset, self.args.cv_seed, _fold, nfolds=self.args.nfolds, cache_fold_info=cache_fold_info))
            self.train_dataset = StackedLMDBDataset(train_folds)
        elif self.args.split_mode == 'random':
            cache_fold_info = SplitLMDBDataset(raw_dataset, self.args.seed, 0).get_fold_info()   
            self.train_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 0, cache_fold_info=cache_fold_info)
            self.valid_dataset = SplitLMDBDataset(raw_dataset, self.args.seed, 1, cache_fold_info=cache_fold_info)

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        charge_dictionary = Dictionary.load(os.path.join(args.data, args.charge_dict_name))
        logger.info("charge dictionary: {} types".format(len(charge_dictionary)))
        return cls(args, dictionary, charge_dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        self.split = split
        if self.args.split_mode != 'predefine':
            if split == 'train':
                dataset = self.train_dataset
            elif split == 'valid':
                dataset =self.valid_dataset
        else:
            split_path = os.path.join(self.args.data, split + ".lmdb")
            dataset = LMDBDataset(split_path)
        tgt_dataset = KeyDataset(dataset, "target")
        if split in ['train', 'train.small']:
            tgt_list = [tgt_dataset[i] for i in range(len(tgt_dataset))]
            self.mean = sum(tgt_list) / len(tgt_list)
            self.std = 1
        elif split in ['novartis_acid', 'novartis_base', 'sampl6', 'sampl7', 'sampl8']:
            self.mean = 6.504894871171601  # precompute from dwar_8228 full set
            self.std = 1
        id_dataset = KeyDataset(dataset, "ori_smi")

        def GetPKAInput(dataset, metadata_key):
            mol_dataset = TTAPKADataset(dataset, self.args.seed, metadata_key, "atoms", "coordinates", "charges")
            idx2key = mol_dataset.get_idx2key()
            if split in ["train","train.small"]:
                sample_dataset = ConformerSamplePKADataset(
                    mol_dataset, self.args.seed, "atoms", "coordinates", "charges", self.args.conf_size
                )
            else:
                sample_dataset = TTADataset(
                    mol_dataset, self.args.seed, "atoms", "coordinates","charges","id", self.args.conf_size
                )

            sample_dataset = RemoveHydrogenDataset(
                sample_dataset,
                "atoms",
                "coordinates",
                "charges",
                self.args.remove_hydrogen,
                self.args.remove_polar_hydrogen,
            )
            sample_dataset = CroppingDataset(
                sample_dataset, self.seed, "atoms", "coordinates","charges", self.args.max_atoms
            )
            sample_dataset = NormalizeDataset(sample_dataset, "coordinates", normalize_coord=True)
            src_dataset = KeyDataset(sample_dataset, "atoms")
            src_dataset = TokenizeDataset(
                src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )
            src_charge_dataset = KeyDataset(sample_dataset, "charges")
            src_charge_dataset = TokenizeDataset(
                src_charge_dataset, self.charge_dictionary, max_seq_len=self.args.max_seq_len
            )
            coord_dataset = KeyDataset(sample_dataset, "coordinates")
            expand_dataset = MaskPointsDataset(
                src_dataset,
                coord_dataset,
                src_charge_dataset,
                self.dictionary,
                self.charge_dictionary,
                pad_idx=self.dictionary.pad(),
                charge_pad_idx=self.charge_dictionary.pad(),
                mask_idx=self.mask_idx,
                charge_mask_idx=self.charge_mask_idx,
                noise_type=self.args.noise_type,
                noise=self.args.noise,
                seed=self.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
            )

            def PrependAndAppend(dataset, pre_token, app_token):
                dataset = PrependTokenDataset(dataset, pre_token)
                return AppendTokenDataset(dataset, app_token)

            encoder_token_dataset = KeyDataset(expand_dataset, "atoms")
            encoder_target_dataset = KeyDataset(expand_dataset, "targets")
            encoder_coord_dataset = KeyDataset(expand_dataset, "coordinates")
            encoder_charge_dataset = KeyDataset(expand_dataset, "charges")
            encoder_charge_target_dataset = KeyDataset(expand_dataset, "charge_targets")

            src_dataset = PrependAndAppend(
                encoder_token_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
            src_charge_dataset = PrependAndAppend(
                encoder_charge_dataset, self.charge_dictionary.bos(), self.charge_dictionary.eos()
            )
            token_tgt_dataset = PrependAndAppend(
                encoder_target_dataset, self.dictionary.pad(), self.dictionary.pad()
            )
            charge_tgt_dataset = PrependAndAppend(
                encoder_charge_target_dataset, self.charge_dictionary.pad(), self.charge_dictionary.pad()
            )
            encoder_coord_dataset = PrependAndAppend(encoder_coord_dataset, 0.0, 0.0)
            encoder_distance_dataset = DistanceDataset(encoder_coord_dataset)

            edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
            coord_dataset = FromNumpyDataset(coord_dataset)
            coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
            distance_dataset = DistanceDataset(coord_dataset)

            return PKAMLMInputDataset(idx2key, src_dataset, src_charge_dataset, encoder_coord_dataset, encoder_distance_dataset, edge_type, token_tgt_dataset, charge_tgt_dataset, distance_dataset, coord_dataset, self.dictionary.pad(), self.charge_dictionary.pad(), split, self.args.conf_size)

        input_a_dataset = GetPKAInput(dataset, "metadata_a")
        input_b_dataset = GetPKAInput(dataset, "metadata_b")

        nest_dataset = NestedDictionaryDataset(
                {
                    "net_input_a": input_a_dataset,
                    "net_input_b": input_b_dataset,
                    "target": {
                        "finetune_target": RawLabelDataset(tgt_dataset),
                    },
                    "id": id_dataset,
                },
            )

        if not self.args.no_shuffle and split in ["train","train.small"]:
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(id_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
