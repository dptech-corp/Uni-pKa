# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
import argparse


def smi2scaffold(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smi, includeChirality=True)
    except:
        print("failed to generate scaffold with smiles: {}".format(smi))
        return smi 


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with {}".format(smi)
    return coordinates


def smi2_3Dcoords(smi,cnt, gen_mode='mmff'):
    assert gen_mode in ['mmff', 'no_mmff']
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)
            if res == 0:
                try:
                    if gen_mode == 'mmff':
                        AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)            
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    if gen_mode == 'mmff':
                        AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi) 

        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(smi)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def smi2metadata(smi, cnt, gen_mode): # input: single smi; output: molecule metadata (atom, charge, mol. smi, coords...)
    scaffold = smi2scaffold(smi)
    mol = Chem.MolFromSmiles(smi)

    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(smi)] * (cnt+1)
        print("atom num >400,use 2D coords",smi)
    else:
        # gen cnt num 3D conf
        coordinate_list = smi2_3Dcoords(smi,cnt, gen_mode)
        # gen 1 2D conf
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms, charges = [], []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
        charges.append(atom.GetFormalCharge())

    return {'atoms': atoms,'charges': charges, 'coordinates': coordinate_list, 'mol':mol,'smi': smi, 'scaffold': scaffold}


def inner_smi2coords_pka(content):
    smi_all, target = content
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d
    gen_mode = 'mmff'  # 'mmff', 'no_mmff'

    # get single smi from original SMILES
    smi_list_a, smi_list_b = smi_all.split('>>')
    smi_list_a = smi_list_a.split(',')
    smi_list_b = smi_list_b.split(',')

    # get whole datapoint metadata
    metadata_a, metadata_b = [], []
    for i in range(len(smi_list_a)):
        metadata_a.append(smi2metadata(smi_list_a[i], cnt, gen_mode))
    for i in range(len(smi_list_b)):
        metadata_b.append(smi2metadata(smi_list_b[i], cnt, gen_mode))

    return pickle.dumps({'ori_smi': smi_all, 'metadata_a': metadata_a, 'metadata_b': metadata_b, 'target': target}, protocol=-1)


def smi2coords(content):
    try:
        return inner_smi2coords_pka(content)
    except:
        print("failed smiles: {}".format(content[0]))
        return None
    

def load_rawdata_pka(input_csv):

    # read tsv file
    df = pd.read_csv(input_csv, sep='\t')
    smi_col = 'SMILES'
    target_col = 'TARGET'
    if target_col not in df.columns:
        # If not exist, add "-1.0" as a placeholder.
        df["TARGET"] = -1.0
    col_list = [smi_col, target_col]
    df = df[col_list]
    print(f'raw_data size: {df.shape[0]}')
    return df

def write_lmdb(task_name, input_csv, output_dir='.', nthreads=16):

    df = load_rawdata_pka(input_csv)
    content_list = zip(*[df[c].values.tolist() for c in df])
    lmdb_name = '{}.lmdb'.format(task_name)
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, lmdb_name)
    try:
        os.remove(output_name)
    except:
        pass
    env_new = lmdb.open(
        output_name,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords, content_list), total=len(df)):
            if inner_output is not None:
                txn_write.put(f'{i}'.encode("ascii"), inner_output)
                i += 1
        print('{} process {} lines'.format(lmdb_name, i))
        txn_write.commit()
        env_new.close()

def main():
    parser = argparse.ArgumentParser(
        description="use rdkit to generate conformers"
    )
    parser.add_argument(
        "--raw-csv-file",
        type=str,
        default="Datasets/tsv/chembl_train.tsv",
        help="the original data csv file path",
    )
    parser.add_argument(
        "--processed-lmdb-dir",
        type=str,
        default="chembl",
        help="dir of the processed lmdb data",
    )
    parser.add_argument("--nthreads", type=int, default=22, help="num of threads")
    parser.add_argument(
        "--task-name",
        type=str,
        default="train",
        help="name of the lmdb file; train and valid for chembl",
        choices=['train', 'valid', 'dwar-iBond', 'novartis_acid', 'novartis_base', 'sampl6', 'sampl7', 'sampl8']
    )
    args = parser.parse_args()
    write_lmdb(task_name = args.task_name, input_csv=args.raw_csv_file, output_dir=args.processed_lmdb_dir, nthreads = args.nthreads)


if __name__ == '__main__':
    main()
