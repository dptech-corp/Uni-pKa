# ProtoEnumerator

The implementation of the Microstate Enumerator in the paper Uni-pKa: An Accurate and Physically Consistent pKa Prediction through Protonation Ensemble Modeling.

## Introduction

It uses iterated template-matching algorithm to enumerate all the microstates in adjacent macrostates of a molecule's protonation ensemble from at least one microstate stored as SMILES.

The protonation template `smarts_pattern.tsv` modifies and augments the one in the paper [MolGpka: A Web Server for Small Molecule pKa Prediction Using a Graph-Convolutional Neural Network](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00075) and its open source implementation (MIT license) in the Github repository [MolGpKa](https://github.com/Xundrug/MolGpKa/blob/master/src/utils/smarts_pattern.tsv).

## Usage

`main.py` is to reconstruct a plain pKa dataset to the Uni-pKa standard format with fully enumerated microstates.

```shell
python main.py enum -i <input> -o <output> -m <mode>
```

The recommended environment is
```yaml
python = 3.8.13
rdkit = 2021.09.5
numpy = 1.20.3
pandas = 1.5.2
```

The <input> dataset is assumed be a csv-like file with a column storing SMILES. There are two cases allowed for each entry in the dataset.

1. It contains only one SMILES. The Enumerator helps to build the protonated/deprotonated macrostate and complete the original macrostate.
    - When <mode> is "A", it will be considered as an acid (thrown into A pool). 
    - When <mode> is "B", it will be considered as a base (thrown into B pool).
2. It contains a string like "A1,...,Am>>B1,...Bn", where A1,...,Am are comma-separated SMILES of microstates in the acid macrostate (all thrown into A pool), and B1,...,Bn are comma-separated SMILES of microstates in the base macrostate(all thrown into B pool). The Enumerator helps to complete the both.

The <mode> "A" (default) or "B" determines which pool (A/B) is the reference structures and the starting point of the enumeration.

The <output> dataset is then constructed after the enumeration.