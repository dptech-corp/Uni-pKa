# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import os
import argparse
import numpy as np
import glob


def cal_metrics(df):
    mae = np.abs(df["predict"] - df["target"]).mean()
    mse = ((df["predict"] - df["target"]) ** 2).mean()
    rmse = np.sqrt(mse)
    return mae, rmse


def get_csv_results(results_path, nfolds, task):

    all_smi_list, all_predict_list, all_target_list = [], [], []

    for fold_idx in range(nfolds):
        print(f"Processing fold {fold_idx}...")
        fold_path = os.path.join(results_path, f'fold_{fold_idx}')
        pkl_files = glob.glob(f"{fold_path}/*.pkl")
        fold_data = pd.read_pickle(pkl_files[0])

        smi_list, predict_list, target_list = [], [], []
        for batch in fold_data:
            sz = batch["bsz"]
            for i in range(sz):
                smi_list.append(batch["smi_name"][i])
                predict_list.append(batch["predict"][i].cpu().item())
                target_list.append(batch["target"][i].cpu().item())
        fold_df = pd.DataFrame({"smiles": smi_list, "predict": predict_list, "target": target_list})
        fold_df.to_csv(f'{fold_path}/fold_{fold_idx}.csv',index=False, sep='\t')

        # for final combined results
        all_smi_list.extend(smi_list)
        all_predict_list.extend(predict_list)
        all_target_list.extend(target_list)
    
    print(f"Combining results from {nfolds} folds into a single file...")
    combined_df = pd.DataFrame({"smiles": all_smi_list, "predict": all_predict_list, "target": all_target_list})
    combined_df.to_csv(f'{results_path}/all_results.csv', index=False, sep='\t')
    
    print(f"Calculating mean results for each SMILES...")
    mean_results = combined_df.groupby('smiles', as_index=False).agg({
    'predict': 'mean', 
    'target': 'mean'
    })
    mean_results.to_csv(f'{results_path}/mean_results.csv', index=False, sep='\t')
    if task == 'pka':
        print(f"MAE and RMSE for this task...")
        mae, rmse = cal_metrics(mean_results)
        print(f'MAE: {round(mae, 4)}, RMSE: {round(rmse, 4)}')
    print(f"Done!")


def main():
    parser = argparse.ArgumentParser(description='Model infer result mean ensemble')
    parser.add_argument(
        '--results-path', 
        type=str, 
        default='results',
        help='path to save infer results'
    )
    parser.add_argument(
        "--nfolds",
        default=5,
        type=int,
        help="cross validation split folds"
    )
    parser.add_argument(
        "--task",
        default='pka',
        type=str,
        choices=['pka', 'free_energy']
    )
    args = parser.parse_args()
    get_csv_results(args.results_path, args.nfolds, args.task)


if __name__ == "__main__":
    main()