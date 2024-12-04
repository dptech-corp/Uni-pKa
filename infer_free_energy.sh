data_path='./unimol/examples'
infer_task='fe_example'
results_path='fe_results'
head_name='chembl_small'
conf_size=11
dict_name='dict.txt'
charge_dict_name='dict_charge.txt'
task_num=1
batch_size=16
model_path='dwar_finetune'
loss_func="infer_free_energy"
nfolds=5
only_polar=-1

for ((fold=0;fold<$nfolds;fold++))
       do     
              python ./unimol/infer.py --user-dir ./unimol ${data_path}  --task-name $infer_task --valid-subset $infer_task \
                     --results-path $results_path/fold_${fold}  \
                     --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
                     --task mol_free_energy --loss $loss_func --arch unimol \
                     --classification-head-name $head_name --num-classes $task_num \
                     --dict-name $dict_name --charge-dict-name $charge_dict_name --conf-size $conf_size \
                     --only-polar $only_polar  \
                     --path $model_path/fold_$fold/checkpoint_best.pt \
                     --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                     --log-interval 50 --log-format simple --required-batch-size-multiple 1
       done