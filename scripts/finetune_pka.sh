data_path='./unimol/examples'
MASTER_PORT=10090
task_name="dwar_small"
head_name='chembl_small'
weight_path='chembl_pretrain/checkpoint_best.pt'
n_gpu=1
save_dir='dwar_finetune'

# train params
seed=0
nfolds=5
cv_seed=42
task_num=1
loss_func="finetune_mse"
dict_name='dict.txt'
charge_dict_name='dict_charge.txt'
only_polar=-1
conf_size=11
local_batch_size=16
lr=3e-4
bs=32
epoch=20
dropout=0.1
warmup=0.06

for ((fold=0;fold<$nfolds;fold++))
    do
        export NCCL_ASYNC_ERROR_HANDLING=1
        export OMP_NUM_THREADS=1
        echo "params setting lr: $lr, bs: $bs, epoch: $epoch, dropout: $dropout, warmup: $warmup, cv_seed: $cv_seed, fold: $fold"
        update_freq=`expr $bs / $local_batch_size`
        fold_save_dir="fold_${fold}"
        model_dir="${save_dir}/${fold_save_dir}"
        python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
                --conf-size $conf_size --nfolds $nfolds --fold $fold --cv-seed $cv_seed\
                --num-workers 8 --ddp-backend=c10d \
                --dict-name $dict_name --charge-dict-name $charge_dict_name \
                --task mol_pka --loss $loss_func --arch unimol_pka  \
                --classification-head-name $head_name --num-classes $task_num \
                --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm 1.0 \
                --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
                --update-freq $update_freq --seed $seed \
                --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
                --log-interval 100 --log-format simple \
                --finetune-from-model $weight_path \
                --validate-interval 1 --keep-last-epochs 1 \
                --all-gather-list-size 102400 \
                --save-dir $model_dir \
                --best-checkpoint-metric valid_rmse --patience 2000 \
                --only-polar $only_polar --split-mode cross_valid
    done