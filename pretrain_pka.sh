data_path="chembl"
task_name="chembl"
n_gpu=8
save_dir="pretrain_save"
tmp_save_dir="tmp_save"
task_num=1
loss_func="pretrain_mlm"
dict_name="dict.txt"
charge_dict_name="dict_charge.txt"
only_polar=-1
conf_size=11
split_mode="predefine"
MASTER_PORT=10090

# train params
local_batch_size=16
batch_size=16
lr=1e-4
epoch=100
dropout=0.1
warmup=0.06
seed=0
mask_prob=0.05
update_freq=`expr $batch_size / $local_batch_size`
global_batch_size=`expr $batch_size \* $n_gpu \* $update_freq`
echo "params setting lr: $lr, bs: $global_batch_size, epoch: $epoch, dropout: $dropout, warmup: $warmup, seed: $seed"

# loss
masked_token_loss=1
masked_charge_loss=2
masked_coord_loss=2
masked_dist_loss=1
x_norm_loss=0.01
delta_pair_repr_norm_loss=0.01


export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT \
        $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
        --conf-size $conf_size \
        --num-workers 8 --ddp-backend=c10d \
        --dict-name $dict_name --charge-dict-name $charge_dict_name \
        --task mol_pka_mlm --loss $loss_func --arch unimol_pka  \
        --classification-head-name $task_name --num-classes $task_num \
        --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
        --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout\
        --update-freq $update_freq --seed $seed \
        --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
        --keep-last-epochs 1 \
        --log-interval 100 --log-format simple \
        --validate-interval 1 \
        --save-dir $save_dir --tmp-save-dir $tmp_save_dir --tensorboard-logdir $save_dir/tsb \
        --best-checkpoint-metric valid_rmse --patience 2000 \
        --only-polar $only_polar --mask-prob $mask_prob \
        --masked-token-loss $masked_token_loss --masked-coord-loss $masked_coord_loss --masked-dist-loss $masked_dist_loss \
        --masked-charge-loss $masked_charge_loss --x-norm-loss $x_norm_loss --delta-pair-repr-norm-loss $delta_pair_repr_norm_loss

