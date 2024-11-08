model_path=./baseline_embedders/roberta-small
data_path=./data/
trn_dataset=arts-games-home-electronics-sports-tools
val_dataset=arts-games-home-electronics-sports-tools
# There is another argument, total_diverse_profile_num, in line 88 of train_easyrec.py. We set it to 3, but if you have more, you should increase it.
# total_diverse_profile_num >= used_diverse_profile_num
used_diverse_profile_num=3
output_model=./checkpoints/easyrec-small
metric_for_best_model=recall@10


# Allow multiple threads
export OMP_NUM_THREADS=1

# Use distributed data parallel
# 8 * A100 (40G)
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port 40001 train_easyrec.py \
    --model_name_or_path ${model_path} \
    --data_path ${data_path} \
    --trn_dataset ${trn_dataset} \
    --val_dataset ${val_dataset} \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model ${metric_for_best_model} \
    --load_best_model_at_end \
    --eval_steps 1000 \
    --save_steps 1000 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_mlm \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --save_safetensors False \
    --add_item_raw_meta True \
    --used_diverse_profile_num ${used_diverse_profile_num} \
    "$@"