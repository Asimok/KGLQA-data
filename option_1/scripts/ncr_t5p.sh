#!/usr/bin/env sh
export PYTHONPATH=/data0/maqi/KGLTQA:${PYTHONPATH}
output_dir=/data0/maqi/KGLTQA/output/NCR_t5p/ncr_format_for_train_t5p_lr_3e4_with_options
logging_dir=${output_dir}/log
mkdir -p ${logging_dir}

max_seq_length=1536

nohup torchrun --nnodes 1 --nproc_per_node=2  main/run_kgltqa.py \
    --model_name_or_path /data0/maqi/huggingface_models/imxly/t5-pegasus \
    --model_mode encoder-decoder \
    --predict_with_generate True\
    --task_name custom \
    --task_base_path /data0/maqi/KGLTQA/datasets/NCR/ncr_format_for_train_t5p \
    --output_dir  ${output_dir} \
    --logging_dir ${logging_dir} \
    --learning_rate 3e-4 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --eval_accumulation_steps 2 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 70 \
    --max_seq_length 500 \
    --use_fast_tokenizer True \
    --log_level info \
    --local_rank -1 \
    --seed 318 > ${logging_dir}/log.out 2>&1 &

echo main/run_kgltqa.py \
    --model_name_or_path /data0/maqi/KGLTQA/output/NCR_t5p/ncr_format_for_train_t5p_lr_3e4/checkpoint-14000 \
    --model_mode encoder-decoder \
    --predict_with_generate True\
    --task_name custom \
    --task_base_path /data0/maqi/KGLTQA/datasets/NCR/ncr_format_for_train_t5p \
    --output_dir  ${output_dir} \
    --logging_dir ${logging_dir} \
    --learning_rate 3e-4 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --eval_accumulation_steps 500 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 70 \
    --max_seq_length 500 \
    --use_fast_tokenizer True \
    --log_level info \
    --local_rank -1 \
    --seed 318