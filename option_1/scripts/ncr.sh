#!/usr/bin/env sh
output_dir=/data0/maqi/KGLTQA/output/NCR/ncr_format_for_train_lr_3e5
logging_dir=${output_dir}/log
mkdir -p ${logging_dir}

nohup python -u -m main.run_kgltqa \
    --model_name_or_path /data0/maqi/huggingface_models/chinese-macbert-base \
    --model_mode mc \
    --task_name custom \
    --task_base_path /data0/maqi/KGLTQA/datasets/NCR/ncr_format_for_train \
    --output_dir  ${output_dir} \
    --logging_dir ${logging_dir} \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --do_predict \
    --logging_steps 100 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --max_seq_length 500 \
    --use_fast_tokenizer True \
    --local_rank -1 \
    --seed 318 > ${logging_dir}/log.out 2>&1 &