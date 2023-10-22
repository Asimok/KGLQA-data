#!/usr/bin/env sh
# cosmosqa
python -m main.run_kgltqa \
    --model_name_or_path /data0/maqi/quality/models/roberta-base \
    --model_mode mc \
    --task_name cosmosqa \
    --output_dir /data0/maqi/quality/baselines/output/cosmosqa \
    --logging_dir /data0/maqi/quality/baselines/log \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --logging_steps 100 \
    --eval_steps 100 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --logging_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --max_seq_length 300 \
    --use_fast_tokenizer True \
    --disable_tqdm False \
    --log_level info \
    --seed 318