export  PYTHONPATH=/data0/maqi/quality/baselines:$PYTHONPATH




# quality_v1.0.1
python baselines/lrqa/lrqa/run_lrqa.py \
    --model_name_or_path /data0/maqi/quality/models/roberta-base \
    --model_mode mc \
    --task_name custom \
    --task_base_path /data0/maqi/quality/datasets/quality_v1.0.1 \
    --output_dir /data0/maqi/quality/baselines/output/quality_v1.0.1 \
    --logging_dir /data0/maqi/quality/baselines/log/quality_v1.0.1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --logging_steps 500 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 20 \
    --max_seq_length 512 \
    --use_fast_tokenizer True \
    --disable_tqdm False \
    --seed 318


# ft
nohup python -u baselines/lrqa/lrqa/run_lrqa.py \
    --model_name_or_path /data0/maqi/quality/baselines/output/NCR/checkpoint-4000 \
    --model_mode mc \
    --task_name custom \
    --task_base_path /data0/maqi/quality/datasets/ncr_clean_for_train \
    --output_dir /data0/maqi/quality/baselines/output/NCR_ft \
    --logging_dir /data0/maqi/quality/baselines/log/NCR_ft \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 15 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --logging_steps 500 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 50 \
    --max_seq_length 512 \
    --use_fast_tokenizer True \
    --disable_tqdm False \
    --log_level info \
    --seed 318 > log.out 2>&1 &

# large
nohup python -u baselines/lrqa/lrqa/run_lrqa.py \
    --model_name_or_path /data0/maqi/huggingface_models/chinese-roberta-wwm-ext-large \
    --model_mode mc \
    --task_name custom \
    --task_base_path /data0/maqi/quality/datasets/ncr_clean_for_train \
    --output_dir /data0/maqi/quality/baselines/output/NCR_chinese-roberta-wwm-ext-large \
    --logging_dir /data0/maqi/quality/baselines/log/NCR_chinese-roberta-wwm-ext-large \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 2 \
    --do_train \
    --do_eval \
    --logging_steps 500 \
    --eval_steps 100 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 50 \
    --max_seq_length 512 \
    --use_fast_tokenizer True \
    --log_level info \
    --seed 318 > log.out 2>&1 &

# mac bert
nohup python -u baselines/lrqa/lrqa/run_lrqa.py \
    --model_name_or_path /data0/maqi/huggingface_models/chinese-macbert-base \
    --model_mode mc \
    --task_name custom \
    --task_base_path /data0/maqi/quality/datasets/ncr_clean_for_train \
    --output_dir /data0/maqi/quality/baselines/output/NCR_chinese-macbert-base \
    --logging_dir /data0/maqi/quality/baselines/log/NCR_chinese-macbert-base \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 15 \
    --per_device_eval_batch_size 15 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --do_eval \
    --logging_steps 100 \
    --eval_steps 100 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_strategy steps \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --num_train_epochs 50 \
    --max_seq_length 512 \
    --use_fast_tokenizer True \
    --log_level info \
    --seed 318 > log.out 2>&1 &