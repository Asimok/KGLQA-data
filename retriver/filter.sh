export  CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/data0/maqi/KGLTQA
nohup python -u rocket_qa.py --type train --max_word_count 1536 --output_dir ncr_format_for_train_1536_no_qo > train.log 2>&1 &
nohup python -u rocket_qa.py --type validation --max_word_count 1536 --output_dir ncr_format_for_train_1536_no_qo > validation.log 2>&1 &
nohup python -u rocket_qa.py --type test --max_word_count 1536 --output_dir ncr_format_for_train_1536_no_qo > test.log 2>&1 &
