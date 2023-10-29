import math
import sys

sys.path.append('../')
import argparse
import os

from tqdm import tqdm
from transformers import AutoTokenizer

from knowledage_bank.captions import Captions
from knowledage_bank.relativity import Relativity
from utils.formats import clean_string
from utils.io_json import read_jsonl, write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
tokenizer = AutoTokenizer.from_pretrained(
    '/data0/maqi/huggingface_models/llama-2-7b',
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False
)


def process_data(data, captions_, relativity_, caption_max_seq_length_, datasets_type_):
    out = []
    for row in tqdm(data[0], desc='process data'):
        sent_data, word_count = captions_.get_sent_data(row["passage"])
        query = row['question']
        options = row['options']
        # 分段
        # 分块数
        max_chunk_num = math.ceil(1900 / caption_max_seq_length_)
        # 平均每块大小
        average_chunk_num = math.ceil(word_count / 400)
        chunk_num = min(max_chunk_num, average_chunk_num)
        # 每块大小
        chunk_size = math.ceil(word_count / chunk_num)
        chunks = captions_.get_chunks(sent_data=sent_data, max_chunk_tokens=chunk_size)
        query = clean_string(query)
        options = [clean_string(option) for option in options]
        # get caption
        chunk_captions = []
        for idx, chunk in enumerate(chunks):
            # 打印进度
            print(f'process {idx}/{len(chunks)}', end='\r')
            chunk_caption = captions_.get_caption(sent=chunk)
            # rel = relativity_.get_relativity(query=query, options=options, passage=chunk)
            # chunk_captions.append({'idx': idx, 'caption': chunk_caption, 'rel': rel})
            chunk_captions.append(chunk_caption)

        out.append({
            "captions_and_rel": chunk_captions,
            "context": chunks,
            "query": query,
            "option_0": 'A.' + options[0],
            "option_1": 'B.' + options[1],
            "option_2": 'C.' + options[2],
            "option_3": 'D.' + options[3],
            "label": LABEL_TO_ID_DICT[row["answer"]] if datasets_type_ != 'quality test' else None
        })
    return out


def process_file(input_path_, output_path_, captions_, relativity_, caption_max_seq_length_, datasets_type_):
    data = read_jsonl(input_path_)
    out = process_data(data, captions_, relativity_, caption_max_seq_length_, datasets_type_)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u race_process.py --dataset middle --type test  > logs/race_middle_test.log 2>&1 &
    nohup python -u race_process.py --dataset middle --type dev  > logs/race_middle_dev.log 2>&1 &
    nohup python -u race_process.py --dataset high --type test  > logs/race_high_test.log 2>&1 &
    nohup python -u race_process.py --dataset high --type dev  > logs/race_high_dev.log 2>&1 &
    
    
    """
    # train 7035
    # dev  7036
    # test 7037
    url_dict = {
        'dev': "http://219.216.64.231:7030/get_captions",
        'test': "http://219.216.64.231:7030/get_captions",
    }
    PHASES = ["dev", 'test', 'train']

    # PHASES = ['train']
    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--dataset", type=str, required=False, default='middle', choices=['middle', 'high'],
                        help="datasets")
    parser.add_argument("--type", type=str, required=False, default='dev', choices=PHASES,
                        help="datasets type")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.dataset)
    input_path = f'/data0/maqi/KGLQA-data/datasets/RACE/raw/{args.dataset}_{args.type}.jsonl'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/RACE/caption_and_rel'

    output_path = os.path.join(output_base_path, f"{args.dataset}_{args.type}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    caption_max_seq_length = 250
    captions = Captions(url=url_dict[phase], tokenizer=tokenizer, language='en', max_seq_length=caption_max_seq_length)
    relativity = Relativity(language='en', max_seq_length=50)
    datasets_type = 'quality test' if phase == 'test' else phase
    process_file(input_path, output_path, captions_=captions, relativity_=relativity,
                 caption_max_seq_length_=caption_max_seq_length, datasets_type_=datasets_type)
