import sys

sys.path.append('../')
import argparse
import os
import math
from tqdm import tqdm
from transformers import AutoTokenizer

from knowledage_bank.core.captions import Captions
from knowledage_bank.core.relativity import Relativity
from utils.formats import clean_string
from utils.io_json import read_jsonl, write_jsonl

tokenizer = AutoTokenizer.from_pretrained(
    '/data0/maqi/huggingface_models/llama-2-7b',
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False
)


def process_data(data, captions_, relativity_, caption_max_seq_length_, datasets_type_):
    out = []
    for row in tqdm(data, desc='process data'):
        sent_data, word_count = captions_.get_sent_data(row["article"])
        for question in row['questions']:
            query = question['question']
            options = question['options']
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
                "captions": chunk_captions,
                "context": chunks,
                "query": query,
                "option_0": 'A.' + options[0],
                "option_1": 'B.' + options[1],
                "option_2": 'C.' + options[2],
                "option_3": 'D.' + options[3],
                "question_unique_id": question['question_unique_id'],
                "label": question["gold_label"] if datasets_type_ != 'quality test' else None
            })
    return out


def process_file(input_path_, output_path_, captions_, relativity_, caption_max_seq_length_, datasets_type_):
    data = read_jsonl(input_path_)
    out = process_data(data, captions_, relativity_, caption_max_seq_length_, datasets_type_)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u quality_process.py --type train --output_dir quality_caption_and_rel > logs/quality_train.log 2>&1 &
    nohup python -u quality_process.py --type dev --output_dir quality_caption_and_rel > logs/quality_dev.log 2>&1 &
    nohup python -u quality_process.py --type test --output_dir quality_caption_and_rel > logs/quality_test.log 2>&1 &
    
    """
    PHASES = ["train", "dev", 'test']
    # train 7035
    # dev  7036
    # test 7037
    url_dict = {
        'train': "http://219.216.64.231:7035/get_captions",
        'dev': "http://219.216.64.231:7036/get_captions",
        'test': "http://219.216.64.231:7037/get_captions",
    }

    # PHASES = ['train']
    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--type", type=str, required=False, default='dev', choices=PHASES,
                        help="datasets")
    parser.add_argument("--output_dir", type=str, required=False, default='quality_caption_and_rel',
                        help="output_dir")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.output_dir)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/{args.output_dir}'

    input_path = f"{input_base_path}.{phase}"
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    caption_max_seq_length = 250
    captions = Captions(url=url_dict[phase], tokenizer=tokenizer, language='en', max_seq_length=caption_max_seq_length)
    relativity = Relativity(language='en', max_seq_length=50)
    datasets_type = 'quality test' if phase == 'test' else phase
    process_file(input_path, output_path, captions_=captions, relativity_=relativity,
                 caption_max_seq_length_=caption_max_seq_length, datasets_type_=datasets_type)
