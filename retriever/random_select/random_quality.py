import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys

sys.path.append('/data0/maqi/KGLQA-data')
from retriever.core.retriever import Retrieval, RocketScorer

import argparse

import torch
from tqdm import tqdm

from utils.formats import clean_string
from utils.io_json import read_jsonl, write_jsonl
from transformers import AutoTokenizer

load_type = torch.float16
device = torch.device('cuda')

tokenizer = AutoTokenizer.from_pretrained(
    '/data0/maqi/huggingface_models/llama-2-7b',
    trust_remote_code=True,
    # llama不支持fast
    use_fast=False
)


def process_data(data, scorer_, retrieval, max_word_count, datasets_type_='train', chunk=False):
    out = []
    for row in tqdm(data):
        sent_data, word_count = retrieval.get_sent_data(row["article"])
        for question in row['questions']:
            query = question['question']
            options = question['options']
            if word_count >= max_word_count:
                need_word_count = max_word_count - retrieval.get_token_num(query) - retrieval.get_token_num(
                    options[0]) - retrieval.get_token_num(
                    options[1]) - retrieval.get_token_num(options[2]) - retrieval.get_token_num(options[3])
                if chunk:
                    # chunk select
                    raw_context, chosen_sent_indices = retrieval.chunk(sent_data=sent_data,
                                                                       max_word_count=need_word_count)
                else:
                    # random select
                    raw_context, chosen_sent_indices = retrieval.random_select(sent_data=sent_data, max_word_count=need_word_count)
                shortened_article = ''.join(raw_context)
                context = clean_string(shortened_article)

            else:
                context = clean_string(row["article"])

            query = clean_string(query)
            options = [clean_string(option) for option in options]
            out.append({
                "context": context,
                "query": query,
                "option_0": 'A.' + options[0],
                "option_1": 'B.' + options[1],
                "option_2": 'C.' + options[2],
                "option_3": 'D.' + options[3],
                "question_unique_id": question['question_unique_id'],
                "label": question["gold_label"] if datasets_type_ != 'quality test' else None
            })
    # lens = []
    # for d in out:
    #     lens.append(retrieval.get_token_num(str(d)))
    # # 平均
    # print('dataset samples:', len(lens))
    # print('average tokens:', sum(lens) / len(lens))

    return out


def process_file(input_path_, output_path_, scorer_, retrieval, max_word_count_=512, datasets_type='train', chunk=False):
    data = read_jsonl(input_path_)
    out = process_data(data, scorer_, retrieval, max_word_count_, datasets_type, chunk)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u random_quality.py --type train --max_word_count 2048 --output_dir quality_random_2048 > logs/quality_train.log 2>&1 &
    nohup python -u random_quality.py --type dev --max_word_count 2048 --output_dir quality_random_2048 >  logs/quality_dev.log 2>&1 &
    nohup python -u random_quality.py --type test --max_word_count 2048 --output_dir quality_random_2048 >  logs/quality_test.log 2>&1 &
    
    nohup python -u random_quality.py --type train --max_word_count 2048 --output_dir quality_chunk_2048 > logs/quality_train.log 2>&1 &
    nohup python -u random_quality.py --type dev --max_word_count 2048 --output_dir quality_chunk_2048 >  logs/quality_dev.log 2>&1 &
    nohup python -u random_quality.py --type test --max_word_count 2048 --output_dir quality_chunk_2048 >  logs/quality_test.log 2>&1 &
    """
    PHASES = ["train", "dev", 'test']

    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--type", type=str, required=False, default='test', choices=PHASES,
                        help="datasets")
    parser.add_argument("--max_word_count", type=int, required=False, default=2048,
                        help="max_word_count")
    parser.add_argument("--output_dir", type=str, required=False, default='quality_rocketqa_2048',
                        help="output_dir")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.max_word_count, args.output_dir)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/QuALITY/QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/random_select/{args.output_dir}'
    query_type = 'question'
    scorer = RocketScorer(model_name='v2_nq_de', batch_size=64)
    Retriever = Retrieval(scorer=scorer, tokenizer=tokenizer)

    input_path = f"{input_base_path}.{phase}"
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    datasets_type = 'quality test' if phase == 'test' else 'train'
    process_file(input_path_=input_path, output_path_=output_path, scorer_=scorer, retrieval=Retriever,
                 max_word_count_=args.max_word_count - 100, datasets_type=datasets_type, chunk=True)
