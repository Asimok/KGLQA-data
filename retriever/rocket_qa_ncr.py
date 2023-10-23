import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys

sys.path.append('/data0/maqi/KGLQA-data')
import argparse
from retriever import RocketScorer, Retrieval

import torch
from tqdm import tqdm

from utils.formats import clean_string
from utils.io_json import read_jsonl, write_jsonl
from transformers import LlamaTokenizer

ckpt_path = '/data0/maqi/N_BELLE/BELLE/train/Version7/st5/'
load_type = torch.float16
device = torch.device(0)
tokenizer = LlamaTokenizer.from_pretrained(ckpt_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2
tokenizer.padding_side = "left"


def process_data(data, scorer_, retriever, max_word_count):
    out = []
    for row in tqdm(data):
        sent_data = retriever.get_sent_data(row["article"])
        for question in row['questions']:
            query = question['question']
            options = question['options']
            need_word_count = max_word_count - retriever.get_token_num(query) - retriever.get_token_num(
                options[0]) - retriever.get_token_num(
                options[1]) - retriever.get_token_num(options[2]) - retriever.get_token_num(options[3])
            shortened_article = retriever.get_top_sentences(
                query=query,
                sent_data=sent_data,
                opt_data=options,
                max_word_count=need_word_count,
                scorer_=scorer_,
            )

            context = clean_string(shortened_article)
            query = clean_string(query)
            options = [clean_string(option) for option in options]
            out.append({
                "context": context,
                "query": query,
                "option_0": 'A.' + options[0],
                "option_1": 'B.' + options[1],
                "option_2": 'C.' + options[2],
                "option_3": 'D.' + options[3],
                "label": question["gold_label"],
            })
    lens = []
    for d in out:
        lens.append(retriever.get_token_num(str(d)))
    # 平均
    print(sum(lens) / len(lens))

    return out


def process_file(input_path_, output_path_, scorer_, retriever, max_word_count=512):
    data = read_jsonl(input_path_)
    out = process_data(data, scorer_, retriever, max_word_count)
    write_jsonl(out, output_path_)
    print('save to ', output_path_)


if __name__ == '__main__':
    """
    nohup python -u rocket_qa_ncr.py --type train --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_train.log 2>&1 &
    nohup python -u rocket_qa_ncr.py --type dev --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_dev.log 2>&1 &
    nohup python -u rocket_qa_ncr.py --type test --max_word_count 1400 --output_dir ncr_rocketqa_1400 > logs/ncr_test.log 2>&1 &
    """
    PHASES = ["train", "dev", "test"]

    parser = argparse.ArgumentParser(description="rocket_qa preprocessing")
    parser.add_argument("--type", type=str, required=False, default='dev', choices=PHASES,
                        help="datasets")
    parser.add_argument("--max_word_count", type=int, required=False, default=1400,
                        help="max_word_count")
    parser.add_argument("--output_dir", type=str, required=False, default='ncr_rocketqa_1536',
                        help="output_dir")

    args = parser.parse_args()

    phase = args.type
    print(phase, args.max_word_count, args.output_dir)
    input_base_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_format'
    output_base_path = f'/data0/maqi/KGLQA-data/datasets/NCR/{args.output_dir}'
    query_type = 'question'
    scorer = RocketScorer(model_name='zh_dureader_de', batch_size=512)
    Retriever = Retrieval(scorer=scorer, tokenizer=tokenizer)

    input_path = os.path.join(input_base_path, f"{phase}.jsonl")
    output_path = os.path.join(output_base_path, f"{phase}.jsonl")
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)

    process_file(input_path_=input_path, output_path_=output_path, scorer_=scorer, retriever=Retriever,
                 max_word_count=args.max_word_count - 100)
