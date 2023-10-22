import sys

sys.path.append('/data0/maqi/Option-2/')
import argparse

from tqdm import tqdm

from techgpt_api import make_requests_caption

import json
import os

from utils.io_json import write_jsonl

LABEL_TO_ID_DICT = {"A": 0, "B": 1, "C": 2, "D": 3}
DICT_TO_LABEL = {0: "A", 1: "B", 2: "C", 3: "D"}
PHASES = ["train", "dev", "test"]


def gen_request(query, passage):
    #     return (f'请判断下列段落是否可以解答问题:\n'
    #             f'问题:\n{query}\n'
    #             f'段落:\n{passage}')
    return (f'Please determine if the following paragraphs can answer the question:\n'
            f'question:\n{query}\n'
            f'passage:\n{passage}')


def process(dataset_type, save_path_):
    phase = dataset_type
    raw_file = phase
    dataset = []
    with open(f'/data0/maqi/KGLTQA/datasets/quality_process/quality_caption_techgpt/{phase}.jsonl', 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    out_data = []
    for elem in tqdm(dataset):
        # 统计有几个option
        option_num = 0
        for i in range(4):
            if elem.get(f"option_{i}", None) is not None:
                option_num += 1
        options = ''
        for option_id in range(option_num):
            options += f'{elem[f"option_{option_id}"]}\n'

        for caption_idx, caption in enumerate(elem['caption_data']):
            # 判断相关性
            query_text_rel = make_requests_caption(gen_request(elem['query'], ' '.join(caption['text'])))
            query_caption_rel = make_requests_caption(gen_request(elem['query'], caption['caption']))

            answer = elem[f"option_{elem['label']}"]
            ans_text_rel = make_requests_caption(gen_request(answer, ' '.join(caption['text'])))
            ans_caption_rel = make_requests_caption(gen_request(answer, caption['caption']))

            caption['query_text_rel'] = query_text_rel
            caption['query_caption_rel'] = query_caption_rel
            caption['ans_text_rel'] = ans_text_rel
            caption['ans_caption_rel'] = ans_caption_rel
        out_data.append(elem)

    write_jsonl(out_data, os.path.join(save_path_, f"{raw_file}.jsonl"))
    print('save to: ', os.path.join(save_path_, f"{raw_file}.jsonl"))


if __name__ == '__main__':

    PHASES = ["train", "dev", "test"]
    """
    nohup python -u get_relativity_from_techgpt.py --type train  --output_dir quality_caption_and_relativity_techgpt > train.log 2>&1 &
    nohup python -u get_relativity_from_techgpt.py --type dev  --output_dir quality_caption_and_relativity_techgpt > dev.log 2>&1 &
    nohup python -u get_relativity_from_techgpt.py --type test  --output_dir quality_caption_and_relativity_techgpt > test.log 2>&1 &
    """
    parser = argparse.ArgumentParser(description="techgpt caption preprocessing")
    parser.add_argument("--type", type=str, required=False, choices=PHASES,default='train',
                        help="datasets")
    parser.add_argument("--output_dir", type=str, required=False, default='quality_caption_and_relativity_techgpt',
                        help="output_dir")

    args = parser.parse_args()
    # save_path = '/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_and_relativity_techgpt'
    save_path = f'/data0/maqi/KGLTQA/datasets/quality_process/{args.output_dir}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    process(args.type, save_path)
    # process('train', '/data0/maqi/KGLQA-data/datasets/NCR/ncr_caption_techgpt')
    print('save to ', save_path)

