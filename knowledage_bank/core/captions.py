import json
import re
from typing import List

import requests

from knowledage_bank.core.prompt import get_caption_format
from utils.formats import clean_string


class Captions:
    def __init__(self,url, tokenizer=None, language='en', max_seq_length=200):
        self.tokenizer = tokenizer
        # train 7035
        # dev  7036
        # test 7037
        self.url = url
        self.max_seq_length = max_seq_length
        self.language = language
        self.split_token = '<question>:\n'

    def get_token_num(self, text):
        token = self.tokenizer.encode(text=text, add_special_tokens=False)
        return len(token)

    @staticmethod
    def split_text_into_sentences(text):
        # 使用正则表达式将文本按照句子分隔进行拆分
        sentences_ = re.split(r'[。！？；;.]', text)

        # 去除空白句子
        sentences_ = [s.strip() for s in sentences_ if s.strip()]

        # 合并使用省略号（...）结尾的句子
        merged_sentences = []
        for i in range(len(sentences_)):
            if sentences_[i].endswith('…') and i < len(sentences_) - 1:
                merged_sentence = sentences_[i] + sentences_[i + 1]
                merged_sentences.append(merged_sentence)
            else:
                merged_sentences.append(sentences_[i])

        return merged_sentences

    def get_sent_data(self, raw_text):
        sent_data = []
        word_count = 0
        for idx, sent_obj in enumerate(self.split_text_into_sentences(raw_text)):
            token_num = self.get_token_num(sent_obj)
            sent_data.append({
                "text": str(sent_obj).strip(),
                "word_count": token_num,
                "idx": idx
            })
            word_count += token_num
        return sent_data, word_count

    @staticmethod
    def get_chunks(sent_data: List, max_chunk_tokens: int):
        chunks = []
        cur_chunk_text = ''
        cur_chunk_tokens = 0
        for sent in sent_data:
            sent['text'] = clean_string(sent['text'])
            if cur_chunk_tokens + sent['word_count'] > max_chunk_tokens:
                chunks.append(cur_chunk_text)
                cur_chunk_text = ''
                cur_chunk_tokens = 0
            cur_chunk_text += sent['text']
            cur_chunk_tokens += sent['word_count']
        if cur_chunk_text != '':
            chunks.append(cur_chunk_text)
        return chunks

    def get_caption(self, sent: str):

        payload = json.dumps({
            "inputs": get_caption_format(language=self.language, passage=sent),
            "max_seq_length": self.max_seq_length,
            "split_token": self.split_token
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        output = response.json()["response"]
        return output
