import json
from typing import List

import requests

from knowledage_bank.prompt import mark_relativity


class Relativity:
    def __init__(self, language='en', max_seq_length=5):
        self.url = "http://219.216.64.75:7035/get_captions"
        self.max_seq_length = max_seq_length
        self.language = language
        self.split_token = '<question>:\n'

    @staticmethod
    def cls_rel(rel: str):
        if rel.lower().__contains__('yes'):
            rel = 'yes'
        elif rel.lower().__contains__('no'):
            rel = 'no'
        elif rel.lower().__contains__('uncertain'):
            rel = 'uncertain'
        return rel

    def get_relativity(self, query: str, options: List, passage: str):
        query_rel = self.get_query_rel(query=query, passage=passage)
        options_rel = self.get_options_rel(options=options, passage=passage)
        query_rel_cls = self.cls_rel(query_rel)
        options_rel_cls = self.cls_rel(options_rel)
        if query_rel_cls == 'yes' or options_rel_cls == 'yes':
            print('yes')
        return {
            "query_rel": query_rel,
            "options_rel": options_rel,
            "query_rel_cls": query_rel_cls,
            "options_rel_cls": options_rel_cls
        }

    def get_query_rel(self, query: str, passage: str):
        return self.get_mark_rel(query=query, passage=passage, mark_type='query')

    def get_options_rel(self, options: List, passage: str):
        return self.get_mark_rel(query=';'.join(options), passage=passage, mark_type='options')

    def get_mark_rel(self, query: str, passage: str, mark_type='query'):
        payload = json.dumps({
            "inputs": mark_relativity(language=self.language, query=query, passage=passage, mark_type=mark_type),
            "max_seq_length": self.max_seq_length,
            "split_token": self.split_token
        })
        headers = {
            'Content-Type': 'application/json'
        }
        response = requests.request("POST", self.url, headers=headers, data=payload)
        output = response.json()["response"]
        return output
