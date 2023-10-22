import numpy as np
from rouge_chinese import Rouge


class Rouge1ScorerCh:
    def __init__(self):
        self.scorer = Rouge()

    def score(self, reference: str, target: str):
        scores = self.scorer.get_scores(hyps=target, refs=reference, avg=True)
        return scores
