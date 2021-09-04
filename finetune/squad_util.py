import paddle
import numpy as np
import random

from typing import List, Tuple


def prepare(example, tokenizer):
    context: str = example['context']
    question: str = example['question']
    tkd: dict = tokenizer(context, question, stride=128, max_seq_len=128)

    answers: List[str] = example['answers']
    answers_starts: List[int] = example['answer_starts']
    answers_ends = [i + len(ans) for i, ans in zip(answers_starts, answers)]
    is_impossible: bool = example['is_impossible']

    if is_impossible:
        labels = [(0, 0)]
    else:
        labels = [(st, ed) for st, ed in zip(answers_starts, answers_ends)]

    return tkd, labels


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def forward(self, y, labels: List[Tuple[int, int]]):
        start_logits, end_logits = y
        cnt = (len(labels) * 2)
        loss = 0.
        for start_position, end_position in labels:
            start_loss = paddle.nn.functional.cross_entropy(
                input=start_logits, label=start_position)
            end_loss = paddle.nn.functional.cross_entropy(
                input=end_logits, label=end_position)
            loss += (start_loss + end_loss)
        return loss / cnt


def set_seed(_seed=0xCAFE):
    random.seed(_seed)
    np.random.seed(_seed)
    paddle.seed(_seed)








