import paddle
import numpy as np
import random

from typing import List, Tuple

from paddlenlp.transformers import BertTokenizer
from paddlenlp.datasets import load_dataset

from paddle.io import Dataset

# hacks for MobileBertTokenizer
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
PAD_TOKEN = '[PAD]'


def prepare(example, tokenizer: 'BertTokenizer', max_sq=512):
    context: str = example['context']
    c = tokenizer.tokenize(context)

    question: str = example['question']
    q = tokenizer.tokenize(question)

    answers: List[str] = example['answers']
    answers_starts: List[int] = example['answer_starts']
    answers_ends = [i + len(ans) for i, ans in zip(answers_starts, answers)]

    tk_level_starts = []
    tk_level_ends = []

    is_impossible: bool = example['is_impossible']

    if is_impossible or len(answers) == 0:
        labels = [[0, 0]] * 4
    else:

        # char-level start -> token-level start
        for ans, char_start in zip(answers, answers_starts):
            b4 = context[:char_start]
            tk_start = len(tokenizer.tokenize(b4))
            tk_level_starts.append(tk_start + 1)  # add the [CLS] token ahead
            tk_len = len(tokenizer.tokenize(ans))
            tk_level_ends.append(tk_start + tk_len)

        # see if we need to move back the start
        head = min(tk_level_starts)
        tail = max(tk_level_ends)
        tol = max_sq - 3 - len(q)  # tokens left after removing the question

        if tail > tol:  # need offset
            offset = tail - tol
            new_head = (head - offset) // 2
            offset = offset + new_head
            for i in range(len(answers)):
                answers_starts[i] -= offset
                answers_ends[i] -= offset
            c = c[offset:]
        else:
            c = c[:tol]

        labels = [[st, ed] for st, ed in zip(tk_level_starts, tk_level_ends)]
        if len(labels) < 4: labels = (labels * 4)[:4]

    input_ids = [CLS_TOKEN] + c + [SEP_TOKEN] + q + [SEP_TOKEN]
    pad_cnt = (max_sq - len(input_ids))
    input_ids += [PAD_TOKEN] * pad_cnt
    input_ids = tokenizer.convert_tokens_to_ids(input_ids)
    input_ids = np.array(input_ids)
    token_type_ids = [0] * (len(c)+2) + [1] * (len(q)+1) + [0] * pad_cnt
    token_type_ids = np.array(token_type_ids)

    return input_ids, token_type_ids, np.array(labels)


class SquadFixDataset(Dataset):
    def __init__(self, name, tokenizer):
        super().__init__()
        self.name = name
        self.tokenizer = tokenizer
        self.ds = load_dataset('squad', splits=name)

    def __getitem__(self, idx):
        return prepare(self.ds[idx], self.tokenizer)

    def __len__(self):
        return len(self.ds)


class CrossEntropyLossForSQuAD(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self,start_logits, end_logits, labels: List[Tuple[int, int]]):
        labels = labels.transpose((1, 2, 0))
        loss = 0.
        for start_position, end_position in labels:
            start_loss = paddle.nn.functional.cross_entropy(
                input=start_logits, label=start_position)
            end_loss = paddle.nn.functional.cross_entropy(
                input=end_logits, label=end_position)
            loss += (start_loss + end_loss)
        return loss


def set_seed(_seed=0xCAFE):
    random.seed(_seed)
    np.random.seed(_seed)
    paddle.seed(_seed)









