from paddlenlp.transformers import BertTokenizer
from paddlenlp.datasets import load_dataset

from functools import partial
from typing import Dict, Union, List

from .squad_util import *


bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

squad_v2_dev = load_dataset('squad', splits='dev_v2')


squad_v2_dev.map(partial(prepare, tokenizer=bert_tokenizer), lazy=True)








