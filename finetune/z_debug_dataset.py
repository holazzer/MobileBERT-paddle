from mobilebert_paddle import MobileBertTokenizer,\
    MobileBertModel, MobileBertConfig, MobileBertForQuestionAnswering
from paddlenlp.datasets import load_dataset

from functools import partial
from typing import Dict, Union, List

from finetune.squad_util import prepare_one, CrossEntropyLossForSQuAD, set_seed

vocab_path = r'C:\Users\what\Desktop\mobile_bert\mobilebert_paddle\mobilebert-uncased\vocab.txt'

mobile_bert_tokenizer = mbt = MobileBertTokenizer(vocab_path)

squad_v2_train = load_dataset('squad', splits='train_v2')

squad_v2_train.map(partial(prepare_one, tokenizer=mbt), lazy=True)

