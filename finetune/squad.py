import paddle
import paddlenlp as ppnlp
import random
import numpy as np
import os
import time
import json
import math

from functools import partial

from paddle.io import DataLoader

from finetune.ppnlp_examples.SQuAD.run_squad import prepare_train_features,\
    prepare_validation_features, evaluate, CrossEntropyLossForSQuAD
from mobilebert_paddle import MobileBertForQuestionAnswering, \
    MobileBertModel, MobileBertConfig, MobileBertTokenizer


# model and tokenizer
config = MobileBertConfig()
mobile_bert = MobileBertModel(config)

weight_path = ''
pretrained_weights = paddle.load(weight_path)
mobile_bert.load_dict(pretrained_weights)

vocab_path = 'mobilebert_paddle/mobilebert-uncased/vocab.txt'
tokenizer = MobileBertTokenizer(vocab_path, do_lower_case=True)


# run squad v1 dataset
train_ds = ppnlp.datasets.load_dataset('squad', splits='train_v1')
dev_ds = ppnlp.datasets.load_dataset('squad', splits='dev_v1')


# seed
seed = 0xCAFE
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


#

























