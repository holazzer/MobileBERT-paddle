from mobilebert_paddle import MobileBertTokenizer,\
    MobileBertModel, MobileBertConfig, MobileBertForQuestionAnswering
from paddlenlp.datasets import load_dataset

from functools import partial
from typing import Dict, Union, List

from .squad_util import prepare, CrossEntropyLossForSQuAD, set_seed

from paddle.io import DataLoader
import paddle

from sklearn.metrics import confusion_matrix

weight_path = r''
vocab_path = r''


set_seed()

config = MobileBertConfig()
bert = MobileBertModel(config)
pretrained_weights = paddle.load(weight_path)
bert.load_dict(pretrained_weights)
qa = MobileBertForQuestionAnswering(bert)

mobile_bert_tokenizer = mbt = MobileBertTokenizer(vocab_path)

loss = CrossEntropyLossForSQuAD()
opt = paddle.optimizer.AdamW(parameters=qa.parameters())
acc = paddle.metric.Accuracy()


squad_v2_train = load_dataset('squad', splits='train_v2')
squad_v2_dev = load_dataset('squad', splits='dev_v2')

squad_v2_train.map(partial(prepare, tokenizer=mbt), lazy=True)
squad_v2_dev.map(partial(prepare, tokenizer=mbt), lazy=True)

squad_v2_train_loader = DataLoader(squad_v2_train, batch_size=8, shuffle=True)
squad_v2_dev_loader = DataLoader(squad_v2_dev, batch_size=8, shuffle=False)


model = paddle.Model(qa)
model.prepare(optimizer=opt, loss=loss, metrics=acc)
model.fit(squad_v2_train_loader, eval_data=squad_v2_dev_loader, epochs=1)















