from mobilebert_paddle import MobileBertTokenizer,\
    MobileBertModel, MobileBertConfig, MobileBertForQuestionAnswering
from paddlenlp.datasets import load_dataset

from functools import partial
from typing import Dict, Union, List

from squad_util import prepare, CrossEntropyLossForSQuAD, set_seed

from paddle.io import DataLoader
import paddle
from paddle.static import InputSpec

# from sklearn.metrics import confusion_matrix

weight_path = r'd:/nd.bin'
vocab_path = r'C:\Users\what\Desktop\mobile_bert\mobilebert_paddle\mobilebert-uncased\vocab.txt'


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

squad_v2_train_loader = DataLoader(squad_v2_train, batch_size=1, shuffle=True)
squad_v2_dev_loader = DataLoader(squad_v2_dev, batch_size=1, shuffle=False)

inp = [InputSpec([None, 512], paddle.int64, 'input_ids'),
       InputSpec([None, 512], paddle.int64, 'token_type_ids')]
lab = InputSpec([None, 4, 2], paddle.float32, 'pred')

model = paddle.Model(qa, inputs=inp, labels=lab)
model.prepare(optimizer=opt, loss=loss, metrics=acc)
model.fit(squad_v2_train_loader, eval_data=squad_v2_dev_loader, epochs=1)















