import paddle
from paddle import nn
import random
import numpy as np

from mobilebert_paddle import MobileBertForQuestionAnswering, \
    MobileBertModel, MobileBertConfig, MobileBertTokenizer


# seed
seed = 0xCAFE
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)


# model and tokenizer
config = MobileBertConfig()
# mobile_bert = MobileBertModel(config)
qa = MobileBertForQuestionAnswering(config)

weight_path = 'd:/nd.bin'
pretrained_weights = paddle.load(weight_path)
qa.mobilebert.load_dict(pretrained_weights)

# vocab_path = 'mobilebert_paddle/mobilebert-uncased/vocab.txt'
vocab_path = r'C:\Users\what\Desktop\mobile_bert\mobilebert_paddle\mobilebert-uncased\vocab.txt'
tokenizer = MobileBertTokenizer(vocab_path, do_lower_case=True)

ctx = "Extractive Question Answering is the task of extracting an answer" \
      " from a text given a question. An example of a question answering" \
      " dataset is the SQuAD dataset, which is entirely based on that task." \
      " If you would like to fine-tune a model on a SQuAD task, you may " \
      "leverage the examples/pytorch/question-answering/run_squad.py script."""

q = "What is extractive question answering?"

d = tokenizer(ctx, q, max_seq_len=384, pad_to_max_seq_len=1)
d = {k: paddle.to_tensor(v).unsqueeze(0) for k, v in d.items()}
d['start_positions'] = paddle.to_tensor([4], dtype=paddle.int64)
d['end_positions'] = paddle.to_tensor([6], dtype=paddle.int64)


qa.eval()
out = qa(**d)

print(out)

# run squad v1 dataset
# train_ds = ppnlp.datasets.load_dataset('squad', splits='train_v1')
# dev_ds = ppnlp.datasets.load_dataset('squad', splits='dev_v1')






























