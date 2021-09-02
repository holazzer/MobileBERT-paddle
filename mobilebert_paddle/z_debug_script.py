import paddle
from mobile_bert_model import *
from config import MobileBertConfig
from ppnlp_tokenizer import MobileBertTokenizer


config = MobileBertConfig()
model = MobileBertModel(config, add_pooling_layer=False)
pretrained_weights = paddle.load('d:/nd.bin')
model.load_dict(pretrained_weights)
tokenizer = MobileBertTokenizer('./mobilebert-uncased/vocab.txt', do_lower_case=True)


def tk(s):
    d = tokenizer(s)
    for k, v in d.items():
        d[k] = paddle.to_tensor((v, ))
    return d


sentence = "Advancing the state of the art: We work on computer science problems that define the technology of today and tomorrow."

i = tk(sentence)

model.eval()
o = model(**i)





