from mobile_bert_model import *
from config import MobileBertConfig

config = MobileBertConfig()
model = MobileBertModel(config)

i = {'input_ids': paddle.to_tensor([[ 101, 7592, 2023, 2003, 2026, 3899,  102]]),
     'token_type_ids': paddle.to_tensor([[0, 0, 0, 0, 0, 0, 0]]),
     'attention_mask': paddle.to_tensor([[1, 1, 1, 1, 1, 1, 1]])}
o = model(**i)




