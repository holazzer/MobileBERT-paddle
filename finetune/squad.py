import paddle
import paddlenlp

squad_train_v1 = paddlenlp.datasets.load_dataset('squad', splits=['train_v1'])

mnli_train = paddlenlp.datasets.load_dataset('glue', 'mnli', splits=['train_v1'])



from mobilebert_paddle import MobileBertTokenizer
tokenizer = MobileBertTokenizer('mobilebert_paddle/mobilebert-uncased/vocab.txt')
print(tokenizer("Hello world"))
