from paddlenlp.transformers import BertTokenizer


class MobileBertTokenizer(BertTokenizer):
    pretrained_resource_files_map = {
        "mobilebert-uncased":
            "https://huggingface.co/google/mobilebert-uncased/blob/main/vocab.txt"
    }
    pretrained_init_configuration = {
        "mobilebert-uncased": {
            "do_lower_case": True
        }
    }


class BertTokenizerForMNLI(MobileBertTokenizer):
    def mnli(self, sentence_1, sentence_2):
        sentence = "[CLS] {} [SEP] {} [SEP]".format(sentence_1, sentence_2)
        return self(sentence)
