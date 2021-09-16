from transformers.models.mobilebert import MobileBertModel, MobileBertTokenizer

mb = MobileBertModel.from_pretrained('google/mobilebert-uncased')

tk = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

sentence = "Advancing the state of the art: We work on computer science problems that define the technology of today and tomorrow."

i = tk(sentence, return_tensors='pt')

mb.eval()
o = mb(**i)


print(o)

