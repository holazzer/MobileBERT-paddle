import paddle
from paddle import nn
from . import MobileBertModel


class MobileBertForQuestionAnswering(nn.Layer):
    def __init__(self, mobile_bert):
        super().__init__()
        self.bert: MobileBertModel = mobile_bert
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids=None):
        output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        sequence_output = output.last_hidden_state

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


class MobileBertForMNLI(nn.Layer):
    def __init__(self, mobile_bert, dropout=None):
        super().__init__()
        self.bert: MobileBertModel = mobile_bert
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 3)

    def forward(self, input_ids, token_type_ids=None,
                position_ids=None, attention_mask=None):
        output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = output.last_hidden_state  # todo: use pooler ?

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


