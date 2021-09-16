# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
from .. import register_base_model
from .mobile_bert_pretrained_model import MobileBertConfig, MobileBertPreTrainedModel
from .mobile_bert_layers import MobileBertEmbeddings, MobileBertEncoder, MobileBertPooler, BaseModelOutputWithPooling

__all__ = [
    'MobileBertModel',
    'MobileBertForQuestionAnswering',
    'MobileBertForMNLI'
]

@register_base_model
class MobileBertModel(MobileBertPreTrainedModel):
    def __init__(self, add_pooling_layer=True):  # pretrained weight no pool
        super().__init__()
        self.mb_config = MobileBertConfig()
        self.embeddings = MobileBertEmbeddings(self.mb_config)
        self.encoder = MobileBertEncoder(self.mb_config)
        self.pooler = MobileBertPooler(self.mb_config) if add_pooling_layer else None
        self.apply(self.init_weights)
        self.dtype = self.mb_config.dtype

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                output_hidden_states=None, output_attentions=None, return_dict=None,):

        output_attentions = output_attentions \
            if output_attentions is not None else self.mb_config.output_attentions

        output_hidden_states = output_hidden_states \
            if output_hidden_states is not None else self.mb_config.output_hidden_states

        return_dict = return_dict \
            if return_dict is not None else self.mb_config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.mb_config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MobileBertForQuestionAnswering(MobileBertPreTrainedModel):

    def __init__(self, mobile_bert, dropout=None):
        super(MobileBertForQuestionAnswering, self).__init__()
        self.mobile_bert = mobile_bert
        self.classifier = nn.Linear(self.mobile_bert.mb_config.hidden_size, 2)


    def forward(self, input_ids, token_type_ids=None):
        output = self.mobile_bert(
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
