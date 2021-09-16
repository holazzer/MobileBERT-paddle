import paddle
from paddle import nn
import paddle.nn.functional as F

import math
from .monkey import Linear as _Linear
nn.Linear = _Linear

from .monkey import NumpyMatmul
# paddle.matmul = NumpyMatmul.apply


class MobileBertSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.true_hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.true_hidden_size, self.all_head_size)
        self.value = nn.Linear(
            config.true_hidden_size if config.use_bottleneck_attention else config.hidden_size, self.all_head_size
        )
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = tuple(x.shape[:-1]) + (self.num_attention_heads, self.attention_head_size)
        x = x.reshape(new_x_shape)
        return x.transpose((0, 2, 1, 3))

    def forward(
        self,
        query_tensor,
        key_tensor,
        value_tensor,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
    ):
        mixed_query_layer = self.query(query_tensor)
        mixed_key_layer = self.key(key_tensor)
        mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        # attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        perm = list(range(key_layer.dim()))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        attention_scores = paddle.matmul(query_layer, key_layer.transpose(perm))  # bad matmul
        # todo: check transpose behavior.

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.

        attention_probs = F.softmax(attention_scores, axis=-1)  # replaced nn.Softmax() with Functional call

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose((0, 2, 1, 3)) # .contiguous()

        new_context_layer_shape = tuple(context_layer.shape[:-2]) + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


