import paddle
from paddle import nn

from .nonorm import NoNorm
from .config import MobileBertConfig
from .pretrained_model import PreTrainedModel

from typing import Tuple

from .monkey import Linear as _Linear
nn.Linear = _Linear


class MobileBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MobileBertConfig
    base_model_prefix = "mobilebert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            norm_data = paddle.normal(mean=0.0, std=self.config.initializer_range, shape=module.weight.shape)
            module.weight.set_value(norm_data)
            if module.bias is not None:
                zero_data = paddle.zeros_like(module.bias, dtype=module.bias.dtype)
                module.bias.set_value(zero_data)

        elif isinstance(module, nn.Embedding):
            norm_data = paddle.normal(mean=0.0, std=self.config.initializer_range,
                                      shape=module.weight.shape)
            module.weight.set_value(norm_data)
            if not (module._padding_idx is None):
                module.weight[module._padding_idx] = 0  # Hope this won't be a problem ...

        elif isinstance(module, (nn.LayerNorm, NoNorm)):
            zero_data = paddle.zeros_like(module.bias, dtype=module.bias.dtype)
            module.bias.set_value(zero_data)
            one_data = paddle.ones_like(module.weight, dtype=module.weight.dtype)
            module.weight.set_value(one_data)

    def get_extended_attention_mask(self, attention_mask: paddle.Tensor, input_shape: Tuple[int, ...]) -> paddle.Tensor:
        """
        Makes broadcast-able attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = paddle.arange(seq_length)
                causal_mask = seq_ids.unsqueeze(0).unsqueeze(0).tile((batch_size, seq_length, 1)) <= seq_ids.unsqueeze(0).unsqueeze(-1)
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.astype(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = paddle.concat(
                        [
                            paddle.ones(
                                (batch_size, seq_length, prefix_seq_len), dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask.unsqueeze(1) * attention_mask.unsqueeze(1).unsqueeze(1)
            else:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.astype(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

