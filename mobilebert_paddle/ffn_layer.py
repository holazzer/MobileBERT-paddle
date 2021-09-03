from paddle import nn

from .mobile_bert_intermediate import MobileBertIntermediate
from .ffn_out import FFNOutput


class FFNLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.intermediate = MobileBertIntermediate(config)
        self.output = FFNOutput(config)

    def forward(self, hidden_states):
        intermediate_output = self.intermediate(hidden_states)
        layer_outputs = self.output(intermediate_output, hidden_states)
        return layer_outputs


