# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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

import copy
import io
import json
import os
import six
import unicodedata

from ..bert.tokenizer import BertTokenizer


__all__ = ['MobileBertTokenizer']


class MobileBertTokenizer(BertTokenizer):
    pretrained_resource_files_map = {
        "vocab_file": {
            "mobilebert-uncased":
                "https://huggingface.co/google/mobilebert-uncased/resolve/main/vocab.txt"
        }
    }

    pretrained_init_configuration = {
        "mobilebert-uncased": {
            "do_lower_case": True
        }
    }