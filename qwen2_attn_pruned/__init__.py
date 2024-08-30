# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)
# from ...utils import (
#     OptionalDependencyNotAvailable,
#     _LazyModule,
#     is_tokenizers_available,
#     is_torch_available,
# )


_import_structure = {
    "configuration_qwen2_attn_pruned": ["Qwen2AttnPrunedConfig"],
    "tokenization_qwen2_attn_pruned": ["Qwen2AttnPrunedTokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_qwen2_attn_pruned_fast"] = ["Qwen2AttnPrunedTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qwen2_attn_pruned"] = [
        "Qwen2AttnPrunedForCausalLM",
        "Qwen2AttnPrunedModel",
        "Qwen2AttnPrunedPreTrainedModel",
        "Qwen2AttnPrunedForSequenceClassification",
        "Qwen2AttnPrunedForTokenClassification",
    ]


if TYPE_CHECKING:
    from .configuration_qwen2_attn_pruned import Qwen2AttnPrunedConfig
    from .tokenization_qwen2_attn_pruned import Qwen2AttnPrunedTokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_qwen2_attn_pruned_fast import Qwen2AttnPrunedTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2_attn_pruned import (
            Qwen2AttnPrunedForCausalLM,
            Qwen2AttnPrunedForSequenceClassification,
            Qwen2AttnPrunedForTokenClassification,
            Qwen2AttnPrunedModel,
            Qwen2AttnPrunedPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)