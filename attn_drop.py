import os
import torch
import torch.nn.functional as F
import math
from torch import no_grad
from torch.utils.data import DataLoader
from tqdm import tqdm

from wrapper import HiddenStatesRecordWrapper
from utils import prepare_calibration_input
from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.phi.modeling_phi import PhiPreTrainedModel
from typing import List, Optional, Tuple

from qwen2_attn_pruned.configuration_qwen2_attn_pruned import Qwen2AttnPrunedConfig
from qwen2_attn_pruned.modeling_qwen2_attn_pruned import Qwen2AttnPrunedForCausalLM
from phi_attn_pruned.configuration_phi_attn_pruned import PhiAttnPrunedConfig
from phi_attn_pruned.modeling_phi_attn_pruned import PhiAttnPrunedForCausalLM

@no_grad()
def get_layer_similarities(model, tokenizer, dataloader: DataLoader, n_calib: int, cache_path, cache_filename, device, dtype):
    # use cached similarity if file exists
    cache_file = os.path.join(cache_path, cache_filename)
    if os.path.join(cache_file) is not None and os.path.exists(cache_file):
        similarities = torch.load(cache_file, map_location=device)
    else:
        # get attention layer id
        layers = model.model.layers
        inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, tokenizer, dataloader, n_calib) 

        num_layers = model.config.num_hidden_layers
        layer_idx = list(range(num_layers))
        for i in tqdm(range(num_layers)):
            torch.cuda.empty_cache()
            layer = layers[i]

        similarities = torch.full((num_layers,), -math.inf, device=device)
        
        for i in tqdm(range(num_layers), desc='Calculating Similarity'):
            layer = layers[i]
            input_layernorm = layer.input_layernorm
            post_attention_layernorm = layer.post_attention_layernorm
            input_layernorm_wrapper = HiddenStatesRecordWrapper(layer.input_layernorm, record_input=True, record_output=False)
            post_attention_layernorm_wrapper = HiddenStatesRecordWrapper(layer.post_attention_layernorm, record_input=True, record_output=False)

            # Forward hook for recording hidden states
            def record_input_states_hook(_, input, output):
                input_layernorm_wrapper.record(input[0].data, output[0].data)

            def record_output_states_hook(_, input, output):
                post_attention_layernorm_wrapper.record(input[0].data, output[0].data)

            # Get hidden states
            handles = []
            handles.append(input_layernorm.register_forward_hook(record_input_states_hook))
            handles.append(post_attention_layernorm.register_forward_hook(record_output_states_hook))
            for j in range(n_calib):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            for handle in handles:
                handle.remove()

            input_hidden_states = torch.cat(input_layernorm_wrapper.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(post_attention_layernorm_wrapper.input_hidden_states, dim=0).to(dtype).to(device)

            cos_sim = F.cosine_similarity(input_hidden_states, output_hidden_states, dim=-1)  # (total_token_num)
            cos_sim = cos_sim.mean()
            similarities[i] = cos_sim

            inputs, outputs = outputs, inputs

        if cache_path is not None:
            if not os.path.isdir(cache_path):
                os.mkdir(cache_path)
            torch.save(similarities.clone().cpu(), os.path.join(cache_path,cache_filename))
            print(f"Saving cached similarities to {cache_path}")
    
    for idx, s in enumerate(similarities):
        print(f"layer: {idx}, similarity: {s}")

    return similarities

@no_grad()
def get_layer_similarities_phi(model, tokenizer, dataloader: DataLoader, n_calib: int, cache_path, cache_filename, device, dtype):
    # use cached similarity if file exists
    cache_file = os.path.join(cache_path, cache_filename)
    if os.path.join(cache_file) is not None and os.path.exists(cache_file):
        similarities = torch.load(cache_file, map_location=device)
    else:
        # get attention layer id
        layers = model.model.layers
        inputs, outputs, attention_mask, position_ids = prepare_calibration_input(model, tokenizer, dataloader, n_calib) 

        num_layers = model.config.num_hidden_layers
        layer_idx = list(range(num_layers))

        similarities = torch.full((num_layers,), -math.inf, device=device)
        
        for i in tqdm(range(num_layers), desc='Calculating Similarity'):
            torch.cuda.empty_cache()
            layer = layers[i]
            input_layernorm = layer.input_layernorm
            self_attn = layer.self_attn
            mlp = layer.mlp
            
            input_layernorm_wrapper = HiddenStatesRecordWrapper(input_layernorm, record_input=True, record_output=False)
            self_attn_wrapper = HiddenStatesRecordWrapper(self_attn, record_input=False, record_output=True)
            mlp_wrapper = HiddenStatesRecordWrapper(mlp, record_input=False, record_output=True)



            # Forward hook for recording hidden states
            def record_input_states_hook(_, input, output):
                input_layernorm_wrapper.record(input[0].data, output[0].data)

            def record_self_attn_output_states_hook(_, input, output):
                self_attn_wrapper.record(input, output[0].data)
            
            def record_mlp_output_states_hook(_, input, output):
                mlp_wrapper.record(input[0].data, output[0].data)

            # Get hidden states
            handles = []
            handles.append(input_layernorm.register_forward_hook(record_input_states_hook))
            handles.append(self_attn.register_forward_hook(record_self_attn_output_states_hook))
            handles.append(mlp.register_forward_hook(record_mlp_output_states_hook))
            for j in range(n_calib):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            for handle in handles:
                handle.remove()

            input_hidden_states = torch.cat(input_layernorm_wrapper.input_hidden_states, dim=0).to(dtype).to(device)
            self_attn_output_hidden_states = torch.cat(self_attn_wrapper.output_hidden_states, dim=0).to(dtype).to(device)
            mlp_output_hidden_states = torch.cat(mlp_wrapper.output_hidden_states, dim=0).to(dtype).to(device)

            cos_sim = F.cosine_similarity(input_hidden_states+mlp_output_hidden_states, input_hidden_states+mlp_output_hidden_states+self_attn_output_hidden_states, dim=-1)  # (total_token_num)
            cos_sim = cos_sim.mean()
            similarities[i] = cos_sim

            inputs, outputs = outputs, inputs

        if cache_path is not None:
            if not os.path.isdir(cache_path):
                os.mkdir(cache_path)
            torch.save(similarities.clone().cpu(), os.path.join(cache_path,cache_filename))
            print(f"Saving cached similarities to {cache_path}")
    
    for idx, s in enumerate(similarities):
        print(f"layer: {idx}, similarity: {s}")

    return similarities


def get_dropped_layer_id(drop_n, cache_path, cache_filename, model, tokenizer, dataloader: DataLoader, n_calib: int):
    drop_n = drop_n

    device = model.device
    dtype = model.config.torch_dtype

    if isinstance(model, PhiPreTrainedModel):
        similarities = get_layer_similarities_phi(model, tokenizer, dataloader, n_calib, cache_path, cache_filename, device, dtype)
    else:
        similarities = get_layer_similarities(model, tokenizer, dataloader, n_calib, cache_path, cache_filename, device, dtype)

    _, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layers = sorted_layer_id[:drop_n].tolist()
    print(f"Attention should be dropped in {dropped_layers}")
    return dropped_layers

def save_model(pruned_model_save_path, model, tokenizer, dropped_layers: List):
    layers = model.model.layers
    layer_idx_change = 0
    attn_drop_layers = []
    for i in range(model.config.num_hidden_layers):
        if isinstance(model, PhiPreTrainedModel):
            if i in dropped_layers:
                bound_method = forward_without_attn_phi.__get__(layers[i], layers[i].__class__)
                setattr(layers[i], 'forward', bound_method)
                delattr(layers[i], 'self_attn')
                layer_idx_change += 1
                attn_drop_layers.append(True)
            else:
                model.model.layers[i].self_attn.layer_idx -= layer_idx_change
                attn_drop_layers.append(False)
        else:
            if i in dropped_layers:
                bound_method = forward_without_attn.__get__(layers[i], layers[i].__class__)
                setattr(layers[i], 'forward', bound_method)
                delattr(layers[i], 'self_attn')
                delattr(layers[i], 'input_layernorm')
                layer_idx_change += 1
                attn_drop_layers.append(True)
            else:
                model.model.layers[i].self_attn.layer_idx -= layer_idx_change
                attn_drop_layers.append(False)

    config_dict = model.config.to_dict()
    if isinstance(model, Qwen2PreTrainedModel):
        # init config for the pruned model
        config_dict['architectures'] = ['Qwen2AttnPrunedForCausalLM']
        config_dict['_name_or_path'] = pruned_model_save_path
        config_dict['model_type'] = 'qwen2_attn_pruned'
        config_dict['attn_drop_layers'] = attn_drop_layers
        prunedConfig = Qwen2AttnPrunedConfig().from_dict(config_dict)

        # init pruned model and load weights
        pruned_model = Qwen2AttnPrunedForCausalLM(prunedConfig)
        pruned_model.model.load_state_dict(model.model.state_dict())
        pruned_model.lm_head.load_state_dict(model.lm_head.state_dict())

        Qwen2AttnPrunedConfig.register_for_auto_class()
        Qwen2AttnPrunedForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    # elif isinstance(model, LlamaPreTrainedModel):
    #     config_dict['architectures'] = ['LlamaAttnPrunedForCausalLM']
    #     config_dict['_name_or_path'] = pruned_model_save_path
    #     config_dict['model_type'] = 'llama_attn_pruned'
    #     config_dict['attn_drop_layers'] = attn_drop_layers
    #     prunedConfig = LlamaAttnPrunedConfig().from_dict(config_dict)

    #     # init pruned model and load weights
    #     pruned_model = LlamaAttnPrunedForCausalLM(prunedConfig)
    #     pruned_model.model.load_state_dict(model.model.state_dict())
    #     pruned_model.lm_head.load_state_dict(model.lm_head.state_dict())

    #     LlamaAttnPrunedConfig.register_for_auto_class()
    #     LlamaAttnPrunedForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    elif isinstance(model, PhiPreTrainedModel):
        config_dict['architectures'] = ['PhiAttnPrunedForCausalLM']
        config_dict['_name_or_path'] = pruned_model_save_path
        config_dict['model_type'] = 'phi_attn_pruned'
        config_dict['attn_drop_layers'] = attn_drop_layers
        prunedConfig = PhiAttnPrunedConfig().from_dict(config_dict)

        # init pruned model and load weights
        pruned_model = PhiAttnPrunedForCausalLM(prunedConfig)
        pruned_model.model.load_state_dict(model.model.state_dict())
        pruned_model.lm_head.load_state_dict(model.lm_head.state_dict())

        PhiAttnPrunedConfig.register_for_auto_class()
        PhiAttnPrunedForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    else:
        raise TypeError("Model type not supported")
    # save model and tokenizer
    pruned_model.save_pretrained(pruned_model_save_path)
    tokenizer.save_pretrained(pruned_model_save_path)

def forward_without_attn(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        # Fully Connected
        device = hidden_states.device
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states).to(device)
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if use_cache:
            outputs += (past_key_value,)
        

        return outputs

def forward_without_attn_phi(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range
                `[0, config.n_positions - 1]`. [What are position IDs?](../glossary#position-ids)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)


        feed_forward_hidden_states = self.resid_dropout(self.mlp(hidden_states))
        hidden_states = feed_forward_hidden_states + residual
        outputs = (hidden_states,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs