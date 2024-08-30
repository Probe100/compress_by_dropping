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
from transformers.models.phi.modeling_phi import PhiPreTrainedModel
from typing import List, Optional, Tuple


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

        similarities = torch.full((num_layers,), -math.inf, device=device)
        
        for i in tqdm(range(num_layers), desc='Calculating Similarity'):
            torch.cuda.empty_cache()
            layer = layers[i]
            layer_wrapper = HiddenStatesRecordWrapper(layer, record_input=True, record_output=True)

            # Forward hook for recording hidden states
            def record_states_hook(_, input, output):
                layer_wrapper.record(input[0].data, output[0].data)

            # Get hidden states
            handles = []
            handles.append(layer.register_forward_hook(record_states_hook))
            for j in range(n_calib):
                outputs[j] = layer(inputs[j], attention_mask=attention_mask[j], position_ids=position_ids[j])[0]
            for handle in handles:
                handle.remove()

            input_hidden_states = torch.cat(layer_wrapper.input_hidden_states, dim=0).to(dtype).to(device)
            output_hidden_states = torch.cat(layer_wrapper.output_hidden_states, dim=0).to(dtype).to(device)

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


def get_dropped_layer_id(drop_n, cache_path, cache_filename, model, tokenizer, dataloader: DataLoader, n_calib: int):
    drop_n = drop_n

    device = model.device
    dtype = model.config.torch_dtype

    similarities = get_layer_similarities(model, tokenizer, dataloader, n_calib, cache_path, cache_filename, device, dtype)

    _, sorted_layer_id = torch.sort(similarities, dim=0, descending=True)

    dropped_layers = sorted_layer_id[:drop_n].tolist()
    print(f"Attention should be dropped in {dropped_layers}")
    return dropped_layers

def save_model(pruned_model_save_path, model, tokenizer, dropped_layers: List):
    layers = model.model.layers
    layer_idx_change = 0

    for i in dropped_layers:
        del layers[i-layer_idx_change]
        layer_idx_change += 1
    model.config.num_hidden_layers = model.config.num_hidden_layers - len(dropped_layers)
    # reset layer index
    for i in range(model.config.num_hidden_layers):
        model.model.layers[i].self_attn.layer_idx = i

    model.save_pretrained(pruned_model_save_path)
    tokenizer.save_pretrained(pruned_model_save_path)
