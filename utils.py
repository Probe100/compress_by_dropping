from torch import nn, no_grad
from transformers.models.phi.modeling_phi import PhiPreTrainedModel

@no_grad()
def prepare_calibration_input(model, tokenizer, dataloader, num_samples=16):
    layers = model.model.layers
    cache = {'inputs': [], 'attention_mask': [], "position_ids": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, input, **kwargs):
            cache['inputs'].append(input)
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_ids'].append(kwargs['position_ids'])
            raise ValueError

    # catch inputs of first layer
    layers[0] = Catcher(layers[0])
    for index, batch in enumerate(dataloader):
        if index >= num_samples:
            break
        try:
            prompts = batch['text']
            if isinstance(model, PhiPreTrainedModel):
                tokenizer.pad_token = tokenizer.eos_token
            prompt_tokens = tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt')
            input_ids = prompt_tokens.input_ids
            attn_mask = prompt_tokens.attention_mask
            model(input_ids=input_ids.to(model.device), attention_mask=attn_mask.to(model.device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outputs = [None] * len(cache['inputs'])

    return cache['inputs'], outputs, cache['attention_mask'], cache['position_ids']