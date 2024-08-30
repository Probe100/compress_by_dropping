# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import os
import wandb
from peft import get_peft_model
# from syne_tune import Reporter

from finetune_utils.finetune_prep import get_dataset, config_prep, create_alpaca_prompt
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["https_proxy"]="http://10.10.20.100:1089"
os.environ['HF_HOME'] = "/mnt/public/hanling/cache"
os.environ['TRANSFORMERS_CACHE'] = "/mnt/public/hanling/cache"
os.environ['HF_DATASETS_CACHE'] = "/mnt/public/hanling/dataset_cache"
os.environ["WANDB_PROJECT"] = "alpaca_ft"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# TODO: add unsloth 

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen2-7B-BD6")
    parser.add_argument("--dataset", type=str, default="llamafactory/alpaca_en")
    parser.add_argument("--use_lora", type=bool, default=True)
    parser.add_argument("--use_unsloth", type=bool, default=True)

    args = parser.parse_args()

    if args.use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(args.model)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)

    train_dataset, eval_dataset = get_dataset(args.dataset)

    training_args, peft_config = config_prep(args.dataset)

    if not args.use_lora:
        peft_config = None
    else:
        if args.use_unsloth:
            peft_model = FastLanguageModel.get_peft_model(kwargs=peft_config)
        else:
            peft_model = get_peft_model(model, peft_config)
            peft_model.print_trainable_parameters()

    trainer = SFTTrainer(
        peft_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        max_seq_length=1024, # maximum packed length 
        args=training_args,
        formatting_func=create_alpaca_prompt, # format samples with a model schema
        peft_config = peft_config
    )
    trainer.train()

if __name__ == "__main__":
    main()