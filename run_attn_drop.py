import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["https_proxy"]="http://10.10.20.100:1089"
os.environ['HF_HOME'] = "/mnt/public/hanling/cache"
os.environ['TRANSFORMERS_CACHE'] = "/mnt/public/hanling/cache"
os.environ['HF_DATASETS_CACHE'] = "/mnt/public/hanling/dataset_cache"

import argparse
from tqdm import tqdm
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from attn_drop import get_dropped_layer_id, save_model

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--n_calib", type=int, default=4)
    parser.add_argument("--n_drop", type=int, default=4)
    parser.add_argument("--cache_path", type=str, default="output/")
    parser.add_argument("--cache_filename", type=str, default="similarity.pt")
    parser.add_argument("--model_save_path", type=str, default="models/Qwen2-7B-AD4")

    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto",
        device_map="auto"
    ) 
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    data = load_dataset("emozilla/pg19", split="validation")  #sample 10,000 texts to compute block influences
    dataloader = DataLoader(
        data,
        batch_size=1,
        shuffle=True,
    )

    dropped_layers = get_dropped_layer_id(args.n_drop, args.cache_path, args.cache_filename, model, tokenizer, dataloader, args.n_calib)
    if args.model_save_path != "":
        save_model(args.model_save_path, model, tokenizer, dropped_layers)
    else:
        print("Invalid path, failed to save the pruned model")

if __name__ == "__main__":
    main()