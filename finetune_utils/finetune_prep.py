from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig


def prompt_no_input(row):
    return ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:\n{output}").format_map(row)

def prompt_input(row):
    return ("Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}").format_map(row)

def create_alpaca_prompt(row):
    return prompt_no_input(row) if row["input"] == "" else prompt_input(row)

# prepare dataset for finetuning
# Input: name of dataset
# Output: train dataset and validation dataset
def get_dataset(dataset_name:str, val_size=0.05):
    dataset = load_dataset(dataset_name)
    if "alpaca" in dataset_name:
        dataset = dataset['train'].train_test_split(test_size=val_size)
    return dataset['train'], dataset['test']

def config_prep(dataset_name:str):
    if "alpaca" in dataset_name:
        training_args = SFTConfig(
            report_to="wandb",
            per_device_train_batch_size=2,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            num_train_epochs=3,
            packing=True,
            gradient_accumulation_steps=2, # simulate larger batch sizes
            output_dir="ft/",
            eval_strategy="steps",
            eval_steps=50,
        )
        peft_config = LoraConfig(
            r=16,  # the rank of the LoRA matrices
            lora_alpha=8, # the weight
            lora_dropout=0.1, # dropout to add to the LoRA layers
            bias="none", # add bias to the nn.Linear layers?
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # the name of the layers to add LoRA
            modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
        )
        return training_args, peft_config
    else:
        raise NotImplementedError