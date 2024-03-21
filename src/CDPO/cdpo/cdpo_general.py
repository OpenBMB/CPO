# This file is an example of a DPO implementation using the TRL framework
# Author: Bowen Sun
# Date: 2024-03
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset

train_dataset = load_dataset("json", data_files="demo.json", split='train')

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/home/model",      # Local 4bit-mistral model
    max_seq_length = 2048,
    dtype = None,       # Automatically determines whether BF16 is enabled
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0,       # Supports any, but = 0 is optimized
    bias = "none",      # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = 2048,
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "./outputs",
    ),
    beta = 0.1,
    train_dataset = train_dataset,
    # eval_dataset = YOUR_DATASET_HERE,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()

# Save the model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
