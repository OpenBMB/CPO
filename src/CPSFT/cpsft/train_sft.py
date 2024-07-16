# This file is the CPSFT training process.
# Author: Yiju Guo
# Date: 2024-01
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import sys
import torch
import transformers
from datasets import load_dataset
# from peft import (
#     LoraConfig,
#     get_peft_model,
# )
import json
from transformers import AutoModelForCausalLM, AutoTokenizer  
from utils.prompter import Prompter
from dataclasses import dataclass, field

os.environ["TOKENIZERS_PARALLELISM"] = "false"  


@dataclass
class ModelArguments:
    base_model: str = field(default="mistral-7b-hf")  
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='["q_proj","v_proj"]')


@dataclass
class DataArguments:
    data_path: str = field(default="helpful.json")
    prompt_template_name: str = field(default="mistral")
    train_on_inputs: bool = field(default=True)
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=8192)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=32)
    gradient_checkpointing: bool = field(default=True)
    warmup_steps: int = field(default=100)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=1e-5)
    bf16: bool = field(default=True)  
    logging_steps: int = field(default=10)
    val_set_size: int = field(default=500)
    save_strategy: str = field(default="steps")
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=100)  
    save_steps: int = field(default=100)  
    output_dir: str = field(default="/data/checkpoints/")
    save_total_limit: int = field(default=10)
    group_by_length: bool = field(default=False)

class CustomTrainer(transformers.Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"
       
        # Save model checkpoint
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        TRAINER_STATE_NAME = "trainer_state.json"

        
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            logger.warning(
                f"Checkpoint destination directory {output_dir} already exists and is non-empty."
                "Saving will proceed but saved results may be invalid."
            )
            staging_output_dir = output_dir
        else:
            staging_output_dir = os.path.join(
                run_dir, f"tmp-{checkpoint_folder}")
        self.save_model(staging_output_dir, _internal_call=True)

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(staging_output_dir)
            # Save RNG state
            self._save_rng_state(staging_output_dir)

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(
                staging_output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(staging_output_dir)

        # Place checkpoint in final location after all saving is finished.
        # First wait for everyone to finish writing
        self.args.distributed_state.wait_for_everyone()
        # Then go through the rewriting process starting on process 0
        try:
            if staging_output_dir != output_dir:
                with self.args.main_process_first(
                    desc="Renaming model checkpoint folder to true location", local=self.args.save_on_each_node
                ):
                    if os.path.exists(staging_output_dir):
                        os.rename(staging_output_dir, output_dir)

            # Maybe delete some older checkpoints.
            if self.args.should_save:
                self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)
        except Exception:
            print("Error rotating checkpoints skipping")
            pass

def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    train_args.evaluation_strategy = "steps" if train_args.val_set_size > 0 else "no"
    model_args.lora_target_modules = json.loads(model_args.lora_target_modules)

    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {model_args.base_model}\n"
        f"data_path: {data_args.data_path}\n"
        f"output_dir: {train_args.output_dir}\n"
        f"batch_size: {train_args.per_device_train_batch_size*train_args.gradient_accumulation_steps}\n"
        f"micro_batch_size: {train_args.per_device_train_batch_size}\n"
        f"num_epochs: {train_args.num_train_epochs}\n"
        f"learning_rate: {train_args.learning_rate}\n"
        f"cutoff_len: {data_args.cutoff_len}\n"
        f"val_set_size: {train_args.val_set_size}\n"
        f"lora_r: {model_args.lora_r}\n"
        f"lora_alpha: {model_args.lora_alpha}\n"
        f"lora_dropout: {model_args.lora_dropout}\n"
        f"lora_target_modules: {model_args.lora_target_modules}\n"
        f"train_on_inputs: {data_args.train_on_inputs}\n"
        f"add_eos_token: {data_args.add_eos_token}\n"
        f"group_by_length: {train_args.group_by_length}\n"
        f"prompt template: {data_args.prompt_template_name}\n"
    )
    assert (
        model_args.base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(data_args.prompt_template_name)  # 构建对话模板生成类

    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)  # 构建Tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        # torch_dtype=torch.bfloat16, 
        # use_flash_attention_2=True,
        # device_map="auto",  
    )  

    tokenizer.pad_token_id = (
        0  
    )

    tokenizer.padding_side = "left"  

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,  
            max_length=data_args.cutoff_len,
            padding=False,  
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < data_args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()  
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )  
        tokenized_full_prompt = tokenize(full_prompt)
        if not data_args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=data_args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if data_args.add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt

    # config = LoraConfig(  # Lora
    #     r=model_args.lora_r,
    #     lora_alpha=model_args.lora_alpha,
    #     target_modules=model_args.lora_target_modules,  
    #     lora_dropout=model_args.lora_dropout,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    # model = get_peft_model(model, config)  

    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    if train_args.val_set_size > 0:  
        train_val = data["train"].train_test_split(
            test_size=train_args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        )
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=train_args,
        data_collator=transformers.DataCollatorForSeq2Seq(  
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)  

    trainer.train()

    model.save_pretrained(train_args.output_dir)
    tmp_dir = os.path.join(train_args.output_dir, "x.success")
    with open(tmp_dir, 'w', encoding='utf8') as f:
        f.write("hack2\n")


if __name__ == "__main__":
    train()



