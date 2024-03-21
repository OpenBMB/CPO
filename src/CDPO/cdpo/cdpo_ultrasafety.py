# This file is script to perform CPO on UltraSafety dataset
# Author: Bowen Sun, Yiju Guo
# Date: 2024-03
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

import json
import random
import argparse
from typing import List, Dict
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchDPOTrainer

#Part 1: data preprocessing logic
def preprocess_data(srcpath):
    def readJsonFile(filePath: str, encoding="utf-8") -> dict:
        with open(filePath, "r", encoding=encoding) as file:
            return json.load(file)

    def SampleTargetResponses(responses: List[dict], cfg: dict):
        output: List[dict] = []
        for r in responses:
            valid = True
            for cfg_key in cfg:
                if cfg_key not in DATA_KEYS:
                    continue

                rate = cfg.get(cfg_key)
                if rate is None:
                    continue

                keys = DATA_KEYS.get(cfg_key)
                cur_rate = None
                for key in keys:
                    if key in r:
                        cur_rate = int(r.get(key))
                        break

                if cur_rate != rate:
                    valid = False
                    break

            if not valid:
                continue

            output.append(r)

        return output
    def changeToJson(obj,isFormat)->str:
        if(not isFormat):
            return json.dumps(obj,ensure_ascii=False, separators=(',', ':'))    
        else:
            return json.dumps(obj,indent=4, sort_keys=True, separators=(',', ':'),ensure_ascii =False)
    def ModifyInstruction(Helpfulness_Rating:int, Honesty_Rating:int, Harmlessness_Rating:int, result:dict):
        text = ""
        if Helpfulness_Rating != None:
            text = text + f"< helplessness: {Helpfulness_Rating} > "
        if Honesty_Rating != None:
            text = text + f"< honesty: {Honesty_Rating} > "
        if Harmlessness_Rating != None:
            text = text + f"< harmlessness: {Harmlessness_Rating} > "
        text = text 
        result["instruction"] = text + result["instruction"]
    def GetRateByKey(response:dict,keys:List[str]):
        for key in keys:
            if response.get(key):
                return int(response.get(key))

    DATA_KEYS = {
        "Help": ["Helpful_Rating", "Helpfulness_Rating"],
        "Honesty": ["Honesty_Rating"],
        "Harmless": ["Harmlessness_Rating"],
    }

    HAS_HARMLESS = False

    RANDOM_CFG = [
        {
            "r1_enable": False,
            "r2_enable": False,
            "random_range": {
                "1": {
                    "max_count": 8966,
                    "Harmless": 1,
                },
                "0": {
                    "max_count": 8966,
                    "Harmless": 0
                }
            },
        },
    ]

    data = readJsonFile(srcpath)
    results = []
    readed: Dict[str, bool] = {}
    for CFG in RANDOM_CFG:
        random_range:Dict[str,dict] = CFG.get("random_range")
        r1_enable:bool = CFG["r1_enable"]
        r2_enable:bool = CFG["r2_enable"]

        for key_name in random_range:
            cfg = random_range[key_name]

            max_count = cfg["max_count"]
            count = 0

            for item in data:
                if count >= max_count:
                    break

                if readed.get(changeToJson(item,False)):
                    continue

                samples:List[dict] = []
                samples = SampleTargetResponses(item["responses"],cfg)
                if not samples:
                    continue
                sample = random.choice(samples)

                def GetR(response:dict):
                    R3 = HAS_HARMLESS and (-abs(GetRateByKey(response, DATA_KEYS["Harmless"])-int(GetRateByKey(sample, DATA_KEYS["Harmless"])))) or 0
                    R1 = r1_enable and -abs(GetRateByKey(response,DATA_KEYS["Help"]) - int(GetRateByKey(sample,DATA_KEYS["Help"]))) or int(GetRateByKey(response,DATA_KEYS["Help"]))
                    R2 = r2_enable and -abs(GetRateByKey(response,DATA_KEYS["Honesty"])) - int(GetRateByKey(sample, DATA_KEYS["Honesty"])) or GetRateByKey(response, DATA_KEYS["Honesty"])
                    return R3+R1+R2

                instruction = item["instruction"]
                responses = item["responses"]
                for i in range(len(responses)):
                    for j in range(i+1,len(responses)):
                        if count >= max_count:
                            break
                        response_i = responses[i]
                        response_j = responses[j]

                        R_i = GetR(response_i)
                        R_j = GetR(response_j)

                        # 比较R值
                        if R_i > R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": response_i["response"],
                                "reject": response_j["response"]
                            }
                        elif R_i < R_j:
                            result = {
                                "instruction": instruction,
                                "chosen": response_j["response"],
                                "reject": response_i["response"]
                            }
                            temp = R_i
                            R_i = R_j
                            R_j = temp
                        else:
                            continue
                        result['R_chosen'] = R_i
                        result['R_reject'] = R_j
                        ModifyInstruction(r1_enable and GetRateByKey(sample, DATA_KEYS["Help"]) or None, r2_enable and GetRateByKey(sample, DATA_KEYS["Honesty"]) or None, GetRateByKey(sample, DATA_KEYS["Harmless"]), result)
                        results.append(result)
                        count = count+1

            #print(f"r1_enable:{r1_enable}, r2_enable:{r2_enable}的{key_name}的{cfg}采了{count}条")
    return results

# Part 2: Further processing the dataset
def further_process_data(data):
    processed_data = []
    for item in data:
        item.pop('R_chosen', None)
        item.pop('R_reject', None)
        if 'instruction' in item:
            item['instruction'] = item['instruction'].replace('< harmlessness: 1 > ', '')
            item['prompt'] = item.pop('instruction')
        if 'reject' in item:
            item['rejected'] = item.pop('reject')
        processed_data.append(item)
    return processed_data

# Part 3: Training
def train_model(data):
    PatchDPOTrainer()
    with open('paths_dpo.json', 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    train_dataset = load_dataset("json", data_files="paths_dpo.json", split='train')

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="model",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=64,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=2048,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            warmup_ratio=0.1,
            num_train_epochs=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            seed=42,
            output_dir="./outputs",
        ),
        beta=0.1,
        train_dataset=train_dataset,
        # eval_dataset = YOUR_DATASET_HERE,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )
    dpo_trainer.train()
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")


if __name__ == '__main__':
    srcpath = './dpo_UltraSafety.json'
    preprocessed_data = preprocess_data(srcpath)
    processed_data = further_process_data(preprocessed_data)
    train_model(processed_data)
