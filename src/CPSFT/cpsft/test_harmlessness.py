# This file is a CPSFT harmlessness test.
# Author: Yiju Guo
# Date: 2024-01
# Copyright (c) RUCBM, Renmin University of China. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import sys
import fire
import torch
# from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
from utils.prompter import Prompter
import json
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    base_model: str = '/path',
    # lora_weights: str = "./lora-checkpoint/checkpoint-800",
    prompt_template: str = "mistral",  
    data_dir:str = '/path',
    output_dir: str = '/path',
    data_dir_1:str = '/path',
    output_dir_1: str = '/path',
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data/user/tc_agi/ylf/mistral-7b-instruct-v0.2")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,  
            device_map="auto",
        )
        # model = PeftModel.from_pretrained(
        #     model,
        #     lora_weights,
        #     torch_dtype=torch.float16,
        # )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)  

    @torch.inference_mode()
    def evaluate(
        model,
        instruction,
        input=None,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                top_p=top_p,  
                top_k=top_k,    
                num_beams=num_beams,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)  
        return prompter.get_response(output)  

    data = []
    print(data_dir)
   
    with open(data_dir, "r") as file:
        json_data = json.load(file)
    id = 0
    
    for entry in json_data:
        instruction = entry["instruction"]
        prompt_level = entry["prompt_level"]
        instruction_level = entry["instruction_level"]
        print("Instruction:", instruction)
        print("-" * 16 + "origin model" + "-" * 16)
        
        response = evaluate(model, instruction)
        print("Response:", response)
        print("id", id)
       
        entry = {"id":id, "instruction": instruction, "prompt_level":prompt_level,"instruction_level":instruction_level, "response": response}
        data.append(entry)
        id+=1

    with open(output_dir, "w") as file:
        json.dump(data, file, indent=4)

    data = []
    print(data_dir_1)

    with open(data_dir_1, "r") as file:
        json_data = json.load(file)
    id = 0

    for entry in json_data:
        instruction = entry["instruction"]
        prompt_level = entry["prompt_level"]
        instruction_level = entry["instruction_level"]
        print("Instruction:", instruction)
        print("-" * 16 + "origin model" + "-" * 16)

        response = evaluate(model, instruction)
        print("Response:", response)
        print("id", id)

        entry = {"id":id, "instruction": instruction, "prompt_level":prompt_level,"instruction_level":instruction_level, "response": response}
        data.append(entry)
        id+=1

    # 将数据保存到JSON文件
    with open(output_dir_1, "w") as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    fire.Fire(main)



